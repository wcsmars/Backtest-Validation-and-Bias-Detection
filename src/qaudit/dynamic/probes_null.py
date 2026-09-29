"""Null-distribution probes: re-run the user's pipeline against nulls.

Every check here re-executes ``backtest_func(signals, asset_returns) ->
net strategy-returns pd.Series`` (the callable applies its own execution
lag and costs internally); without that callable each check SKIPs. One
``numpy.random.default_rng(config.seed)`` is created per :func:`run` and
consumed in a fixed order (placebo panels first, then date-shuffle
permutations, then cross-section relabelings), so every number is
deterministic for a given config; the asset-relabeled control that only
the sparse ``placebo_pipeline_bias`` FAIL path runs draws from its own
``default_rng([config.seed, _TILT_CONTROL_STREAM])`` and never shifts that
order. Every user callable receives defensive copies of the input
frames: a mutating ``backtest_func`` can corrupt its own copy but never
the artifacts or a sibling probe's inputs.

Exposure-aware nulls: raw Sharpes carry whatever net market exposure a
book has, and a null book's exposure is not the real book's by
construction. A date-row shuffle preserves each asset's sample mean and a
fully-invested placebo book earns the equity premium, so raw null Sharpes
cannot distinguish honest beta carry from an engine bug; conversely a
pipeline that maps signal levels into weights (a long-only score used as
proportional weights, a signal used directly as target weights) turns the
real positive score into a fully long book but the mean-zero dense
placebo noise into roughly market-neutral books, so a raw ranking would
credit the equity premium to an information-free signal. Every gate
therefore compares active Sharpes: each run's PnL is beta-hedged against
the equal-weight basket of the same returns panel that run consumed
(slope from the regression, intercept kept). ``placebo_pipeline_bias``
gates on the hedged placebo mean; ``placebo_percentile`` ranks the hedged
actual SR among the hedged placebo SRs; ``shuffled_labels`` gates the
hedged actual SR against hedged nulls. Raw Sharpes are reported in every
check's ``details``.

Two null constructions adapt to the inputs. Placebo panels are AR(1)
noise matched to the real signal's persistence for dense signals
(``_signal_phi``: the larger of the per-column lag-1 autocorrelation and
the persistence of the cross-sectional ranking the pipeline trades); for
sparse/event signals (fewer than ``_PLACEBO_SPARSE_ACTIVITY`` of the
non-NaN cells nonzero) they instead keep the complete real signal panel
(values, NaNs, on/off runs and cross-sectional event breadth) intact and
circularly shift every column by one shared random row offset
(``_placebo_panel_sparse``) - an AR(1) panel is dense, trades every bar
and is eaten by costs, which makes a costly sparse book look like it
beats noise. The rotation keeps which assets carry the events, so a
static tilt toward persistently outperforming names survives into every
sparse placebo; the pipeline-bias gate separates it from an engine credit
with an asset-relabeled control (``_sparse_tilt_control``). The
date-shuffle null permutes contiguous blocks of return rows
(``_block_perm``; block length ``config.shuffle_block_len``, auto-derived
from the larger of the panel's median absolute lag-1 return and
squared-return autocorrelations by ``_auto_block_len`` and equal to 1 (a
plain row permutation) when both dependence measures are negligible), so
signed serial correlation cannot cancel across assets and broad
volatility clustering also lengthens the null's blocks.

Checks
------
dynamic.placebo_pipeline_bias
    Placebo signal panels with no alignment to returns (persistence-matched
    AR(1) noise, or the time-rotated real panel for sparse signals) through
    the pipeline with the real asset returns. A reliably positive mean null
    Sharpe *net of market exposure* - or any constant-positive/riskless
    placebo run - means the engine itself manufactures performance; the
    equity premium earned by a beta-carrying book on random signals does
    not. On the sparse path a would-be FAIL is first re-measured on
    asset-relabeled controls: the engine is convicted only when they show
    the bias too, otherwise the verdict is a scoped WARN naming the static
    per-asset event tilt (WARN HIGH when the controls cannot be measured).
    The placebos trade about as often as the real book, so a credit that
    scales with turnover (costs added instead of charged) is measured at
    the size by which it inflates the real book: a small one on a
    low-turnover book stays under ``placebo_null_sharpe_fail``. The static
    cost reconciliation (``costs.missing_transaction_costs``: net above
    gross) is the check for that class.
dynamic.placebo_percentile
    Where the actual Sharpe *net of market exposure* sits inside the
    hedged placebo null distribution (raw figures in ``details``); skipped
    with a pointer at the engine when the null is degenerate
    (constant-positive placebo runs).
dynamic.shuffled_labels
    Two nulls: real signals against date-shuffled returns (each date's
    cross-section preserved, signal/return alignment destroyed) and
    against cross-section-relabeled returns (each asset's return series
    reassigned to another asset, time structure preserved). An accounting
    leak survives both; a persistent-characteristic (static-tilt) book
    survives the date shuffle but collapses under relabeling and is
    reported as a tilt, not a leak. Also a plus-one block-randomization
    p-value for the active edge, conditional on the selected block-null
    approximation (not a universal time-series calibration).
dynamic.probe_health
    Emitted (WARN, LOW) when a null distribution had to be distrusted:
    more than ``_MAX_BAD_FRAC`` of its re-runs returned NaN/degenerate
    Sharpes (a backtest_func defect), or fewer than 2 valid re-runs exist
    (an AuditConfig count of 1; a 1-point null cannot be tested); the
    checks that depend on that distribution then SKIP instead of trusting
    it. Invalid callback return types, dates, or interior missing returns
    are rejected and surfaced as HIGH warnings even below that fraction.
"""
from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np
import pandas as pd

from .._stats import fmt_tstat
from ..config import AuditConfig
from ..inputs import BacktestArtifacts
from ..types import CheckResult, Severity, failed, passed, skipped, warned
from ._pipeline import PipelineOutputError, validated_returns

_CHECK_PLACEBO_BIAS = "dynamic.placebo_pipeline_bias"
_CHECK_PLACEBO_PCT = "dynamic.placebo_percentile"
_CHECK_SHUFFLE = "dynamic.shuffled_labels"
_CHECK_HEALTH = "dynamic.probe_health"
_CHECK_IDS = (_CHECK_PLACEBO_BIAS, _CHECK_PLACEBO_PCT, _CHECK_SHUFFLE)

_BT_CONTRACT = ("backtest_func(signals: pd.DataFrame, asset_returns: "
                "pd.DataFrame) -> net strategy-returns pd.Series")

# If more than this fraction of null re-runs return NaN/degenerate
# Sharpes, the distribution is not trusted.
_MAX_BAD_FRAC = 0.20

# Required t-stat of the placebo null mean for the pipeline-bias FAIL
# path.
_BIAS_TSTAT_FAIL = 3.0

# |annualized SR| beyond this is not a strategy, it is the riskless-credit
# signature: a constant daily credit whose float representation leaves the
# series std at rounding dust yields finite SRs of ~1e16, the same defect as
# an exact +inf, so both are treated as one degenerate class.
_SR_DEGENERATE_CAP = 1e3

# Hedge-residual scale (relative to the PnL/market magnitude) below which
# the residual is float-rounding dust, not information: a PnL exactly
# linear in the basket leaves residuals ~1e-16 * scale.
_RESID_DUST = 1e-9

# The shuffle leak paths only convict when the actual (hedged) SR is
# material; below this the "survives shuffling" ratio is noise on noise.
_LEAK_MIN_ACTUAL_SR = 1.0

# A signal cell counts as active when |x| > this (zero-encoded "no event"
# exports are exactly 0.0; the tolerance absorbs float dust from z-scoring).
_ACTIVE_TOL = 1e-12

# Minimum non-NaN observations a column needs before its lag-1
# autocorrelation enters the median in _signal_phi / _returns_phi. Below it
# the estimate is not a measurement: 2 observations make numpy's cov emit
# "Degrees of freedom <= 0" (a warnings.warn call, fatal under -W error,
# not an errstate condition) and 3 observations contribute exactly +/-1.
# A name alive for a handful of bars in the aligned grid is routine; its
# column is NaN for the median (ignored), like a constant column.
_PHI_MIN_OBS = 20

# Minimum names finite on both dates before a consecutive-date rank
# correlation enters the cross-sectional persistence median of
# _xsec_rank_phi. Rank correlations are discrete in small cross-sections:
# with 3 names they take only the values +/-1 and +/-0.5, so the median
# over the dates of pure iid noise lands at +/-0.5 in about half of all
# samples (measured over 40 seeds at 60-1000 dates) and would fake a
# persistent signal; with 4 names it still reaches 0.4 on 60 dates. From
# 5 names the iid median stays within +/-0.1 once the panel has a year of
# data (+/-0.3 at 60 dates). A narrower panel keeps the per-column
# estimate alone.
_XSEC_MIN_ASSETS = 5

# Signals whose active fraction (nonzero share of the non-NaN cells) is
# below this get structure-matched sparse placebos instead of AR(1) noise.
# Calibration: a continuous alpha export (z-scores, ranks, momentum) is
# nonzero on ~100% of its cells and a gated/event export on 1-30%; the bar
# sits at the midpoint of the empty band between them, so every dense
# export keeps the AR(1) null and every event-style export gets a null
# with its own density. The density match matters: a dense AR(1) placebo
# trades every bar and is eaten by costs, so against it an information-
# free sparse book (a few percent of cells active, multi-bar holds, a
# costed pipeline) passes the percentile most of the time; against
# placebos with its own density it is flagged.
_PLACEBO_SPARSE_ACTIVITY = 0.5

# Stream tag of the asset-relabeled control that confirms a sparse
# placebo_pipeline_bias FAIL (_sparse_tilt_control). It draws from
# default_rng([config.seed, _TILT_CONTROL_STREAM]), an independent stream,
# and only on that would-FAIL path, so every placebo, date-shuffle and
# relabeling draw of the shared stream is unchanged whether it runs or not.
_TILT_CONTROL_STREAM = 0x7117

# A Monte Carlo null with only a handful of draws is too coarse to
# exonerate a strategy, even when a lenient percentile threshold would
# make its best attainable plus-one p-value look significant. Twenty is
# the smallest null whose best attainable p=1/21 is strictly below 0.05;
# stricter configured alpha levels require more.
_MIN_PLACEBO_RANDOMIZATION_RUNS = 20

# Date-shuffle null block policy. A global row permutation
# assumes exchangeable return rows; with serially dependent returns and a
# persistent signal the null's spread understates the actual SR's sampling
# spread. The dependence statistic is the larger of (a) the median across
# assets of absolute lag-1 return autocorrelation and (b) the same statistic
# on squared, scale-normalized returns. Taking absolute values before the
# median prevents positive and negative AR assets from cancelling; the
# squared-return arm responds to broad volatility clustering. Below
# _SHUFFLE_PHI_NEGLIGIBLE a plain permutation is kept. Above it, the block
# length is _SHUFFLE_BLOCK_MULT x the AR(1) variance-ratio
# (1+dependence)/(1-dependence), clipped so at least ~10 blocks are
# permuted. This remains a data-driven block-null approximation rather
# than a proof of exchangeability for every return process; that assumption
# is carried explicitly in each result's details.
_SHUFFLE_PHI_NEGLIGIBLE = 0.10
_SHUFFLE_BLOCK_MULT = 4.0
_SHUFFLE_BLOCK_MAX_DIV = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ann_sharpe(returns: Any, periods_per_year: float) -> float:
    """Annualized Sharpe of a per-period return series.

    Unlike :func:`qaudit._stats.annualized_sharpe`, a zero-variance series
    with a nonzero mean maps to +/-inf instead of NaN: a constant PnL
    stream is the limiting infinite-Sharpe case and the mechanical-leak
    path of ``dynamic.shuffled_labels`` must still see it. Empty, single
    point, or flat-zero series map to NaN (degenerate)."""
    try:
        x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    except Exception:
        return float("nan")
    if x.size < 2:
        return float("nan")
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    if not np.isfinite(m) or not np.isfinite(sd):
        return float("nan")
    if sd == 0.0:
        return float("nan") if m == 0.0 else float(np.copysign(np.inf, m))
    return float(m / sd * np.sqrt(periods_per_year))


def _hedged_resid_sr(out: Any, mkt: pd.Series, raw_sr: float,
                     periods_per_year: float) -> float:
    """Active (exposure-hedged) annualized Sharpe of one pipeline run.

    Regresses the run's PnL on the equal-weight basket of the returns
    panel that run consumed, subtracts the beta-implied component and
    Sharpes the residual (the intercept - the alpha - is kept). Falls
    back to ``raw_sr`` when the hedge cannot be estimated (short overlap,
    flat basket, index mismatch): with no measurable exposure the raw SR
    is the active SR, and the fallback fails toward detection, never
    toward a silent pass. A PnL exactly linear in the basket leaves only
    float-rounding residuals: dust in both mean and std maps to NaN
    (nothing beyond exposure to measure), a dust std with a real mean maps
    to +/-inf (a riskless credit on top of pure exposure)."""
    if not np.isfinite(raw_sr):
        # constant PnL (+/-inf) hedges to itself (beta 0); NaN stays NaN
        return raw_sr
    try:
        p = pd.Series(out).dropna().astype(float)
        m = mkt.reindex(p.index).astype(float)
        keep = m.notna().to_numpy()
        pv = p.to_numpy(dtype=float)[keep]
        mv = m.to_numpy(dtype=float)[keep]
    except Exception:
        return raw_sr
    if pv.size < 3:
        return raw_sr
    dm = mv - mv.mean()
    var_m = float(dm @ dm)
    if not np.isfinite(var_m) or var_m <= 0.0:
        return raw_sr
    beta = float(dm @ (pv - pv.mean())) / var_m
    resid = pv - beta * mv
    mr = float(resid.mean())
    sr = float(resid.std(ddof=1))
    if not (np.isfinite(mr) and np.isfinite(sr)):
        return raw_sr
    scale = max(float(np.max(np.abs(pv))), float(np.max(np.abs(mv))), 1e-300)
    if sr <= _RESID_DUST * scale:
        if abs(mr) <= _RESID_DUST * scale:
            return float("nan")
        return float(np.copysign(np.inf, mr))
    return float(mr / sr * np.sqrt(periods_per_year))


def _run_pipeline(backtest_func: Callable, signals: pd.DataFrame,
                  asset_returns: pd.DataFrame,
                  periods_per_year: float) -> tuple[float, float]:
    """One pipeline re-run -> (raw annualized Sharpe, active annualized
    Sharpe); (NaN, NaN) if the run crashes or its Sharpe is undefined.

    The exposure proxy (equal-weight basket) is taken from the pristine
    frame before the callable runs, and the callable receives defensive
    copies of both frames: an in-place-mutating backtest_func corrupts
    only its own copies, never the caller's artifacts or the shared
    aligned frames a sibling probe will consume."""
    mkt = asset_returns.mean(axis=1)
    try:
        out = backtest_func(signals.copy(), asset_returns.copy())
    except Exception:
        return float("nan"), float("nan")
    raw = _ann_sharpe(out, periods_per_year)
    return raw, _hedged_resid_sr(out, mkt, raw, periods_per_year)


def _lag1_autocorr_by_column(frame: pd.DataFrame) -> np.ndarray:
    """Lag-1 autocorrelation of each column's non-NaN stretch; NaN for a
    column with fewer than _PHI_MIN_OBS observations (not estimated, and
    never handed to numpy's cov, whose "Degrees of freedom <= 0" is a
    warnings.warn call - fatal under -W error - that np.errstate cannot
    silence) and for a constant column (numpy's invalid-divide, which
    errstate does silence)."""
    def one(col: pd.Series) -> float:
        s = col.dropna()
        if s.size < _PHI_MIN_OBS:
            return float("nan")
        return float(s.autocorr(lag=1))

    with np.errstate(invalid="ignore", divide="ignore"):
        ac = frame.apply(one)
    return np.asarray(ac, dtype=float)


def _rank_corr_to_ar1(rho_s: float, n_names: float) -> float:
    """AR(1) coefficient whose Gaussian placebo reproduces a measured
    consecutive-date Spearman correlation ``rho_s`` over ``n_names`` names.

    Consecutive cross-sections of an AR(1) placebo with coefficient r are
    n iid bivariate-normal pairs with correlation r, whose sample Spearman
    correlation has the exact expectation
    E[r_s] = 6/(pi*(n+1)) * (arcsin(r) + (n-2)*arcsin(r/2)), strictly
    increasing from -1 at r=-1 to 1 at r=1 for n >= 2; it is inverted by
    linear interpolation on a 1e-4 grid. As n grows it tends to the
    population relation r = 2*sin(pi*rho_s/6), and plugging rho_s in as r
    under-states r further. Both leave the placebo of a persistent ranking
    too churny, i.e. too costly (measured on a static score plus slow
    noise: placebo turnover 1.5x the real book's at 10 names and 1.26x at
    30 with the population relation, 0.89x and 0.97x with the finite-n
    one). The caller feeds the median of the per-date correlations, which
    sits slightly above their mean on small cross-sections, so the residual
    error leans toward a smoother, cheaper placebo: the side on which the
    percentile under-credits an edge instead of inventing one."""
    grid = np.linspace(-1.0, 1.0, 20_001)
    expected = (6.0 / (np.pi * (n_names + 1.0))
                * (np.arcsin(grid) + (n_names - 2.0) * np.arcsin(grid / 2.0)))
    return float(np.interp(rho_s, expected, grid))


def _xsec_rank_phi(signals: pd.DataFrame) -> float:
    """Persistence of the signal's cross-sectional ranking, as an AR(1)
    coefficient: the median over consecutive date pairs of the Spearman
    correlation between the two dates' cross-sections, each ranked over
    the names finite on both dates (the Pearson correlation of the
    demeaned rank vectors), converted by :func:`_rank_corr_to_ar1` at the
    median number of such names. NaN when fewer than _PHI_MIN_OBS pairs
    are measurable.

    This is the persistence a cross-sectional pipeline actually trades: a
    signal whose ordering is a stable per-asset level plus fast noise has
    a per-column autocorrelation near 0 (Series.autocorr removes each
    column's own mean, i.e. the level) yet re-ranks the book only slowly.
    The median, not the mean: a monthly re-sort book keeps an identical
    ranking on most consecutive dates, and the median asks for a placebo
    that re-ranks about as rarely (measured on daily bars: placebo turnover
    0.73-0.86x the real book's), while the mean is diluted by the re-sort
    dates (about 4x). A pair counts only with >= _XSEC_MIN_ASSETS common
    finite names and a non-constant ranking on both dates (an all-zero
    sparse row or a fully tied row has no ordering); such pairs are skipped
    by explicit masks, so no numpy empty-slice or divide warning (fatal
    under -W error) can fire. A panel narrower than _XSEC_MIN_ASSETS names
    therefore returns NaN."""
    vals = signals.to_numpy(dtype=float)
    if vals.shape[0] < 2 or vals.shape[1] < _XSEC_MIN_ASSETS:
        return float("nan")
    cur, prev = vals[1:], vals[:-1]
    both = np.isfinite(cur) & np.isfinite(prev)
    n_common = both.sum(axis=1)
    rank_cur = pd.DataFrame(np.where(both, cur, np.nan)).rank(axis=1)
    rank_prev = pd.DataFrame(np.where(both, prev, np.nan)).rank(axis=1)
    # average ranks over n common names always sum to n(n+1)/2, so their
    # mean is exactly (n+1)/2; cells outside the common set contribute 0
    centre = ((n_common + 1.0) / 2.0)[:, None]
    dc = np.where(both, rank_cur.to_numpy(dtype=float) - centre, 0.0)
    dp = np.where(both, rank_prev.to_numpy(dtype=float) - centre, 0.0)
    sxy = np.sum(dc * dp, axis=1)
    sxx = np.sum(dc * dc, axis=1)
    syy = np.sum(dp * dp, axis=1)
    ok = (n_common >= _XSEC_MIN_ASSETS) & (sxx > 0.0) & (syy > 0.0)
    if int(np.sum(ok)) < _PHI_MIN_OBS:
        return float("nan")
    rho_s = float(np.median(sxy[ok] / np.sqrt(sxx[ok] * syy[ok])))
    return _rank_corr_to_ar1(rho_s, float(np.median(n_common[ok])))


def _signal_phi(signals: pd.DataFrame) -> float:
    """Estimate the real signal's persistence as the pipeline sees it: the
    larger of (a) the median across assets of the lag-1 autocorrelation of
    each column's non-NaN stretch and (b) the persistence of the
    cross-sectional ranking (:func:`_xsec_rank_phi`), clipped into
    [0, 0.999]. Placebo panels are generated with this AR(1) coefficient
    so their turnover - and therefore their cost drag through the
    pipeline - matches the real signal's smoothness; iid placebo noise
    would churn the book, get eaten by costs and understate the null.

    (a) alone misses persistence that lives in a static per-asset level: a
    value/quality-style score plus fast noise measures ~0 per column while
    its ranking barely moves, so its placebos churned 3-4x the real
    book's turnover and a losing low-turnover book cleared the percentile
    under costs. (b) is NaN (ignored) on panels with fewer than
    _XSEC_MIN_ASSETS names or fewer than _PHI_MIN_OBS measurable date
    pairs, leaving (a). A constant column (an all-zero sparse export, a
    dead name) has no autocorrelation: its NaN is ignored by the median
    instead of raising numpy's divide warning (fatal under -W error); a
    column with fewer than _PHI_MIN_OBS observations is NaN too (2
    observations warn "Degrees of freedom <= 0" through warnings.warn,
    which errstate does not cover, and 3 give exactly +/-1)."""
    vals = _lag1_autocorr_by_column(signals)
    med = float(np.nanmedian(vals)) if vals.size and not np.all(np.isnan(vals)) \
        else 0.0
    if not np.isfinite(med):
        med = 0.0
    xsec = _xsec_rank_phi(signals)
    if np.isfinite(xsec):
        med = max(med, xsec)
    return float(np.clip(med, 0.0, 0.999))


def _placebo_panel(signals: pd.DataFrame, phi: float,
                   rng: np.random.Generator) -> pd.DataFrame:
    """One placebo signal panel: stationary AR(1) standard-gaussian noise
    with persistence ``phi`` (x_t = phi*x_{t-1} + sqrt(1-phi^2)*eps_t,
    unit variance), masked to the real signals' NaN pattern so dead assets
    stay dead."""
    n_t, n_a = signals.shape
    eps = rng.standard_normal((n_t, n_a))
    x = np.empty((n_t, n_a))
    x[0] = eps[0]
    scale = float(np.sqrt(1.0 - phi * phi))
    for t in range(1, n_t):
        x[t] = phi * x[t - 1] + scale * eps[t]
    panel = pd.DataFrame(x, index=signals.index, columns=signals.columns)
    return panel.where(signals.notna())


def _signal_activity(signals: pd.DataFrame) -> float:
    """Fraction of the real signal's non-NaN cells that are active
    (|x| > _ACTIVE_TOL). 1.0 when there are no non-NaN cells at all (the
    dense AR(1) path then masks everything)."""
    vals = signals.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return 1.0
    return float(np.sum(np.abs(vals[finite]) > _ACTIVE_TOL) / n_finite)


def _placebo_panel_sparse(signals: pd.DataFrame,
                          rng: np.random.Generator) -> pd.DataFrame:
    """One structure-matched placebo panel for a sparse/event signal.

    One random circular row offset is drawn for the whole panel and applied
    to every column, including its values and missingness. This is stronger
    than shifting each column within its own finite stretch: even with
    staggered birth/death calendars, the complete cross-sectional event
    vector from a date moves as one unit. Thus event co-occurrence, each
    date's breadth, signed cross-sectional scores, the availability breadth,
    and every column's density/value/run sequence are preserved exactly up
    to one circular rotation; only their alignment to returns is destroyed.

    The availability calendar moves with the sparse signal. Keeping the NaN
    mask fixed at its original dates as well would be incompatible with a
    non-zero whole-panel rotation when assets have different birth/death
    calendars. Moving it is the conservative structural null: the pipeline
    sees an actually observed signal row intact, rather than a synthetic row
    assembled from independently shifted assets. Offset zero is deliberately
    part of the randomization support; ties to the observed statistic count
    against significance in the plus-one test. RNG consumption is exactly
    one integer draw, so the panel is deterministic for a given seed."""
    n_t = len(signals.index)
    if n_t == 0:
        return signals.copy()
    offset = int(rng.integers(0, n_t))
    out = np.roll(signals.to_numpy(dtype=float), offset, axis=0)
    return pd.DataFrame(out, index=signals.index, columns=signals.columns)


def _returns_phi(rets: pd.DataFrame) -> float:
    """Median across assets of the lag-1 autocorrelation of each column's
    non-NaN return stretch, sign kept, clipped into [-0.999, 0.999]; 0.0
    when unmeasurable. This signed statistic is retained as an interpretable
    diagnostic; auto block selection uses :func:`_returns_dependence`
    instead, because a signed median can cancel positive and negative
    dependence across assets. Constant columns and columns with fewer than
    _PHI_MIN_OBS observations contribute NaN (ignored), not a numpy warning
    (see _lag1_autocorr_by_column)."""
    vals = _lag1_autocorr_by_column(rets)
    med = float(np.nanmedian(vals)) if vals.size and not np.all(np.isnan(vals)) \
        else 0.0
    if not np.isfinite(med):
        med = 0.0
    return float(np.clip(med, -0.999, 0.999))


def _median_abs_lag1(frame: pd.DataFrame) -> float:
    """Median of per-column absolute lag-1 autocorrelations.

    Absolute values are taken *before* aggregation so equally strong
    positive and negative dependence cannot cancel. Unmeasurable columns
    are ignored; an entirely unmeasurable panel returns 0.0."""
    vals = _lag1_autocorr_by_column(frame)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0
    return float(np.clip(np.median(np.abs(finite)), 0.0, 0.999))


def _returns_dependence(rets: pd.DataFrame) -> dict[str, float]:
    """Dependence diagnostics used by automatic shuffle blocking.

    ``level_abs`` catches ordinary positive *or negative* serial
    correlation. ``squared_abs`` catches broad volatility clustering using
    squared returns. Each column is first divided by its finite maximum
    absolute return; this leaves autocorrelation unchanged under normal
    scaling while preventing overflow when otherwise-valid large inputs are
    squared. ``combined`` is the conservative maximum of the two.

    The statistic chooses a defensible block size; it does not prove that
    the resulting blocks are exchangeable for every possible process."""
    signed = _returns_phi(rets)
    level_abs = _median_abs_lag1(rets)

    scales = rets.abs().max(axis=0, skipna=True)
    scales = scales.where(np.isfinite(scales) & (scales > 0.0))
    normalized = rets.div(scales, axis="columns")
    squared_abs = _median_abs_lag1(normalized * normalized)
    combined = max(level_abs, squared_abs)
    return {
        "signed": signed,
        "level_abs": level_abs,
        "squared_abs": squared_abs,
        "combined": combined,
    }


def _auto_block_len(phi: float, n_dates: int) -> int:
    """Block length from a non-negative lag-1 dependence summary.

    The caller supplies the larger absolute level/squared-return dependence
    from :func:`_returns_dependence`. One (plain row permutation) is used
    when dependence is negligible or the sample is too short to hold >= 2
    blocks of length 2; otherwise use
    ceil(_SHUFFLE_BLOCK_MULT * (1+|phi|)/(1-|phi|)) clipped to
    [2, n_dates // _SHUFFLE_BLOCK_MAX_DIV]."""
    a = abs(float(phi))
    cap = int(n_dates) // _SHUFFLE_BLOCK_MAX_DIV
    if not np.isfinite(a) or a < _SHUFFLE_PHI_NEGLIGIBLE or cap < 2:
        return 1
    raw = int(np.ceil(_SHUFFLE_BLOCK_MULT * (1.0 + a) / (1.0 - a)))
    return int(np.clip(raw, 2, cap))


def _block_perm(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block permutation of range(n): the index is cut into
    contiguous blocks of ``block_len`` rows at a random phase, the block
    order is shuffled and the within-block order is kept. ``block_len <= 1``
    is exactly ``rng.permutation(n)`` (same rng consumption), so the
    dependence-free path is bit-identical to a plain row permutation."""
    if block_len <= 1:
        return rng.permutation(n)
    start = int(rng.integers(0, block_len))
    idx = np.roll(np.arange(n), -start)
    nb = int(np.ceil(n / block_len))
    blocks = [idx[i * block_len:(i + 1) * block_len] for i in range(nb)]
    order = rng.permutation(nb)
    return np.concatenate([blocks[i] for i in order])[:n]


def _split_valid(nulls: list[float]) -> tuple[np.ndarray, int]:
    """(non-NaN null SRs, count of NaN/degenerate runs). +/-inf SRs are
    kept - they are informative (constant-PnL pipelines), not degenerate."""
    arr = np.asarray(nulls, dtype=float)
    ok = arr[~np.isnan(arr)]
    return ok, int(arr.size - ok.size)


def _split_degenerate(ok: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Split non-NaN SRs into (finite core within +/-_SR_DEGENERATE_CAP,
    n constant-positive/riskless runs, n constant-negative runs). +inf and
    absurd finite positives are one class - see _SR_DEGENERATE_CAP."""
    pos = np.isposinf(ok) | (ok > _SR_DEGENERATE_CAP)
    neg = np.isneginf(ok) | (ok < -_SR_DEGENERATE_CAP)
    return ok[~pos & ~neg], int(np.sum(pos)), int(np.sum(neg))


def _gate_val(arr: np.ndarray) -> float:
    """Mean of a null-SR array for gate comparisons, safe under mixed
    +/-inf entries (whose np.mean is NaN, which would fail every >= gate
    open): any constant-positive replicate is retention-side evidence, so
    a mixed array gates as +inf."""
    if arr.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        m = float(np.mean(arr))
    if np.isnan(m) and bool(np.any(np.isposinf(arr))):
        return float("inf")
    return m


def _degenerate_skip(check_id: str, which: str, n_bad: int, n_total: int,
                     config_field: str) -> CheckResult:
    n_ok = n_total - n_bad
    if n_ok < 2 and n_bad <= _MAX_BAD_FRAC * n_total:
        # Too-few-valid-runs arm - only reachable at n_total=1 with 0 bad
        # runs (for n >= 2, ok < 2 forces bad >= n-1 > 20%): the pipeline
        # is healthy, the null just has no distribution to test. Blaming
        # backtest_func here would be mathematically false.
        return skipped(
            check_id,
            f"not run: {n_ok}/{n_total} {which} null re-runs were valid, but "
            f"a {n_ok}-point null distribution cannot be tested (need >= 2 "
            f"valid re-runs) - raise AuditConfig.{config_field} (currently "
            f"{n_total}), then re-run",
            n_bad=n_bad, n_total=n_total,
        )
    return skipped(
        check_id,
        f"not run: {n_bad}/{n_total} {which} null re-runs returned NaN/degenerate "
        f"Sharpes (> {_MAX_BAD_FRAC:.0%}), so the null distribution cannot be "
        f"trusted - fix backtest_func to return a valid net-return Series for "
        f"arbitrary {which} inputs (see dynamic.probe_health), then re-run",
        n_bad=n_bad, n_total=n_total,
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _null_mean_stats(act_ok: np.ndarray) -> dict[str, Any]:
    """Finite-core summary of a hedged null for the pipeline-bias gate:
    mean, std and t = mean / (std / sqrt(n)) over the runs within
    +/-_SR_DEGENERATE_CAP, plus the constant-positive/-negative counts."""
    core, n_pos, n_neg = _split_degenerate(act_ok)
    n_f = int(core.size)
    mean_null = float(np.mean(core)) if n_f else float("nan")
    std_null = float(np.std(core, ddof=1)) if n_f >= 2 else float("nan")
    if np.isfinite(std_null) and std_null > 0:
        t = float(mean_null / (std_null / np.sqrt(n_f)))
    elif n_f >= 2 and std_null == 0.0 and mean_null != 0.0:
        # every finite null run produced the identical nonzero Sharpe: the
        # limit of mean/(std/sqrt(n)) is +/-inf - overwhelming, not
        # missing, evidence
        t = float(np.copysign(np.inf, mean_null))
    else:
        t = float("nan")
    return dict(mean_null=mean_null, std_null=std_null, t=t, n_finite=n_f,
                n_degenerate_pos=n_pos, n_degenerate_neg=n_neg)


def _bias_gate_fires(config: AuditConfig, stats: dict[str, Any]) -> bool:
    """The pipeline-bias conviction gate on a hedged null summary
    (:func:`_null_mean_stats`): any constant-positive/riskless run, or a
    finite mean >= ``placebo_null_sharpe_fail`` with t >= _BIAS_TSTAT_FAIL.
    NaN statistics compare False, so an unmeasurable null never fires."""
    return bool(stats["n_degenerate_pos"] > 0
                or (stats["mean_null"] >= config.placebo_null_sharpe_fail
                    and stats["t"] >= _BIAS_TSTAT_FAIL))


def _sparse_tilt_control(backtest_func: Callable, signals: pd.DataFrame,
                         rets: pd.DataFrame, periods_per_year: float,
                         config: AuditConfig
                         ) -> tuple[np.ndarray, int, int]:
    """Asset-relabeled control for the sparse ``placebo_pipeline_bias``
    FAIL path -> (non-NaN hedged Sharpes, NaN-raw run count, run count).

    A rotated sparse placebo keeps each asset's event frequency, so a
    static tilt (events concentrated on names with persistent
    idiosyncratic drift, whether a genuine characteristic premium or
    names picked for their in-sample performance) survives into every
    placebo and earns a reliably positive hedged mean through an honest
    engine. Each control run rotates the real signal panel exactly like
    :func:`_placebo_panel_sparse` and pairs it with the return columns
    reassigned across assets, as the cross-section relabeling null of
    ``shuffled_labels`` does: the events, each date's return cross-section,
    the equal-weight basket (so the hedge) and any engine defect that
    works through the returns it is handed survive; which asset earns
    which return, and with it the per-asset tilt, does not. It runs
    ``config.n_placebo`` pipeline calls drawn from
    ``default_rng([config.seed, _TILT_CONTROL_STREAM])``, so every
    placebo, date-shuffle and relabeling draw of the shared stream stays
    bit-identical whether or not the control runs."""
    crng = np.random.default_rng([int(config.seed), _TILT_CONTROL_STREAM])
    vals = rets.to_numpy()
    n_assets = rets.shape[1]
    raws: list[float] = []
    acts: list[float] = []
    for _ in range(int(config.n_placebo)):
        panel = _placebo_panel_sparse(signals, crng)
        relabeled = pd.DataFrame(vals[:, crng.permutation(n_assets)],
                                 index=rets.index, columns=rets.columns)
        raw, act = _run_pipeline(backtest_func, panel, relabeled,
                                 periods_per_year)
        raws.append(raw)
        acts.append(act)
    _, n_bad = _split_valid(raws)
    act_arr = np.asarray(acts, dtype=float)
    return act_arr[~np.isnan(act_arr)], n_bad, len(raws)


def _check_placebo_bias(config: AuditConfig, act_ok: np.ndarray,
                        raw_ok: np.ndarray, phi: float,
                        activity: float | None = None, *,
                        relabeled_control: Callable[
                            [], tuple[np.ndarray, int, int]] | None = None
                        ) -> CheckResult:
    """``relabeled_control`` (sparse path only, see
    :func:`_sparse_tilt_control`) is called only when the rotated placebo
    family would convict the engine on its finite mean; without it that
    path FAILs as before."""
    n = int(act_ok.size)
    # activity is None on the dense AR(1) path; a float (< the sparse bar)
    # names the structure-matched sparse placebos in the messages
    if activity is None:
        panel_kind = f"placebo AR(1) signal panels (phi={phi:.2f})"
        placebo_kind = "ar1"
        null_panels = "signal-free random panels"
    else:
        panel_kind = (f"structure-matched sparse placebo panels "
                      f"({100 * activity:.1f}% of cells active, the real "
                      f"signal panel - values, NaN mask and run structure - "
                      f"circularly shifted by one shared random row offset)")
        placebo_kind = "sparse"
        null_panels = ("time-rotated copies of the real signal panel (no "
                       "alignment to returns)")

    stats = _null_mean_stats(act_ok)
    n_pos, n_f = stats["n_degenerate_pos"], stats["n_finite"]
    mean_null, std_null, t = stats["mean_null"], stats["std_null"], stats["t"]
    raw_core, _, _ = _split_degenerate(raw_ok)
    mean_raw = float(np.mean(raw_core)) if raw_core.size else float("nan")
    details = dict(n=n, mean_null=mean_null, std_null=std_null, t=t, phi=phi,
                   n_finite=n_f, n_degenerate_pos=n_pos,
                   n_degenerate_neg=stats["n_degenerate_neg"],
                   mean_null_raw=mean_raw, placebo_kind=placebo_kind,
                   activity=activity)

    if n_pos > 0:
        # the strongest form of manufactured performance: signal-free
        # panels earning a constant-positive / riskless PnL. Never let
        # +/-inf entries poison the t arithmetic into a silent PASS.
        finite_note = (f"; the other {n_f} finite runs average SR "
                       f"{mean_null:.2f} (t={fmt_tstat(t)})") if n_f >= 2 else ""
        return failed(
            _CHECK_PLACEBO_BIAS,
            f"{n_pos}/{n} placebo runs earn a constant-positive or near-"
            f"riskless PnL (annualized SR +inf or > {_SR_DEGENERATE_CAP:.0f} "
            f"net of market exposure) through this pipeline on "
            f"{null_panels} - the engine pays a fixed credit regardless of "
            f"the signal; every backtest run through it is inflated"
            f"{finite_note}",
            severity=Severity.CRITICAL,
            remediation="audit the PnL accounting for fixed credits and "
                        "idle-cash/carry accrual paths: a signal-free panel "
                        "must not earn a riskless positive return; rebuild "
                        "net returns from (positions * asset_returns) - costs "
                        "first principles and re-run",
            details=details,
        )
    if n_f == 0:
        # every run was constant-negative (n_neg == n): measured, and it is
        # bleed, not manufactured performance - the bias check has nothing
        # to convict; the health/percentile machinery reports usability.
        return passed(
            _CHECK_PLACEBO_BIAS,
            f"all {n} {panel_kind} run through "
            f"the pipeline lose money at constant/degenerate rates "
            f"(annualized SR -inf or < -{_SR_DEGENERATE_CAP:.0f}) - no "
            f"manufactured performance",
            details=details,
        )
    if _bias_gate_fires(config, stats):
        remediation = ("audit the backtest engine itself, not the signal: "
                       "verify the execution lag (positions held during t "
                       "must come from signals at t-lag or earlier), that "
                       "positions.loc[t] earn asset_returns.loc[t], and the "
                       "cost/PnL accounting - random inputs must not earn a "
                       "positive Sharpe beyond their market exposure")
        if activity is None:
            return failed(
                _CHECK_PLACEBO_BIAS,
                f"random signals earn mean annualized SR {mean_null:.2f} net "
                f"of market exposure (t={fmt_tstat(t)}, n={n_f}) through "
                f"this pipeline - the engine manufactures performance "
                f"(execution-timing or PnL-accounting bug); every backtest "
                f"run through it is inflated",
                severity=Severity.CRITICAL,
                remediation=remediation,
                details=details,
            )
        # Sparse family: the placebos are the real signal panel rotated in
        # time, so every asset keeps its event frequency and a static tilt
        # toward persistently outperforming names survives into each run.
        # The asset-relabeled control separates that tilt from an engine
        # credit; only the would-FAIL path pays for its extra runs.
        rotated = (f"{n_f} time-rotated copies of the real signal panel "
                   f"({100 * activity:.1f}% of cells active; each date's "
                   f"cross-section intact, its alignment to returns "
                   f"destroyed) earn mean annualized SR {mean_null:.2f} net "
                   f"of market exposure (t={fmt_tstat(t)}) through this "
                   f"pipeline")
        verdict = ("the engine manufactures performance (execution-timing "
                   "or PnL-accounting bug); every backtest run through it is "
                   "inflated")
        if relabeled_control is None:
            return failed(_CHECK_PLACEBO_BIAS, f"{rotated} - {verdict}",
                          severity=Severity.CRITICAL,
                          remediation=remediation, details=details)
        ctl_act, ctl_bad, ctl_total = relabeled_control()
        c = _null_mean_stats(ctl_act)
        c_valid = int(ctl_act.size)
        measured = c_valid >= 2 and ctl_bad <= _MAX_BAD_FRAC * ctl_total
        convicts = measured and _bias_gate_fires(config, c)
        details.update(control_kind="asset_relabeled", control_n=ctl_total,
                       control_n_bad=ctl_bad, control_n_valid=c_valid,
                       control_mean_null=c["mean_null"],
                       control_std_null=c["std_null"], control_t=c["t"],
                       control_n_finite=c["n_finite"],
                       control_n_degenerate_pos=c["n_degenerate_pos"],
                       control_measured=measured,
                       static_tilt=measured and not convicts)
        c_desc = (f"{ctl_total} asset-relabeled controls (the same kind of "
                  f"rotations run on return columns reassigned across "
                  f"assets, which removes any per-asset event tilt)")
        if convicts:
            c_pos = (f"; {c['n_degenerate_pos']} of them constant-positive "
                     f"or near-riskless" if c["n_degenerate_pos"] else "")
            return failed(
                _CHECK_PLACEBO_BIAS,
                f"{rotated}, and {c_desc} still earn mean annualized SR "
                f"{c['mean_null']:.2f} (t={fmt_tstat(c['t'])}{c_pos}) - "
                f"{verdict}",
                severity=Severity.CRITICAL,
                remediation=remediation,
                details=details,
            )
        if not measured:
            # the discriminator is missing: neither conviction nor the tilt
            # reading is established (the shuffled_labels analogue is its
            # WARN(HIGH) for an unmeasurable relabeling null)
            return warned(
                _CHECK_PLACEBO_BIAS,
                f"{rotated}, but the {ctl_total} asset-relabeled controls "
                f"that separate a static per-asset event tilt from an engine "
                f"credit could not be measured ({ctl_bad}/{ctl_total} runs "
                f"NaN/degenerate, {c_valid} with a measurable hedged "
                f"Sharpe) - cannot distinguish manufactured performance "
                f"from a tilt toward persistently outperforming names",
                severity=Severity.HIGH,
                remediation="make backtest_func return a valid net-return "
                            "Series for column-relabeled asset_returns so "
                            "the control can run; until then treat the "
                            "engine as suspect: verify the execution lag, "
                            "that positions.loc[t] earn asset_returns.loc[t], "
                            "and the cost/PnL accounting",
                details=details,
            )
        # Scoped verdict: the engine is cleared, the tilt is named. The
        # tilt itself may be a genuine characteristic premium or names
        # picked for their in-sample returns - this check cannot tell.
        return warned(
            _CHECK_PLACEBO_BIAS,
            f"{rotated}, but {c_desc} earn mean annualized SR "
            f"{c['mean_null']:.2f} (t={fmt_tstat(c['t'])}): the rotated "
            f"panels' edge follows which assets carry the events - a "
            f"static per-asset event tilt toward names that outperformed "
            f"in this sample, not performance manufactured by the engine",
            severity=Severity.MEDIUM,
            remediation="a static event tilt can be a genuine characteristic "
                        "premium or selection of names on in-sample returns: "
                        "validate the event universe out of sample with "
                        "deflated-Sharpe / trial-registry discipline instead "
                        "of hunting engine bugs; placebo_percentile ranks the "
                        "strategy against these tilted rotations, so it "
                        "credits only timing beyond the tilt",
            details=details,
        )
    if mean_null < 0:
        sign_note = ("; a slightly negative mean is the expected "
                     "transaction-cost drag")
    else:
        # The drag sentence is a claim about this mean: a non-negative null
        # mean shows no drag, so name the gate(s) that kept it a PASS
        # instead of asserting a sign the printed number contradicts.
        gates = []
        if not mean_null >= config.placebo_null_sharpe_fail:
            gates.append(f"the mean sits under the "
                         f"{config.placebo_null_sharpe_fail:.2f} bar")
        if not t >= _BIAS_TSTAT_FAIL:
            gates.append(f"t {fmt_tstat(t)} sits under the "
                         f"{_BIAS_TSTAT_FAIL:g} gate")
        sign_note = (f"; the mean is non-negative (no transaction-cost drag "
                     f"visible) and stays a PASS because "
                     f"{' and '.join(gates)}")
    return passed(
        _CHECK_PLACEBO_BIAS,
        f"{n} {panel_kind} run through the "
        f"pipeline earn mean annualized SR {mean_null:.2f} net of market "
        f"exposure (std {std_null:.2f}, t={fmt_tstat(t)}) - no manufactured "
        f"performance{sign_note}",
        details=details,
    )


def _p_clears_alpha(p_value: float, alpha: float) -> bool:
    """Numerically stable strict ``p < alpha`` decision.

    ``alpha`` is computed as ``1 - percentile`` and can land one ULP above
    a decimal cutoff (for example 0.050000000000000044). Treat numerical
    equality as equality, not as having cleared the configured bar. Cap
    the absolute tolerance relative to alpha so a small positive alpha
    remains attainable (an absolute 1e-15 bar would reject even p=0 when
    alpha <= 1e-15). Ordinary probability cutoffs keep the same tolerance.
    """
    return bool(p_value < alpha
                and not np.isclose(p_value, alpha,
                                   rtol=1e-12,
                                   atol=min(1e-15, abs(alpha) * 1e-12)))


def _minimum_placebo_runs(alpha: float) -> int | None:
    """Minimum null draws that can attain ``p < alpha``, with a floor.

    ``None`` means no finite Monte Carlo sample can attain the configured
    alpha (currently possible when ``placebo_percentile_warn == 1``).
    Exponential bracketing and binary search use the exact same comparison
    as the verdict. Work grows logarithmically with the required count:
    stepping one draw at a time near a very small alpha can otherwise
    take billions of iterations merely to calculate a resolution warning.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        return None
    lower, upper = 0, 1
    # Integer division avoids converting enormous counts to float first.
    while not _p_clears_alpha(1 / (upper + 1), alpha):
        lower, upper = upper, 2 * upper
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if _p_clears_alpha(1 / (middle + 1), alpha):
            upper = middle
        else:
            lower = middle
    return max(_MIN_PLACEBO_RANDOMIZATION_RUNS, upper)


def _placebo_quantiles(nulls: np.ndarray) -> tuple[float, float, float]:
    """Linear 5/50/95% quantiles, allowing constant-loss (-inf SR) runs.

    Positive infinities are rejected before this helper is called. For a
    quantile between -inf and a finite observation, the extended-real
    interpolated value is -inf; an exact finite order statistic stays
    finite. Computing those cases explicitly avoids numpy's inf-inf and
    zero-times-inf interpolation warnings. Nulls without -inf go straight
    through np.percentile.
    """
    if not np.isneginf(nulls).any():
        values = np.percentile(nulls, [5, 50, 95])
        return float(values[0]), float(values[1]), float(values[2])
    ordered = np.sort(nulls)
    quantiles = []
    for q in (0.05, 0.5, 0.95):
        rank = q * (len(ordered) - 1)
        lower, upper = int(np.floor(rank)), int(np.ceil(rank))
        lo, hi = float(ordered[lower]), float(ordered[upper])
        quantiles.append(lo if lower == upper or np.isneginf(lo)
                         else lo + (hi - lo) * (rank - lower))
    return quantiles[0], quantiles[1], quantiles[2]


def _check_placebo_percentile(config: AuditConfig, placebo_ok: np.ndarray,
                              actual_sr: float,
                              n_pos_active: int, *,
                              hedge_dust: bool = False,
                              n_hedge_dust: int = 0,
                              actual_sr_hedged: float = float("nan"),
                              placebo_hedged: np.ndarray | None = None,
                              sparse: bool = False
                              ) -> CheckResult:
    """Rank the actual Sharpe inside the placebo null.

    ``placebo_ok`` / ``actual_sr`` are the raw Sharpes: they drive the
    degenerate-null SKIP and the "callable did not respond" WARN. With
    ``placebo_hedged`` (the non-NaN exposure-hedged placebo Sharpes, which
    run() always supplies) the verdict ranks ``actual_sr_hedged`` among
    them (``details["sr_basis"] == "hedged"``): raw Sharpes carry whatever
    net exposure a book has, and a pipeline that maps signal levels into
    weights makes a long-only score a fully long book but the mean-zero
    placebo noise roughly market-neutral books, so a raw ranking would
    credit the equity premium to the signal. ``actual_sr_raw`` and the
    raw null quantiles stay in ``details``. Fewer than 2 hedged placebo
    Sharpes count as hedge dust (the callable's placebo PnL is basket
    exposure); without ``placebo_hedged`` the raw Sharpes are ranked
    (``sr_basis == "raw"``). ``sparse`` names the null family (time-rotated
    real signal panels instead of random AR(1) panels) in the messages.

    When only some placebo runs are exactly linear in the basket, those
    runs have no hedged Sharpe and are left out of the ranked null instead
    of being scored as 0: the ranked actual has an active component (an
    actual without one takes the "pure market exposure" path), so the
    comparable null is the placebo runs that have one too, i.e. the
    randomization test conditions on that event. It is also the
    conservative side: scored as 0 they would add runs that a positive
    actual beats and lower its p-value. ``details["n_hedge_dust"]`` counts
    them and the message names them next to the ranked ``n``."""
    n_raw = int(placebo_ok.size)
    if n_raw == 0:
        return skipped(
            _CHECK_PLACEBO_PCT,
            "not run: no valid placebo outcomes remain, so a randomization "
            "p-value cannot be computed",
            n=0, p_value=float("nan"), alpha=1.0 -
            float(config.placebo_percentile_warn),
            p_value_method="plus_one_randomization",
        )
    n_pos_raw = int(np.sum(np.isposinf(placebo_ok)
                           | (placebo_ok > _SR_DEGENERATE_CAP)))
    if placebo_hedged is not None:
        n_pos_active = max(n_pos_active, int(np.sum(
            np.isposinf(placebo_hedged)
            | (placebo_hedged > _SR_DEGENERATE_CAP))))
    if n_pos_active > 0 or n_pos_raw > 0:
        # a null containing constant-positive/riskless runs is not a
        # yardstick for the strategy - the defect is engine-side; do not
        # print inf/nan quantiles and blame the strategy for losing to them
        k = max(n_pos_active, n_pos_raw)
        return skipped(
            _CHECK_PLACEBO_PCT,
            f"not run: {k}/{n_raw} placebo re-runs returned constant-positive/"
            f"degenerate PnL (annualized SR +inf or > "
            f"{_SR_DEGENERATE_CAP:.0f}) - the placebo null is degenerate, so "
            f"the actual SR's percentile within it measures nothing about "
            f"the strategy; see dynamic.placebo_pipeline_bias for the "
            f"engine-side defect",
            n_degenerate=k, n=n_raw)
    if placebo_hedged is not None and placebo_hedged.size < 2:
        # fail closed: fewer than 2 placebo runs with a cross-sectional
        # component cannot rank anything net of exposure (run() already
        # passes hedge_dust here; a direct caller is held to the same rule)
        hedge_dust = True
        n_hedge_dust = max(n_hedge_dust, n_raw - int(placebo_hedged.size))
    hedged = placebo_hedged is not None and not hedge_dust
    null_label = "time-rotated-signal" if sparse else "random-signal"
    null_desc = ("its own signal panel rotated in time" if sparse
                 else "signal-free noise")
    q05_raw, q50_raw, q95_raw = _placebo_quantiles(placebo_ok)
    null_q50_hedged = (_placebo_quantiles(placebo_hedged)[1]
                       if placebo_hedged is not None and hedged
                       else float("nan"))
    alpha = 1.0 - float(config.placebo_percentile_warn)
    min_runs = _minimum_placebo_runs(alpha)

    def ranked(null: np.ndarray, actual: float, basis: str) -> dict[str, Any]:
        # The percentile is retained as a descriptive rank, with strict
        # comparison so ties are never counted as beaten. The decision is
        # made by the valid plus-one upper-tail randomization p-value: the
        # observed strategy is one of n+1 possible outcomes, and every
        # placebo >= actual (including ties) is an exceedance.
        n = int(null.size)
        n_exceed = int(np.sum(null >= actual))
        q05, q50, q95 = _placebo_quantiles(null)
        return dict(actual_sr=actual, percentile=float(np.mean(null < actual)),
                    p_value=float((1.0 + n_exceed) / (n + 1.0)), alpha=alpha,
                    n=n, n_tied=int(np.sum(null == actual)),
                    n_exceed=n_exceed, best_possible_p_value=1.0 / (n + 1.0),
                    min_null_runs=min_runs,
                    resolution_sufficient=min_runs is not None
                    and n >= min_runs,
                    p_value_method="plus_one_randomization",
                    null_q05=q05, null_q50=q50, null_q95=q95, sr_basis=basis,
                    actual_sr_raw=actual_sr, null_q05_raw=q05_raw,
                    null_q50_raw=q50_raw, null_q95_raw=q95_raw,
                    actual_sr_hedged=actual_sr_hedged,
                    null_q50_hedged=null_q50_hedged,
                    n_hedge_dust=n_hedge_dust)

    details = ranked(placebo_ok, actual_sr, "raw")
    n_tied, pct = details["n_tied"], details["percentile"]
    if hedge_dust or n_tied == n_raw:
        # The null did not move: every valid placebo SR equals the actual
        # SR (zero-spread null), and/or every placebo PnL was exactly
        # linear in the equal-weight basket (hedge dust, computed in run()).
        # A percentile against a null the callable cannot move measures
        # nothing about the strategy; "extend the sample" would be the
        # wrong remediation.
        why = []
        if n_tied:
            why.append(f"{n_tied}/{n_raw} placebo outcomes equal the actual "
                       f"SR exactly")
        if hedge_dust:
            why.append(f"{n_hedge_dust}/{n_raw} placebo re-runs produced PnL "
                       f"exactly linear in the equal-weight basket (hedge "
                       f"dust)")
        return warned(
            _CHECK_PLACEBO_PCT,
            f"actual annualized SR {actual_sr:.2f} vs {n_raw} {null_label} "
            f"re-runs: {' and '.join(why)} - the callable's output did not "
            f"respond to the signal panel, so its percentile "
            f"({100 * pct:.0f}th, ties not counted) measures nothing about "
            f"the strategy; verify backtest_func actually consumes its "
            f"signals argument",
            severity=Severity.HIGH,
            remediation="make backtest_func build positions from the "
                        "`signals` argument it is handed (not from a cached "
                        "or closure series, and not from the returns alone), "
                        "then re-run; a null the callable cannot move is not "
                        "evidence in either direction",
            details=details,
        )
    if hedged:
        assert placebo_hedged is not None
        details = ranked(placebo_hedged, actual_sr_hedged, "hedged")
        if np.isnan(actual_sr_hedged):
            # The placebo books carry a cross-sectional component but the
            # actual book is exactly linear in the basket (hedge dust on the
            # actual only): there is no active return to rank, and ranking
            # its raw SR (market beta) against hedged placebos would credit
            # the equity premium to the signal. NaN compares False, so the
            # rank would read as p=1/(n+1); report no rank at all.
            details.update(percentile=float("nan"), p_value=float("nan"))
            return warned(
                _CHECK_PLACEBO_PCT,
                f"actual annualized SR {actual_sr:.2f} is pure market "
                f"exposure: the book's PnL is exactly linear in the "
                f"equal-weight basket, so it has no component beyond "
                f"exposure to rank against {details['n']} {null_label} "
                f"outcomes hedged the same way - no signal edge is "
                f"measurable",
                severity=Severity.HIGH,
                remediation="the book's return is a fixed multiple of the "
                            "equal-weight basket: verify backtest_func turns "
                            "the signals into cross-sectional weights; "
                            "holding the market is not a signal edge",
                details=details,
            )
    act, n, p = details["actual_sr"], details["n"], details["p_value"]
    n_tied, pct = details["n_tied"], details["percentile"]
    q05, q50, q95 = (details["null_q05"], details["null_q50"],
                     details["null_q95"])
    # what is ranked, against what
    actual_txt = (f"actual annualized SR {act:.2f} net of market exposure "
                  f"(raw {actual_sr:.2f})" if hedged
                  else f"actual annualized SR {act:.2f}")
    null_txt = (f"{null_label} outcomes hedged the same way" if hedged
                else f"{null_label} outcomes")
    notes = (f"; {n_tied}/{n} placebo outcome(s) tie the actual SR "
             f"exactly and are not counted as beaten") if n_tied else ""
    # basket-linear placebo runs are not ranked (see the docstring): name
    # them next to the ranked n
    if hedged and n_hedge_dust:
        notes += (f"; {n_hedge_dust} placebo re-run(s) with PnL exactly "
                  f"linear in the equal-weight basket have no active "
                  f"component and are not ranked")
    if not details["resolution_sufficient"]:
        if min_runs is None:
            resolution_note = (f"no finite number of valid null runs can "
                               f"produce p < alpha={alpha:.3g}")
            resolution_remediation = (
                "set AuditConfig.placebo_percentile_warn below 1 so alpha "
                "is positive, use enough n_placebo draws to meet the "
                "reported resolution requirement, then re-run")
        else:
            resolution_note = (f"the plus-one randomization test needs "
                               f">= {min_runs} valid null runs at "
                               f"alpha={alpha:.3g}")
            resolution_remediation = (
                "increase AuditConfig.n_placebo enough to meet the reported "
                "min_null_runs (the default 100 does), then re-run; a "
                "coarse null is not evidence of an edge")
        return warned(
            _CHECK_PLACEBO_PCT,
            f"{actual_txt} sits at the {100 * pct:.0f}th percentile of only "
            f"{n} {null_txt}, but {resolution_note} (best attainable with "
            f"this null is p={1.0 / (n + 1.0):.3g}) - this Monte Carlo "
            f"resolution cannot establish an edge{notes}",
            severity=Severity.HIGH,
            remediation=resolution_remediation,
            details=details,
        )
    if not _p_clears_alpha(p, alpha):
        return warned(
            _CHECK_PLACEBO_PCT,
            f"{actual_txt} sits at the {100 * pct:.0f}th percentile of {n} "
            f"{null_txt} (null 5/50/95% = {q05:.2f}/{q50:.2f}/{q95:.2f}) - "
            f"plus-one randomization p={p:.3f} exceeds alpha={alpha:.3g}; "
            f"indistinguishable from noise at the "
            f"{100 * config.placebo_percentile_warn:.0f}% level{notes}",
            severity=Severity.HIGH,
            remediation=f"the strategy does not beat {null_desc} through its "
                        f"own pipeline: extend the sample, strengthen the "
                        f"signal, or shelve the strategy - do not allocate "
                        f"on this evidence; market exposure alone is not a "
                        f"signal edge",
            details=details,
        )
    return passed(
        _CHECK_PLACEBO_PCT,
        f"{actual_txt} sits at the {100 * pct:.0f}th percentile of {n} "
        f"{null_txt} (null 5/50/95% = {q05:.2f}/{q50:.2f}/{q95:.2f}); "
        f"plus-one randomization p={p:.3f} clears alpha={alpha:.3g} and the "
        f"{100 * config.placebo_percentile_warn:.0f}% bar{notes}",
        details=details,
    )


def _check_shuffled_labels(config: AuditConfig, shuffle_ok: np.ndarray,
                           xsec_ok: np.ndarray, xsec_trusted: bool,
                           actual_sr: float, actual_sr_raw: float,
                           mean_null_raw: float,
                           block_info: dict[str, Any] | None = None
                           ) -> CheckResult:
    n = int(shuffle_ok.size)
    with np.errstate(invalid="ignore"):
        mean_null = float(np.mean(shuffle_ok))
    gate_null = _gate_val(shuffle_ok)
    p = float((1.0 + np.sum(shuffle_ok >= actual_sr)) / (n + 1.0))
    n_x = int(xsec_ok.size)
    mean_x = _gate_val(xsec_ok)
    p_x = float((1.0 + np.sum(xsec_ok >= actual_sr)) / (n_x + 1.0)) \
        if n_x else float("nan")
    details = dict(p_value=p, actual_sr=actual_sr, mean_null=mean_null, n=n,
                   actual_sr_raw=actual_sr_raw, mean_null_raw=mean_null_raw,
                   mean_null_xsec=mean_x, p_value_xsec=p_x, n_xsec=n_x,
                   xsec_trusted=xsec_trusted, **(block_info or {}))

    survives_dates = (gate_null >= config.shuffle_leak_ratio * actual_sr
                      and actual_sr >= _LEAK_MIN_ACTUAL_SR)
    if survives_dates:
        if not xsec_trusted:
            return warned(
                _CHECK_SHUFFLE,
                f"annualized SR survives date shuffling (mean null "
                f"{mean_null:.2f} vs actual {actual_sr:.2f} net of market "
                f"exposure, {n} permutations) but the cross-section "
                f"relabeling null could not be measured - cannot distinguish "
                f"an accounting leak from a persistent cross-sectional tilt",
                severity=Severity.HIGH,
                remediation="make backtest_func return a valid net-return "
                            "Series for column-relabeled asset_returns (and "
                            "provide at least 2 assets) so the two-null "
                            "comparison can separate a static tilt from an "
                            "accounting leak",
                details=details,
            )
        if mean_x >= config.shuffle_leak_ratio * actual_sr:
            return failed(
                _CHECK_SHUFFLE,
                f"annualized SR survives label shuffling (mean null "
                f"{mean_null:.2f} vs actual {actual_sr:.2f} over {n} "
                f"permutations) AND cross-section relabeling (mean null "
                f"{mean_x:.2f}, {n_x} relabelings) - PnL depends on neither "
                f"the signal/return alignment nor which asset the signal "
                f"points at; accounting leak in the pipeline",
                severity=Severity.CRITICAL,
                remediation="the engine credits performance regardless of the "
                            "signal/return pairing: inspect the PnL accounting "
                            "for fixed credits, ignored positions, or returns "
                            "joined on the wrong axis, and rebuild net returns "
                            "from (positions * asset_returns) first principles",
                details=details,
            )
        return warned(
            _CHECK_SHUFFLE,
            f"annualized SR survives date shuffling (mean null {mean_null:.2f} "
            f"vs actual {actual_sr:.2f} net of market exposure) but collapses "
            f"under cross-section relabeling (mean null {mean_x:.2f} over "
            f"{n_x} relabelings) - the edge is a persistent cross-sectional "
            f"tilt (a near-static characteristic bet), not a time-varying "
            f"signal; the relabeling collapse rules out a signal-independent "
            f"accounting credit",
            severity=Severity.MEDIUM,
            remediation="a static tilt can be genuine (value/quality/low-vol) "
                        "or selection on in-sample winners: validate it with "
                        "deflated-Sharpe / trial-registry discipline and "
                        "out-of-sample data instead of hunting plumbing bugs; "
                        "if a time-varying signal was intended, this book is "
                        "not trading it",
            details=details,
        )
    if p >= config.shuffle_pvalue_warn:
        return warned(
            _CHECK_SHUFFLE,
            f"plus-one block-randomization p={p:.2f} vs {n} shuffled-label "
            f"nulls (actual SR "
            f"{actual_sr:.2f} net of market exposure, mean null "
            f"{mean_null:.2f}) - no detectable edge beyond chance in the "
            f"active (cross-sectional) component",
            severity=Severity.HIGH,
            remediation="the exposure-hedged Sharpe is not separable from "
                        "alignment-destroyed noise: gather more history or a "
                        "stronger signal before trusting this backtest; net-"
                        "exposure carry alone is not a cross-sectional edge",
            details=details,
        )
    return passed(
        _CHECK_SHUFFLE,
        f"actual annualized SR {actual_sr:.2f} net of market exposure beats "
        f"{n} shuffled-label nulls (mean null {mean_null:.2f}) with plus-one "
        f"block-randomization p={p:.3f}",
        details=details,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(artifacts: BacktestArtifacts, config: AuditConfig, *,
        signal_func: Callable | None = None,
        backtest_func: Callable | None = None) -> list[CheckResult]:
    """Run the null-distribution probes. Never mutates ``artifacts`` - user
    callables only ever receive defensive copies of the aligned frames."""
    if backtest_func is None:
        msg = (f"needs the pipeline callable: pass {_BT_CONTRACT} "
               f"to qaudit.audit(...) to enable this probe")
        return [skipped(cid, msg) for cid in _CHECK_IDS]

    # The artifact intake validates exported returns, but every callback
    # re-run supplies a new series. Validate those too: RangeIndex results
    # silently disable market hedging, and dropped/blanked dates can make a
    # selected subset of a backtest pass all three null probes.
    original_backtest_func = backtest_func
    n_invalid_outputs = 0
    output_issues: list[str] = []

    def checked_backtest(signals: pd.DataFrame,
                         asset_returns: pd.DataFrame) -> pd.Series:
        nonlocal n_invalid_outputs
        output = original_backtest_func(signals, asset_returns)
        try:
            return validated_returns(output, asset_returns.index)
        except PipelineOutputError as exc:
            n_invalid_outputs += 1
            issue = str(exc)
            if issue not in output_issues and len(output_issues) < 3:
                output_issues.append(issue)
            raise

    backtest_func = checked_backtest

    rng = np.random.default_rng(config.seed)
    # Validation accepts any finite positive real annualization factor. Keep
    # fractional values intact: int(0.5) == 0 would collapse every null and
    # baseline Sharpe to zero and could turn the probe into a false verdict.
    ppy = float(artifacts.periods_per_year)
    signals = artifacts.signals
    rets = artifacts.asset_returns

    # Apples-to-apples baseline: the actual SR is what this callable produces
    # on the real inputs, not the (possibly differently computed)
    # artifacts.strategy_returns.
    actual_raw, actual_act = _run_pipeline(backtest_func, signals, rets, ppy)
    # gate value: active SR when measurable, raw when the active component
    # is pure hedge dust (a book exactly linear in the basket has no
    # cross-sectional component - its raw SR is then the honest gate)
    actual_gate = actual_act if not np.isnan(actual_act) else actual_raw

    # --- placebo nulls (rng draws 1..n_placebo) ---------------------------
    # dense signals: persistence-matched AR(1) noise;
    # sparse/event signals: structure-matched panels (see module docstring)
    phi = _signal_phi(signals)
    activity = _signal_activity(signals)
    sparse = activity < _PLACEBO_SPARSE_ACTIVITY
    if sparse:
        placebo_pairs = [
            _run_pipeline(backtest_func, _placebo_panel_sparse(signals, rng),
                          rets, ppy)
            for _ in range(int(config.n_placebo))
        ]
    else:
        placebo_pairs = [
            _run_pipeline(backtest_func, _placebo_panel(signals, phi, rng),
                          rets, ppy)
            for _ in range(int(config.n_placebo))
        ]
    placebo_raw = [r for r, _ in placebo_pairs]
    placebo_ok, placebo_bad = _split_valid(placebo_raw)
    placebo_act = np.asarray([a for _, a in placebo_pairs], dtype=float)
    placebo_act = placebo_act[~np.isnan(placebo_act)]
    _, n_pos_active, _ = _split_degenerate(placebo_act)

    # --- shuffled-label nulls (rng draws after the placebos) --------------
    # contiguous row blocks are permuted (block_len == 1 is the plain row
    # permutation, same rng consumption) so serially dependent returns keep
    # their short-range structure under the null
    vals = rets.to_numpy()
    n_dates = len(rets.index)
    dependence = _returns_dependence(rets)
    returns_phi = dependence["signed"]
    if config.shuffle_block_len is None:
        block_len = _auto_block_len(dependence["combined"], n_dates)
        block_auto = True
    else:
        block_len = max(1, min(int(config.shuffle_block_len), n_dates))
        block_auto = False
    block_info: dict[str, Any] = dict(shuffle_block_len=block_len,
                                      shuffle_block_auto=block_auto,
                                      returns_lag1_autocorr=returns_phi,
                                      returns_lag1_abs_autocorr=
                                      dependence["level_abs"],
                                      returns_squared_lag1_abs_autocorr=
                                      dependence["squared_abs"],
                                      returns_block_dependence=
                                      dependence["combined"],
                                      shuffle_p_value_method=
                                      "plus_one_block_randomization",
                                      shuffle_null_assumption=
                                      "contiguous return blocks are "
                                      "approximately exchangeable at the "
                                      "selected length; automatic length is "
                                      "a conservative lag-1 level/squared-"
                                      "return heuristic, not a universal "
                                      "time-series calibration")
    shuffle_g: list[float] = []
    shuffle_raw: list[float] = []
    for _ in range(int(config.n_shuffle)):
        perm = _block_perm(n_dates, block_len, rng)
        shuffled = pd.DataFrame(vals[perm], index=rets.index,
                                columns=rets.columns)
        raw, act = _run_pipeline(backtest_func, signals, shuffled, ppy)
        shuffle_g.append(act if not np.isnan(act) else raw)
        shuffle_raw.append(raw)
    shuffle_ok, shuffle_bad = _split_valid(shuffle_g)
    shuffle_raw_ok, _ = _split_valid(shuffle_raw)
    shuffle_mean_raw = float(np.mean(shuffle_raw_ok)) \
        if shuffle_raw_ok.size else float("nan")

    # --- cross-section relabeling nulls (rng draws after the shuffles) ----
    # each asset's whole return series is reassigned to another asset:
    # time structure and the per-date equal-weight basket are preserved,
    # but the signal no longer points at the asset that earned the return.
    # A persistent-characteristic tilt dies here; an accounting leak does
    # not. Meaningless with < 2 assets (identity relabeling only).
    n_assets = rets.shape[1]
    xsec_g: list[float] = []
    if n_assets >= 2:
        for _ in range(int(config.n_shuffle)):
            cperm = rng.permutation(n_assets)
            relabeled = pd.DataFrame(vals[:, cperm], index=rets.index,
                                     columns=rets.columns)
            raw, act = _run_pipeline(backtest_func, signals, relabeled, ppy)
            xsec_g.append(act if not np.isnan(act) else raw)
    xsec_ok, xsec_bad = _split_valid(xsec_g)

    n_placebo = len(placebo_pairs)
    n_shuffle = len(shuffle_g)
    n_xsec = len(xsec_g)
    # hedge-dust runs: raw Sharpe finite, hedged component NaN. A run with
    # a NaN raw Sharpe (crash, constant-zero output) is NaN on both sides
    # and is counted by placebo_bad, never as dust.
    n_placebo_ok = int(placebo_ok.size)
    n_hedge_dust = n_placebo_ok - int(placebo_act.size)
    placebo_unhealthy = placebo_ok.size < 2 or placebo_bad > _MAX_BAD_FRAC * n_placebo
    shuffle_unhealthy = shuffle_ok.size < 2 or shuffle_bad > _MAX_BAD_FRAC * n_shuffle
    xsec_trusted = (n_assets >= 2 and xsec_ok.size >= 2
                    and xsec_bad <= _MAX_BAD_FRAC * n_xsec)
    actual_undefined = np.isnan(actual_raw)

    results: list[CheckResult] = []

    # check 1: placebo pipeline bias (does not need the actual SR)
    if placebo_unhealthy:
        results.append(_degenerate_skip(_CHECK_PLACEBO_BIAS, "placebo",
                                        placebo_bad, n_placebo, "n_placebo"))
    elif placebo_act.size < 2:
        # raw runs were fine (finite Sharpes - the percentile check right
        # below uses them) but the active components were hedge dust: PnL
        # exactly linear in the equal-weight basket. Not _degenerate_skip:
        # its message blames backtest_func for NaN re-runs that did not
        # happen and cites dynamic.probe_health, which is not emitted on
        # this path. Count dust runs explicitly so a mixed
        # case (1 finite active component) does not overclaim "every", and
        # count them among the raw-valid runs only: a crashed / NaN-raw run
        # is absent from placebo_act too but is not hedge dust.
        results.append(skipped(
            _CHECK_PLACEBO_BIAS,
            f"not run: {n_hedge_dust}/{n_placebo_ok} placebo re-runs with a "
            f"finite raw Sharpe produced PnL exactly linear in the "
            f"equal-weight basket (the exposure-hedged active component is "
            f"hedge dust), leaving {int(placebo_act.size)} < 2 "
            f"cross-sectional PnL series to measure pipeline bias on - "
            f"verify backtest_func actually consumes its signals argument, "
            f"then re-run",
            n_hedge_dust=n_hedge_dust, n_raw_valid=n_placebo_ok,
            n_total=n_placebo,
        ))
    else:
        # sparse placebos keep each asset's event frequency: before the
        # rotated family may convict the engine, the asset-relabeled control
        # (own rng stream; n_placebo extra runs, spent only when the check
        # would otherwise FAIL on the finite mean) must show the same bias.
        # A static per-asset tilt does not survive it; with < 2 assets
        # there is no cross-section to tilt and nothing to relabel.
        relabeled_control = (
            functools.partial(_sparse_tilt_control, backtest_func, signals,
                              rets, ppy, config)
            if sparse and n_assets >= 2 else None)
        results.append(_check_placebo_bias(
            config, placebo_act, placebo_ok, phi,
            activity if sparse else None,
            relabeled_control=relabeled_control))

    # check 2: percentile of the actual SR within the placebo nulls
    if placebo_unhealthy:
        results.append(_degenerate_skip(_CHECK_PLACEBO_PCT, "placebo",
                                        placebo_bad, n_placebo, "n_placebo"))
    elif actual_undefined:
        results.append(skipped(
            _CHECK_PLACEBO_PCT,
            "not run: backtest_func(artifacts.signals, artifacts.asset_returns) "
            "returned a series with undefined Sharpe (empty/constant-zero) - "
            "fix the callable's output to enable this check"))
    else:
        # ranks the exposure-hedged actual among the hedged placebo SRs
        # (raw figures in details; see the module docstring)
        hedge_dust = placebo_act.size < 2
        results.append(_check_placebo_percentile(
            config, placebo_ok, actual_raw, n_pos_active,
            hedge_dust=hedge_dust, n_hedge_dust=n_hedge_dust,
            actual_sr_hedged=actual_act, placebo_hedged=placebo_act,
            sparse=sparse))

    # check 3: shuffled labels (+ cross-section relabeling discriminator)
    if shuffle_unhealthy:
        results.append(_degenerate_skip(_CHECK_SHUFFLE, "shuffled-label",
                                        shuffle_bad, n_shuffle, "n_shuffle"))
    elif actual_undefined:
        results.append(skipped(
            _CHECK_SHUFFLE,
            "not run: backtest_func(artifacts.signals, artifacts.asset_returns) "
            "returned a series with undefined Sharpe (empty/constant-zero) - "
            "fix the callable's output to enable this check"))
    else:
        results.append(_check_shuffled_labels(config, shuffle_ok, xsec_ok,
                                              xsec_trusted, actual_gate,
                                              actual_raw, shuffle_mean_raw,
                                              block_info))

    # probe health: only reported when a distribution had to be distrusted.
    # Two distinct causes, each with its own truthful clause: bad re-runs
    # (NaN/degenerate beyond _MAX_BAD_FRAC - blame backtest_func) vs too few
    # valid runs (only reachable at n=1 with 0 bad, since for n >= 2 ok < 2
    # forces bad > 20% - blame the AuditConfig count, the callable is fine).
    xsec_unhealthy = n_assets >= 2 and not xsec_trusted
    if placebo_unhealthy or shuffle_unhealthy or xsec_unhealthy or n_invalid_outputs:
        bad_clauses: list[str] = []
        few_clauses: list[str] = []
        for name, field, unhealthy, nb, nt in (
                ("placebo", "n_placebo", placebo_unhealthy,
                 placebo_bad, n_placebo),
                ("shuffled-label", "n_shuffle", shuffle_unhealthy,
                 shuffle_bad, n_shuffle),
                ("cross-section-relabeled", "n_shuffle", xsec_unhealthy,
                 xsec_bad, n_xsec)):
            if not unhealthy:
                continue
            if nb > _MAX_BAD_FRAC * nt:
                bad_clauses.append(f"{nb}/{nt} {name}")
            else:
                few_clauses.append(f"only {nt - nb} valid {name} re-run(s) "
                                   f"of AuditConfig.{field}={nt}")
        parts: list[str] = []
        if n_invalid_outputs:
            parts.append(
                f"{n_invalid_outputs} pipeline output(s) violated the return "
                f"contract and were rejected: {'; '.join(output_issues)}")
        if bad_clauses:
            parts.append(f"{', '.join(bad_clauses)} pipeline runs returned "
                         f"NaN/degenerate Sharpes (limit {_MAX_BAD_FRAC:.0%})")
        if few_clauses:
            parts.append(f"{'; '.join(few_clauses)} - a null distribution "
                         f"needs >= 2 valid re-runs to be tested")
        rem_parts: list[str] = []
        if n_invalid_outputs:
            rem_parts.append(
                "return a real-valued pd.Series on the supplied calendar, "
                "with one finite net return per period throughout its live span")
        if bad_clauses:
            rem_parts.append("make backtest_func return a valid net-return "
                             "Series for any signal/return panel of this "
                             "shape (guard divisions by zero, handle all-NaN "
                             "rows)")
        if few_clauses:
            rem_parts.append("raise the named AuditConfig count(s) to >= 2")
        outcome = (
            "the affected null distributions were not trusted and their "
            "checks were skipped or downgraded"
            if placebo_unhealthy or shuffle_unhealthy or xsec_unhealthy
            else "invalid outputs were excluded; inspect this return-contract "
                 "finding before trusting the remaining probe statistics")
        results.append(warned(
            _CHECK_HEALTH,
            f"null re-runs are unreliable: {'; '.join(parts)}; {outcome}",
            severity=Severity.HIGH if n_invalid_outputs else Severity.LOW,
            remediation=" and ".join(rem_parts) + " so the null probes can run",
            n_placebo_bad=placebo_bad, n_placebo=n_placebo,
            n_shuffle_bad=shuffle_bad, n_shuffle=n_shuffle,
            n_xsec_bad=xsec_bad, n_xsec=n_xsec,
            n_invalid_outputs=n_invalid_outputs, output_issues=output_issues,
        ))

    return results
