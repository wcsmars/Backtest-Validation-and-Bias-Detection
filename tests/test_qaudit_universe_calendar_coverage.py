"""Universe calendar coverage and calendar-scaled attrition sample floors.

Sparse membership stamps need explicit alignment before auditing. Attrition
uses elapsed calendar time so monthly data do not require decades of bars.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from qaudit.api import audit
from qaudit.checks import survivorship
from qaudit.checks.survivorship import MIN_BARS_NO_EXITS, MIN_YEARS_NO_EXITS
from qaudit.config import AuditConfig
from qaudit.errors import MisalignedInputError
from qaudit.inputs import (_UNIVERSE_MAX_TRAILING_GAP_DAILY_BARS,
                           _UNIVERSE_MIN_COVERAGE, BacktestArtifacts)
from qaudit.synthetic import (momentum_signal, positions_from_signals,
                              simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()


def _by(results):
    return {r.check: r for r in results}


@pytest.fixture(scope="module")
def market():
    return simulate_market(n_assets=30, n_periods=1000, seed=5)


@pytest.fixture(scope="module")
def honest(market):
    return momentum_signal(market["returns"])


def _art(market, sig, universe, **kw):
    base = dict(signals=sig, asset_returns=market["returns"],
                positions=positions_from_signals(sig, 1),
                universe=universe, signal_lag=1)
    base.update(kw)
    return BacktestArtifacts(**base)


def _month_end_stamps(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().sort_values())


# Sparse membership-calendar rejection.

def test_month_end_stamped_universe_rejected(market, honest):
    uni = market["universe"].loc[_month_end_stamps(market["universe"].index)]
    art = _art(market, honest, uni)
    with pytest.raises(MisalignedInputError) as exc:
        art.validate()
    msg = str(exc.value)
    assert "artifacts.universe" in msg
    # actionable: names the fabricated-False mechanism and the ffill fix
    assert "fillna(False)" in msg
    assert "ffill" in msg
    assert re.search(r"\d+ of \d+ trading dates", msg)


def test_weekly_stamped_universe_rejected(market, honest):
    # weekly rebalance-row stamping (~20% coverage) must also fail - the
    # floor is calibrated above the densest common stamping artifact
    idx = market["universe"].index
    uni = market["universe"].loc[idx[idx.weekday == 4]]
    art = _art(market, honest, uni)
    with pytest.raises(MisalignedInputError, match="universe"):
        art.validate()
    # Pin the membership coverage floor within a band: low thresholds admit fabricated
    # missing membership, while values above 0.70 reject the dense late-start control.
    assert 0.5 <= _UNIVERSE_MIN_COVERAGE <= 0.70


def test_sparse_universe_cannot_launder_no_exits(market):
    # A monthly-stamped survivor list fabricates exits at stamp boundaries; validation
    # must reject it before attrition checks.
    m = simulate_market(n_assets=60, seed=3, death_frac=0.0)
    total = (1 + m["returns"]).prod()
    keep = total.sort_values(ascending=False).index[:30]
    rets = m["returns"][keep]
    uni = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    uni = uni.loc[_month_end_stamps(uni.index)]
    sig = momentum_signal(rets)
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            positions=positions_from_signals(sig, 1),
                            universe=uni, signal_lag=1)
    with pytest.raises(MisalignedInputError, match="no_exits"):
        art.validate()


def test_sparse_universe_error_mentions_stamp_spacing(market, honest):
    uni = market["universe"].loc[_month_end_stamps(market["universe"].index)]
    art = _art(market, honest, uni)
    with pytest.raises(MisalignedInputError) as exc:
        art.validate()
    # message estimates the stamping cadence so the user recognises the
    # vendor artifact (~21 trading days for month-end stamps)
    m = re.search(r"stamped every ~(\d+) trading days", str(exc.value))
    assert m and 18 <= int(m.group(1)) <= 23


def test_zero_overlap_gate_still_intact(market, honest):
    # the zero-overlap gate is independent of the coverage gate: fully
    # disjoint dates still raise (not fall through to a divide or a
    # coverage message)
    uni = market["universe"].copy()
    uni.index = uni.index + pd.DateOffset(years=30)
    art = _art(market, honest, uni)
    with pytest.raises(MisalignedInputError, match="shares no"):
        art.validate()


# Dense and partially covered membership controls.

def test_dense_universe_still_validates(market, honest):
    art = _art(market, honest, market["universe"])
    art.validate()   # must not raise
    r = _by(survivorship.run(art.aligned(), CFG))[
        "survivorship.trading_outside_universe"]
    assert r.status is Status.PASS


def test_holiday_drift_universe_still_validates(market, honest):
    # a universe on a real exchange calendar misses a few % of a naive
    # bdate grid - legitimate, far above the floor
    rng = np.random.default_rng(7)
    uni = market["universe"]
    keep = rng.random(len(uni)) > 0.03
    art = _art(market, honest, uni.loc[uni.index[keep]])
    art.validate()   # must not raise


def test_late_starting_dense_universe_still_validates(market, honest):
    # membership history shorter than the panel (dense, back 70%) keeps the
    # same partial-overlap latitude validate() grants positions
    uni = market["universe"]
    art = _art(market, honest, uni.iloc[int(0.30 * len(uni)):])
    art.validate()   # must not raise


# A universe that ends before the backtest: the missing tail is not attrition.

def _survivor_only(market):
    # Ground truth: no name ever leaves; the only exits a check can see are
    # the ones aligned() fabricates from a missing membership tail.
    rets = market["returns"].fillna(0.0)
    sig = momentum_signal(rets)
    full = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    return rets, sig, full


@pytest.mark.parametrize("layout", ["truncated_index", "nan_tail"])
@pytest.mark.parametrize("with_positions", [False, True])
def test_universe_ending_before_the_backtest_is_rejected(market, layout,
                                                         with_positions):
    # A constituents feed that lags the returns feed by ~a month used to fill
    # the tail with False: every name made a simultaneous terminal "exit",
    # no_exits passed a survivor-only list as point-in-time, and with
    # positions trading_outside_universe blamed the book for the data gap.
    rets, sig, full = _survivor_only(market)
    if layout == "truncated_index":
        uni = full.iloc[:-20]
    else:
        uni = full.astype(object)
        uni.iloc[-21:] = np.nan
    art = BacktestArtifacts(
        signals=sig, asset_returns=rets, universe=uni, signal_lag=1,
        positions=positions_from_signals(sig, 1) if with_positions else None)
    with pytest.raises(MisalignedInputError) as exc:
        art.validate()
    msg = str(exc.value)
    assert re.search(r"stops 2[01] grid bar\(s\) before the end", msg)
    assert "no_exits" in msg and "forward-fill" in msg
    with pytest.raises(MisalignedInputError, match="stops"):
        audit(art, include=["survivorship"])


def test_universe_tail_within_the_flicker_window_still_validates(market):
    # A short tail stays legal: no_exits already reports exits that close
    # to the sample end as unverifiable instead of counting them.
    rets, sig, full = _survivor_only(market)
    art = BacktestArtifacts(
        signals=sig, asset_returns=rets, signal_lag=1,
        universe=full.iloc[:-_UNIVERSE_MAX_TRAILING_GAP_DAILY_BARS])
    art.validate()   # must not raise
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.WARN


def test_universe_tail_window_matches_the_no_exits_edge_window():
    # The input gate starts where no_exits' own sample-end guard stops.
    assert (_UNIVERSE_MAX_TRAILING_GAP_DAILY_BARS
            == survivorship.NO_EXITS_FLICKER_MAX_OUT_DAILY_BARS)


def test_genuine_trailing_exits_still_validate(market):
    # Honest layouts whose membership thins out at the end stay legal:
    # per-column trailing NaN (delisted names) and explicit False for index
    # deletions that keep trading with finite returns. The exits are
    # staggered, one name per bar, as real delistings are.
    rets, sig, full = _survivor_only(market)
    uni = full.astype(object)
    for j in range(20):
        start = len(uni) - 60 - 7 * j
        # delisted: NaN tail per name; then removed, still trading
        uni.iloc[start:, j] = np.nan if j < 10 else False
    art = BacktestArtifacts(signals=sig, asset_returns=rets, universe=uni,
                            signal_lag=1)
    art.validate()   # must not raise
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.PASS
    assert r.details["n_exiting_assets"] == 20


def test_universe_tail_is_allowed_where_the_members_stop_trading(market):
    # The input gate only rejects a tail on which the last members keep
    # finite returns; a panel whose returns also end is a shorter live
    # sample. no_exits still must not read the members' "exits" into that
    # unobserved tail as attrition: follow-up after a terminal exit is
    # measured to the last bar with any finite return.
    rets, sig, full = _survivor_only(market)
    rets = rets.copy()
    rets.iloc[-30:] = np.nan
    art = BacktestArtifacts(signals=sig, asset_returns=rets, signal_lag=1,
                            universe=full.iloc[:-30])
    art.validate()   # must not raise
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_exiting_assets"] == 0
    assert r.details["n_unverifiable_exit_events"] == 30


@pytest.mark.parametrize("dead_bars", [30, 60])
def test_no_exits_into_a_dead_returns_tail_are_unverifiable(market,
                                                            dead_bars):
    # A survivor-only list whose universe and returns stop together used to
    # PASS no_exits: the grid rows after both feeds end still counted as
    # follow-up, so every member became an observed terminal delisting.
    rets, sig, full = _survivor_only(market)
    rets = rets.copy()
    rets.iloc[-dead_bars:] = np.nan
    art = BacktestArtifacts(signals=sig, asset_returns=rets, signal_lag=1,
                            universe=full.iloc[:-dead_bars])
    rep = audit(art, include=["survivorship"])
    r = rep["survivorship.no_exits"]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_exiting_assets"] == 0
    assert "sample end" in r.message


def _padded_tail(full):
    return full.iloc[:-20].reindex(full.index, fill_value=False)


def _nan_tail_except_one_true(full):
    uni = full.astype(object)
    uni.iloc[-40:, 1:] = np.nan
    return uni


def _nan_tail_except_one_false(full):
    uni = full.astype(object)
    uni.iloc[-21:, 1:] = np.nan
    uni.iloc[-21:, 0] = False
    return uni


def _half_the_columns_nan_tail(full):
    uni = full.astype(object)
    uni.iloc[-60:, :15] = np.nan
    return uni


@pytest.mark.parametrize("layout", [
    _padded_tail, _nan_tail_except_one_true, _nan_tail_except_one_false,
    _half_the_columns_nan_tail])
def test_mass_simultaneous_exit_is_not_attrition(market, layout):
    # Layouts that clear the trailing-coverage gate (some cell stays
    # stamped to the end) but still fabricate a mass exit of a survivor-only
    # list, e.g. a short feed "fixed" with reindex(fill_value=False). At
    # least half the live members leaving on one bar while they keep
    # trading is a feed gap, not attrition.
    rets, sig, full = _survivor_only(market)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, signal_lag=1,
                            universe=layout(full))
    art.validate()   # must not raise
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_exiting_assets"] == 0
    assert r.details["n_mass_exit_events"] >= 15
    assert "at once" in r.message


def test_reconstitution_wave_below_half_is_attrition(market):
    # Honest guard for the mass-exit prong: an index reconstitution that
    # drops a third of the names on one bar (they keep trading) is real
    # membership turnover and still counts.
    rets, sig, full = _survivor_only(market)
    uni = full.copy()
    uni.iloc[-200:, :10] = False
    art = BacktestArtifacts(signals=sig, asset_returns=rets, signal_lag=1,
                            universe=uni)
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.PASS
    assert r.details["n_exiting_assets"] == 10
    assert r.details["n_mass_exit_events"] == 0


def test_sparse_positions_not_coverage_gated(market, honest):
    # Deliberate asymmetry: positions rows missing from the grid decode to
    # 0.0 = flat, a legitimate sparse encoding (and a wrong one gets caught
    # loudly by the gross/strategy_returns cross-checks) - while universe
    # rows decode to False, which fabricates non-membership. So positions
    # keep only the zero-overlap gate.
    pos = positions_from_signals(honest, 1)
    art = _art(market, honest, market["universe"],
               positions=pos.loc[_month_end_stamps(pos.index)])
    art.validate()   # must not raise


def test_hindsight_universe_detection_power_intact(market, honest):
    # a zombie holding far past the 21-bar grace must still fail CRITICAL
    # on a calendar that passes validate()
    uni = market["universe"].copy()
    uni.iloc[:, :] = True
    uni.iloc[500:, 0] = False
    pos = positions_from_signals(honest, 1).copy()
    pos.iloc[:, 0] = 0.05                    # held throughout, incl. post-exit
    art = _art(market, honest, uni, positions=pos)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[
        "survivorship.trading_outside_universe"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL


# Attrition sample floors on monthly data.

def _monthly_frames(n_periods, n_assets=30, seed=0, n_exits=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-29", periods=n_periods, freq=pd.offsets.BMonthEnd())
    assets = [f"A{i:02d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.008, 0.05, (n_periods, n_assets)),
                        index=dates, columns=assets)
    uni = pd.DataFrame(True, index=dates, columns=assets)
    for j in range(n_exits):
        t = int(rng.integers(int(0.3 * n_periods), int(0.9 * n_periods)))
        rets.iloc[t:, j] = np.nan
        uni.iloc[t:, j] = False
    sig = rets.rolling(3, min_periods=3).mean()
    return BacktestArtifacts(
        signals=sig, asset_returns=rets,
        positions=positions_from_signals(sig, 1),
        universe=uni, signal_lag=1, periods_per_year=12)


def test_no_exits_active_on_three_years_of_monthly_bars():
    # 36 monthly bars = 3 years: plenty of calendar time. A raw 252-bar floor
    # would silently SKIP here (disabled until 21 years of monthly bars).
    art = _monthly_frames(36)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.WARN
    assert "no asset ever leaves" in r.message
    assert r.details["years"] == pytest.approx(3.0)


def test_no_exits_activates_at_thirteen_monthly_bars():
    # ~13 months of monthly bars clears the 1-year floor (12 bars at
    # periods_per_year=12). run() is exercised directly: 13 dates is below
    # validate()'s global _MIN_OVERLAP, which is a separate, orthogonal gate.
    r = _by(survivorship.run(_monthly_frames(13).aligned(), CFG))[
        "survivorship.no_exits"]
    assert r.status is Status.WARN


def test_no_exits_below_floor_monthly_still_skips():
    r = _by(survivorship.run(_monthly_frames(11).aligned(), CFG))[
        "survivorship.no_exits"]
    assert r.status is Status.SKIP
    assert r.details["min_periods_required"] == max(
        int(np.ceil(MIN_YEARS_NO_EXITS * 12)), MIN_BARS_NO_EXITS)
    assert "periods_per_year=12" in r.message


def test_no_exits_token_exit_still_caught_on_monthly_bars():
    # detection power at monthly frequency: one token exit over 30 assets x
    # 5 years = 0.67%/yr, below the 1%/yr plausibility floor -> WARN
    art = _monthly_frames(60, n_exits=1)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.WARN
    assert r.details["n_exiting_assets"] == 1
    assert r.details["exit_rate_per_year"] < CFG.min_exit_rate_per_year


def test_monthly_exits_into_a_dead_returns_tail_are_unverifiable():
    # At monthly bars (window 0) a terminal exit still needs one observed
    # bar after it: a universe and returns feed that stop together three
    # months early used to PASS as 30 terminal delistings.
    art = _monthly_frames(36)
    rets = art.asset_returns.copy()
    rets.iloc[-3:] = np.nan
    dead = BacktestArtifacts(signals=art.signals, asset_returns=rets,
                             universe=art.universe.iloc[:-3], signal_lag=1,
                             periods_per_year=12)
    r = _by(survivorship.run(dead.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.WARN
    assert r.details["n_exiting_assets"] == 0
    assert r.details["n_unverifiable_exit_events"] == 30


# Monthly attrition controls.

def test_monthly_universe_tail_gate_has_no_slack():
    # At periods_per_year=12 the daily window rescales to zero bars, as the
    # no_exits flicker window does; no_exits then counts every terminal
    # exit, so a single missing month would already launder a survivor-only
    # list into a PASS. One missing month is rejected.
    art = _monthly_frames(36)
    art.validate()   # the complete universe is fine
    bad = BacktestArtifacts(signals=art.signals, asset_returns=art.asset_returns,
                            universe=art.universe.iloc[:-1], signal_lag=1,
                            periods_per_year=12)
    with pytest.raises(MisalignedInputError,
                       match=r"stops 1 grid bar.*tolerance 0 bar"):
        bad.validate()


def test_weekly_universe_tail_gate_uses_the_rescaled_window():
    # periods_per_year=52: floor(10 * 52 / 252) = 2 bars, the same window
    # no_exits uses to call a sample-end exit unverifiable.
    rng = np.random.default_rng(3)
    dates = pd.date_range("2015-01-02", periods=160, freq="W-FRI")
    assets = [f"A{i:02d}" for i in range(20)]
    rets = pd.DataFrame(rng.normal(0.002, 0.02, (160, 20)), index=dates,
                        columns=assets)
    uni = pd.DataFrame(True, index=dates, columns=assets)
    sig = rets.rolling(4, min_periods=4).mean()

    def art(n_missing):
        return BacktestArtifacts(signals=sig, asset_returns=rets,
                                 universe=uni.iloc[:len(uni) - n_missing],
                                 signal_lag=1, periods_per_year=52)
    art(2).validate()   # within the window: must not raise
    with pytest.raises(MisalignedInputError, match=r"tolerance 2 bar"):
        art(3).validate()


def test_no_exits_honest_monthly_universe_passes():
    # 4 of 30 names exit over 3 years (~4.4%/yr) - realistic attrition
    art = _monthly_frames(36, n_exits=4)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.PASS
    assert r.details["n_exiting_assets"] == 4


def test_no_exits_daily_floor_unchanged():
    # 1 year of daily bars: the boundary must sit at 252, so calendar scaling
    # leaves the daily calibration unchanged
    def daily(n_periods):
        rng = np.random.default_rng(1)
        dates = pd.bdate_range("2020-01-02", periods=n_periods)
        assets = [f"A{i:02d}" for i in range(12)]
        rets = pd.DataFrame(rng.normal(0.0, 0.01, (n_periods, 12)),
                            index=dates, columns=assets)
        sig = rets.rolling(5, min_periods=5).mean()
        return BacktestArtifacts(
            signals=sig, asset_returns=rets,
            positions=positions_from_signals(sig, 1),
            universe=pd.DataFrame(True, index=dates, columns=assets),
            signal_lag=1)

    at = _by(survivorship.run(daily(252).aligned(), CFG))["survivorship.no_exits"]
    below = _by(survivorship.run(daily(251).aligned(), CFG))["survivorship.no_exits"]
    assert at.status is Status.WARN      # static universe, check active
    assert below.status is Status.SKIP   # guard against short-sample noise
    assert "251" in below.message
