"""Regenerate the sample audit output kept in examples/sample-output/.

Audits two of the synthetic demonstration backtests with the settings that
``qaudit-demo`` uses: the clean control (20-day momentum, one-day signal lag,
10 bps costs) and the planted lookahead case (the same signal with the
next-period return mixed in). It writes:

    lookahead-report.json  complete JSON report for the lookahead case
    ic-by-horizon.csv      mean cross-sectional Spearman IC by forward horizon
                           for both cases, as measured by
                           lookahead.deferred_ic_spike
    ic-by-horizon.svg      the same values as a chart, indexed to the t+1 IC

Run from the repository root after installing the library:
    python examples/make_sample_output.py --output-dir results/sample-output

Only the standard library and qaudit are needed; the chart is plain SVG.
An existing output directory is rejected.

Regenerate the committed sample with this script; tests/test_qaudit_sample_output.py
fails if a check's status or a plotted value drifts from it.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from xml.sax.saxutils import escape

from qaudit import AuditConfig, audit
from qaudit.cli import save_report
from qaudit.report import AuditReport
from qaudit.synthetic import SyntheticBacktest, make_clean, make_lookahead

HORIZON_CHECK = "lookahead.deferred_ic_spike"
DECAY_CHECK = "lookahead.ic_decay_signature"

# Chart colors: light surface, ink tokens and two categorical series.
SURFACE = "#fcfcfb"
BORDER = "#e1e0d9"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CLEAN_COLOR = "#2a78d6"
LOOKAHEAD_COLOR = "#eb6834"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def run_case(factory: Callable[[], SyntheticBacktest]) -> AuditReport:
    """Audit one synthetic case with the demonstration configuration."""
    case = factory()
    n_trials = case.audit_kwargs.get("n_trials")
    config = AuditConfig(seed=1, n_placebo=50, n_shuffle=50,
                         n_trials=n_trials if isinstance(n_trials, int) else None)
    return audit(case.artifacts, config, signal_func=case.signal_func,
                 backtest_func=case.backtest_func)


def ic_by_horizon(report: AuditReport) -> dict[int, float]:
    """Mean IC by forward horizon, as measured by the deferred-spike check."""
    raw = report[HORIZON_CHECK].details["ic_by_horizon"]
    return {int(h): float(v) for h, v in sorted(raw.items(), key=lambda kv: int(kv[0]))}


def relative(curve: dict[int, float]) -> dict[int, float]:
    base = curve[1]
    return {h: v / base for h, v in curve.items()}


def write_csv(path: Path, clean: dict[int, float], leak: dict[int, float]) -> None:
    clean_rel, leak_rel = relative(clean), relative(leak)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["horizon", "clean_ic", "clean_relative_to_t1",
                         "lookahead_ic", "lookahead_relative_to_t1"])
        for h in clean:
            writer.writerow([h, f"{clean[h]:.4f}", f"{clean_rel[h]:.4f}",
                             f"{leak[h]:.4f}", f"{leak_rel[h]:.4f}"])


def _fmt_ratio(value: float) -> str:
    return f"{value:.2f}"


def _text(x: float, y: float, content: str, *, size: int = 12, fill: str = INK_SECONDARY,
          anchor: str = "start", weight: str = "normal", extra: str = "") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{extra}>{escape(content)}</text>')


def ic_chart_svg(clean: dict[int, float], leak: dict[int, float],
                 threshold: float) -> str:
    """Line chart of IC(h) / IC(t+1) for both cases on one indexed axis."""
    width, height = 720, 430
    left, right, top, bottom = 72, 572, 112, 360
    pad = 18
    horizons = list(clean)
    clean_rel, leak_rel = relative(clean), relative(leak)
    values = list(clean_rel.values()) + list(leak_rel.values())
    y_max = max(1.2, math.ceil(max(values) / 0.2) * 0.2)
    y_min = min(0.0, math.floor(min(values) / 0.2) * 0.2)

    def x_of(h: int) -> float:
        span = max(len(horizons) - 1, 1)
        return left + pad + (horizons.index(h) / span) * (right - left - 2 * pad)

    def y_of(v: float) -> float:
        return bottom - (v - y_min) / (y_max - y_min) * (bottom - top)

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" '
        f'font-family="{FONT}">',
        '<title id="title">Predictive power by forward horizon, relative to t+1</title>',
        '<desc id="desc">' + escape(
            "Mean cross-sectional Spearman IC at forward horizons t+1 to "
            f"t+{horizons[-1]}, divided by the IC at t+1. Honest momentum: "
            + ", ".join(f"{clean_rel[h]:.3f}" for h in horizons)
            + f" (IC at t+1 {clean[1]:.3g}). Planted lookahead: "
            + ", ".join(f"{leak_rel[h]:.3f}" for h in horizons)
            + f" (IC at t+1 {leak[1]:.3g}).") + "</desc>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8" '
        f'fill="{SURFACE}" stroke="{BORDER}"/>',
        _text(24, 34, "Predictive power by forward horizon, relative to t+1",
              size=16, fill=INK, weight="600"),
        _text(24, 56, "Mean cross-sectional Spearman IC at horizon h divided by the IC "
              "at t+1. Both series equal 1 at t+1 by construction.", size=12),
    ]

    # Legend: a short line key with its marker beside ink text.
    legend = [(CLEAN_COLOR, f"Honest momentum (clean control): IC at t+1 = {clean[1]:.3g}"),
              (LOOKAHEAD_COLOR, f"Planted lookahead: IC at t+1 = {leak[1]:.3g}")]
    lx = 24.0
    for color, label in legend:
        out.append(f'<line x1="{lx:.1f}" y1="82" x2="{lx + 18:.1f}" y2="82" stroke="{color}" '
                   'stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<circle cx="{lx + 9:.1f}" cy="82" r="4" fill="{color}" '
                   f'stroke="{SURFACE}" stroke-width="2"/>')
        out.append(_text(lx + 26, 86, label))
        lx += 26 + 6.4 * len(label) + 28

    # Gridlines and y-axis ticks.
    steps = round((y_max - y_min) / 0.2)
    for i in range(steps + 1):
        v = y_min + 0.2 * i
        y = y_of(v)
        stroke = BASELINE if abs(v) < 1e-9 else GRID
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                   f'stroke="{stroke}" stroke-width="1"/>')
        out.append(_text(left - 10, y + 4, f"{v:.1f}", size=11, fill=INK_MUTED, anchor="end",
                         extra=' font-variant-numeric="tabular-nums"'))
    for h in horizons:
        out.append(_text(x_of(h), bottom + 22, f"t+{h}", size=11, fill=INK_MUTED,
                         anchor="middle"))
    out.append(_text((left + right) / 2, bottom + 50,
                     "Forward horizon (trading days after the signal date)",
                     size=12, anchor="middle"))
    mid_y = (top + bottom) / 2
    out.append(_text(22, mid_y, "IC(h) / IC(t+1)", size=12, anchor="middle",
                     extra=f' transform="rotate(-90 22 {mid_y:.1f})"'))

    # Decision threshold of lookahead.ic_decay_signature.
    ty = y_of(threshold)
    out.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{right}" y2="{ty:.1f}" '
               f'stroke="{INK_MUTED}" stroke-width="1" stroke-dasharray="4 4"/>')
    out.append(_text(right - 4, ty + 15, f"ic_decay_signature threshold {threshold:.2f}",
                     size=11, fill=INK_MUTED, anchor="end"))

    # Series: 2px lines, markers ringed in the surface color.
    for color, rel in ((CLEAN_COLOR, clean_rel), (LOOKAHEAD_COLOR, leak_rel)):
        points = " ".join(f"{x_of(h):.1f},{y_of(rel[h]):.1f}" for h in horizons)
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                   'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for h in horizons:
            out.append(f'<circle cx="{x_of(h):.1f}" cy="{y_of(rel[h]):.1f}" r="4" '
                       f'fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')

    # Selective labels: the t+2 ratio the decay check reads, and series names.
    if 2 in clean_rel:
        out.append(_text(x_of(2), y_of(clean_rel[2]) - 12, _fmt_ratio(clean_rel[2]),
                         size=11, fill=INK, anchor="middle"))
        out.append(_text(x_of(2) + 9, y_of(leak_rel[2]) - 9, _fmt_ratio(leak_rel[2]),
                         size=11, fill=INK))
    last = horizons[-1]
    out.append(_text(right + 10, y_of(clean_rel[last]) + 4, "Honest momentum", fill=INK))
    out.append(_text(right + 10, y_of(leak_rel[last]) + 4, "Planted lookahead", fill=INK))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sample-output"))
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"{args.output_dir} already exists; choose a new --output-dir.")

    print("Auditing the clean control and the planted lookahead case...", flush=True)
    clean_report = run_case(make_clean)
    leak_report = run_case(make_lookahead)
    clean, leak = ic_by_horizon(clean_report), ic_by_horizon(leak_report)
    threshold = float(leak_report[DECAY_CHECK].details["collapse_ratio"])
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        save_report(leak_report, args.output_dir / "lookahead-report.json")
        write_csv(args.output_dir / "ic-by-horizon.csv", clean, leak)
        with (args.output_dir / "ic-by-horizon.svg").open("x", encoding="utf-8") as handle:
            handle.write(ic_chart_svg(clean, leak, threshold))
    except OSError as exc:
        print(f"Could not write the sample output: {exc}", file=sys.stderr)
        return 2
    print(f"clean control:     {clean_report.summary()}")
    print(f"planted lookahead: {leak_report.summary()}")
    print(f"Wrote lookahead-report.json, ic-by-horizon.csv and ic-by-horizon.svg "
          f"to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
