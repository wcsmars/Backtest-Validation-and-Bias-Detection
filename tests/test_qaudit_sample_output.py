"""The committed sample output in examples/sample-output matches a fresh run.

examples/make_sample_output.py audits the clean control and the planted
lookahead case and writes a JSON report, an IC-by-horizon CSV and an SVG
chart. These tests regenerate them into a temporary directory and compare the
verdicts and plotted values with the committed copies, so the sample shown in
the README cannot silently drift from what the library reports.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "examples" / "make_sample_output.py"
COMMITTED = REPO_ROOT / "examples" / "sample-output"
FILES = ("lookahead-report.json", "ic-by-horizon.csv", "ic-by-horizon.svg")


def _load_script():
    spec = importlib.util.spec_from_file_location("make_sample_output", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("sample") / "out"
    assert _load_script().main(["--output-dir", str(out)]) == 0
    return out


def _statuses(path: Path) -> dict[str, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {r["check"]: r["status"] for r in report["results"]}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("name", FILES)
def test_committed_sample_is_present_ascii_text(name):
    text = (COMMITTED / name).read_text(encoding="utf-8")
    assert text.isascii()
    assert text.endswith("\n")
    # ASCII text can still spell a typographic dash as a JSON escape or an XML
    # character reference; reject those spellings too.
    lowered = text.lower()
    for code in (0x2013, 0x2014):
        for spelling in (f"\\u{code:04x}", f"&#{code};", f"&#x{code:x};"):
            assert spelling not in lowered, spelling
    if name.endswith(".json"):
        assert json.dumps(json.loads(text), ensure_ascii=False).isascii()


def test_fresh_run_writes_every_file(fresh):
    assert sorted(p.name for p in fresh.iterdir()) == sorted(FILES)


@pytest.mark.parametrize("which", ["committed", "fresh"])
def test_svg_is_well_formed(which, request):
    folder = COMMITTED if which == "committed" else request.getfixturevalue("fresh")
    root = ET.parse(folder / "ic-by-horizon.svg").getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.find("{http://www.w3.org/2000/svg}title") is not None


def test_report_verdicts_match_committed(fresh):
    committed = _statuses(COMMITTED / "lookahead-report.json")
    assert _statuses(fresh / "lookahead-report.json") == committed
    assert committed["lookahead.embedded_future_return"] == "fail"
    assert committed["lookahead.ic_decay_signature"] == "warn"


def test_plotted_values_match_committed(fresh):
    committed = _csv_rows(COMMITTED / "ic-by-horizon.csv")
    regenerated = _csv_rows(fresh / "ic-by-horizon.csv")
    assert [r["horizon"] for r in regenerated] == [r["horizon"] for r in committed]
    for new, old in zip(regenerated, committed):
        for column in old:
            assert float(new[column]) == pytest.approx(float(old[column]), abs=2e-3), column
    # The chart's point: the honest signal keeps its IC past t+1; the leak does not.
    t2 = committed[1]
    assert float(t2["clean_relative_to_t1"]) > 0.5
    assert float(t2["lookahead_relative_to_t1"]) < 0.15


def test_existing_output_dir_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        _load_script().main(["--output-dir", str(tmp_path)])
    assert excinfo.value.code == 2
