"""Unit tests for scripts/analyze_m0.py.

All numbers here are small synthetic values chosen so they can be checked by
hand. They are NOT benchmark results.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_m0 as a  # noqa: E402


def test_summarize_hand_checked():
    # samples 1, 2, 3 -> mean 2; sample stdev sqrt(((-1)^2 + 0 + 1^2) / 2) = 1; CV 50%
    s = a.summarize([1.0, 2.0, 3.0])
    assert s["n"] == 3
    assert s["mean"] == 2.0
    assert s["stdev"] == 1.0
    assert math.isclose(s["cv_pct"], 50.0)
    assert s["min"] == 1.0 and s["max"] == 3.0


def test_single_sample_has_nan_stdev():
    s = a.summarize([5.0])
    assert math.isnan(s["stdev"]) and math.isnan(s["cv_pct"])


def test_test_name():
    assert a.test_name({"n_prompt": 512, "n_gen": 0, "n_depth": 0}) == "pp512"
    assert a.test_name({"n_prompt": 0, "n_gen": 128, "n_depth": 0}) == "tg128"
    assert a.test_name({"n_prompt": 0, "n_gen": 128, "n_depth": 2048}) == "tg128@d2048"


def test_load_session_groups_rows_by_test(tmp_path):
    def row(n_prompt, n_gen):
        return {"n_prompt": n_prompt, "n_gen": n_gen, "n_depth": 0,
                "samples_ts": [1.0, 2.0, 3.0], "avg_ts": 2.0, "stddev_ts": 1.0}

    for i in (1, 2, 10):
        (tmp_path / f"s_run{i}.json").write_text(json.dumps([row(512, 0), row(0, 128)]))
    tests = a.load_session(tmp_path, "s")
    assert set(tests) == {"pp512", "tg128"}
    # run10 must sort after run2 (numeric, not lexicographic, order)
    assert [e["run"] for e in tests["pp512"]] == [1, 2, 10]


def test_check_against_tool_flags_a_mismatch():
    good = {"run": 1, "row": {"samples_ts": [1.0, 2.0, 3.0], "avg_ts": 2.0, "stddev_ts": 1.0}}
    bad = {"run": 2, "row": {"samples_ts": [1.0, 2.0, 3.0], "avg_ts": 9.0, "stddev_ts": 1.0}}
    assert a.check_against_tool(good) == []
    assert len(a.check_against_tool(bad)) == 1
