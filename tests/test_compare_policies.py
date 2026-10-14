import json
from pathlib import Path

import numpy as np
import pytest

import compare_policies as cp


def make_report(name, rows, seed_start=10000, max_steps=300):
    """rows: one (max_coverage, final_coverage) pair per seed, in seed order."""
    return {
        "policy": name,
        "protocol": {"eval_seed_start": seed_start, "max_steps": max_steps},
        "episodes": [
            {"seed": seed_start + i, "max_coverage": m, "final_coverage": f}
            for i, (m, f) in enumerate(rows)
        ],
    }


def pair_from_diffs(diffs, base=0.5):
    a = make_report("a", [(base, base)] * len(diffs))
    b = make_report("b", [(base + d, base + d) for d in diffs])
    return a, b


# ----------------------------------------------------------------- statistics


def test_difference_is_b_minus_a_with_known_values():
    a = make_report("a", [(0.1, 0.1), (0.2, 0.2), (0.3, 0.0)])
    b = make_report("b", [(0.3, 0.1), (0.2, 0.0), (0.6, 0.3)])
    r = cp.compare_reports(a, b)
    assert (r["a"], r["b"], r["n_episodes"]) == ("a", "b", 3)
    assert (r["seed_start"], r["seed_end"]) == (10000, 10002)
    m = r["metrics"]["max_coverage"]
    assert m["mean_a"] == pytest.approx(0.2)
    assert m["mean_b"] == pytest.approx(0.11 / 0.3)
    assert m["mean_difference"] == pytest.approx(0.5 / 3)
    assert m["ci95"][0] <= m["mean_difference"] <= m["ci95"][1]
    assert (m["higher"], m["equal"], m["lower"]) == (2, 1, 0)
    f = r["metrics"]["final_coverage"]
    assert f["mean_difference"] == pytest.approx(0.1 / 3)
    assert (f["higher"], f["equal"], f["lower"]) == (1, 1, 1)


def test_counts_use_a_tolerance():
    a, b = pair_from_diffs([1e-9, -1e-9, 2e-6, -2e-6])
    m = cp.compare_reports(a, b)["metrics"]["max_coverage"]
    assert (m["higher"], m["equal"], m["lower"]) == (1, 2, 1)


def test_identical_reports_have_zero_difference():
    a = make_report("a", [(0.1, 0.05), (0.3, 0.2), (0.0, 0.0)])
    m = cp.compare_reports(a, a)["metrics"]["max_coverage"]
    assert m["mean_difference"] == 0.0 and m["ci95"] == [0.0, 0.0]
    assert (m["higher"], m["equal"], m["lower"]) == (0, 3, 0)
    assert m["largest_gains"] == [] and m["gain_concentration"] is None


def test_interval_contains_the_mean_difference():
    rng = np.random.default_rng(0)
    a, b = pair_from_diffs(list(rng.normal(0.02, 0.1, 30)))
    m = cp.compare_reports(a, b)["metrics"]["max_coverage"]
    assert m["ci95"][0] < m["mean_difference"] < m["ci95"][1]


def test_largest_gains_and_losses_are_ordered_by_size():
    a, b = pair_from_diffs([0.1, 0.5, -0.3, 0.3, -0.1])
    m = cp.compare_reports(a, b, top=2)["metrics"]["max_coverage"]
    assert [s for s, _ in m["largest_gains"]] == [10001, 10003]
    assert [s for s, _ in m["largest_losses"]] == [10002, 10004]
    assert m["largest_gains"][0][1] == pytest.approx(0.5)


def test_gain_concentration_is_the_top_share_of_all_gains():
    a, b = pair_from_diffs([0.5, 0.1, 0.1, -0.2])
    one = cp.compare_reports(a, b, top=1)["metrics"]["max_coverage"]
    two = cp.compare_reports(a, b, top=2)["metrics"]["max_coverage"]
    assert one["gain_concentration"] == pytest.approx(0.5 / 0.7)
    assert two["gain_concentration"] == pytest.approx(0.6 / 0.7)


def test_gain_concentration_is_none_without_gains():
    a, b = pair_from_diffs([-0.1, 0.0, 0.0])
    assert (
        cp.compare_reports(a, b)["metrics"]["max_coverage"]["gain_concentration"]
        is None
    )


def test_top_must_be_positive():
    a, b = pair_from_diffs([0.1, 0.2])
    with pytest.raises(ValueError):
        cp.compare_reports(a, b, top=0)


def test_episode_order_in_the_reports_does_not_matter():
    a, b = pair_from_diffs([0.1, 0.5, -0.3, 0.3])
    straight = cp.compare_reports(a, b)
    b["episodes"].reverse()
    assert cp.compare_reports(a, b) == straight


# ------------------------------------------------------------- refusing mixups


def _rows(n=3):
    return [(0.1, 0.1)] * n


def _different_seed_start():
    return make_report("a", _rows()), make_report("b", _rows(), seed_start=20000)


def _different_max_steps():
    return make_report("a", _rows()), make_report("b", _rows(), max_steps=100)


def _different_seeds():
    a, b = make_report("a", _rows()), make_report("b", _rows())
    b["episodes"][0]["seed"] = 99999
    return a, b


def _duplicate_seeds():
    a, b = make_report("a", _rows()), make_report("b", _rows())
    b["episodes"][1]["seed"] = b["episodes"][0]["seed"]
    return a, b


def _no_episodes():
    a, b = make_report("a", _rows()), make_report("b", _rows())
    a["episodes"], b["episodes"] = [], []
    return a, b


def _different_lengths():
    return make_report("a", _rows(3)), make_report("b", _rows(4))


BAD_PAIRS = [
    _different_seed_start,
    _different_max_steps,
    _different_seeds,
    _duplicate_seeds,
    _no_episodes,
    _different_lengths,
]


@pytest.mark.parametrize("build", BAD_PAIRS, ids=lambda f: f.__name__.strip("_"))
def test_incomparable_reports_are_refused(build):
    a, b = build()
    with pytest.raises(cp.ComparisonError):
        cp.compare_reports(a, b)


def test_a_missing_metric_is_refused():
    a, b = make_report("a", _rows()), make_report("b", _rows())
    del b["episodes"][0]["max_coverage"]
    with pytest.raises(cp.ComparisonError):
        cp.compare_reports(a, b)


# -------------------------------------------------------------------- loading


def test_load_report_missing_file(tmp_path):
    with pytest.raises(cp.ComparisonError):
        cp.load_report(tmp_path / "nope.json")


@pytest.mark.parametrize("missing", ["policy", "protocol", "episodes"])
def test_load_report_requires_the_key_entries(tmp_path, missing):
    report = make_report("a", _rows())
    del report[missing]
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report))
    with pytest.raises(cp.ComparisonError):
        cp.load_report(path)


# ------------------------------------------------------------------- command line


def write_pair(tmp_path):
    a, b = pair_from_diffs([0.1, -0.1, 0.3, 0.0])
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    return pa, pb


def test_main_prints_and_saves_the_comparison(tmp_path, capsys):
    pa, pb = write_pair(tmp_path)
    out = tmp_path / "cmp.json"
    assert cp.main([str(pa), str(pb), "--out", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert (
        saved["a"] == "a" and saved["b"] == "b" and "max_coverage" in saved["metrics"]
    )
    printed = capsys.readouterr().out
    assert "B minus A" in printed and "max_coverage" in printed and "note:" in printed


def test_main_default_output_path_is_named_after_the_policies(tmp_path, monkeypatch):
    pa, pb = write_pair(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cp.main([str(pa), str(pb)]) == 0
    assert (tmp_path / "reports" / "compare_a_vs_b.json").exists()


def test_main_reports_incomparable_runs_as_an_error(tmp_path, capsys):
    a, b = _different_seeds()
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert cp.main([str(pa), str(pb), "--out", str(tmp_path / "x.json")]) == 1
    assert "error:" in capsys.readouterr().err


# ------------------------------------------------------- the real saved reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"
needs_floor_reports = pytest.mark.skipif(
    not (REPORTS / "eval_hold.json").exists()
    or not (REPORTS / "eval_random.json").exists(),
    reason="run evaluate.py for the random and hold policies first",
)
needs_bc_report = pytest.mark.skipif(
    not (REPORTS / "eval_hold.json").exists()
    or not (REPORTS / "eval_bc_cnn.json").exists(),
    reason="run evaluate_bc.py first",
)


@needs_floor_reports
def test_random_vs_hold_reproduces_the_earlier_one_off_numbers():
    r = cp.compare_reports(
        cp.load_report(REPORTS / "eval_hold.json"),
        cp.load_report(REPORTS / "eval_random.json"),
    )
    m = r["metrics"]["max_coverage"]
    assert m["mean_difference"] == pytest.approx(0.0111, abs=1e-4)
    assert m["ci95"][0] == pytest.approx(0.0015, abs=1e-4)
    assert m["ci95"][1] == pytest.approx(0.0243, abs=1e-4)
    assert (m["higher"], m["equal"], m["lower"]) == (8, 40, 2)


@needs_bc_report
def test_bc_vs_hold_matches_the_gain_over_start_in_the_bc_summary():
    # Doing nothing scores the starting overlap, so B minus A against it is the gain.
    bc = cp.load_report(REPORTS / "eval_bc_cnn.json")
    m = cp.compare_reports(cp.load_report(REPORTS / "eval_hold.json"), bc)["metrics"][
        "max_coverage"
    ]
    assert m["mean_difference"] == pytest.approx(
        bc["summary"]["mean_gain_over_start"], abs=5e-4
    )
    assert m["ci95"] == pytest.approx(
        bc["summary"]["mean_gain_over_start_ci95"], abs=5e-4
    )
