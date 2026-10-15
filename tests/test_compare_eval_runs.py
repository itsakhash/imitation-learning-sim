import json
import math
from pathlib import Path

import pytest

import compare_eval_runs as cer
import compare_policies as cp
import evaluate as ev

# The keyword names the tests use for a field of the first episode, and the real names of those
# fields. A mismatch here once made every "different" report identical to its partner.
FIELDS = {
    "start": "start_coverage",
    "max_cov": "max_coverage",
    "final": "final_coverage",
    "steps": "steps",
    "success": "success",
}


def episode(seed, steps=300, start=0.1, max_cov=0.3, final=0.2, success=False):
    return {
        "seed": seed,
        "steps": steps,
        "start_coverage": start,
        "max_coverage": max_cov,
        "final_coverage": final,
        "success": success,
        "terminated": success,
        "truncated": not success,
        "coverage_trace": [max_cov],
    }


def report(name="p", n=4, **first):
    episodes = [episode(10000 + i) for i in range(n)]
    for key, value in first.items():
        episodes[0][FIELDS[key]] = value  # a KeyError here means a misspelt field name
    return ev.build_report(name, episodes)


# ------------------------------------------------------------------ the helper


def test_the_report_helper_changes_the_field_it_names():
    changed = report(final=0.25, max_cov=0.31, start=0.12, steps=250, success=True)
    first = changed["episodes"][0]
    assert (
        first["final_coverage"],
        first["max_coverage"],
        first["start_coverage"],
    ) == (0.25, 0.31, 0.12)
    assert first["steps"] == 250 and first["success"] is True
    assert changed["episodes"][1] == report()["episodes"][1]
    with pytest.raises(KeyError):
        report(finl=0.25)


# ------------------------------------------------------------------ comparison


def test_identical_runs_are_reproduced():
    r = cer.compare_runs(report(), report())
    assert r["reproduced"] is True and r["episodes"] == 4
    assert (r["seed_start"], r["seed_end"]) == (10000, 10003)
    assert (
        r["episodes_with_any_exact_difference"] == 0
        and r["episodes_beyond_tolerance"] == 0
    )
    assert all(v == 0.0 for v in r["max_abs_difference"].values())
    assert r["flagged_seeds"] == []


def test_a_last_bit_difference_is_reproduced_but_counted():
    assert 0.2 + 2.8e-17 != 0.2  # the difference is real at the float level
    r = cer.compare_runs(report(), report(final=0.2 + 2.8e-17))
    assert r["reproduced"] is True and r["episodes_with_any_exact_difference"] == 1
    assert 0.0 < r["max_abs_difference"]["final_coverage"] < 1e-15


@pytest.mark.parametrize(
    "delta, expected", [(5e-7, True), (2e-6, False), (1e-3, False)]
)
def test_the_default_tolerance_is_one_millionth(delta, expected):
    r = cer.compare_runs(report(max_cov=0.3), report(max_cov=0.3 + delta))
    assert r["reproduced"] is expected
    assert r["episodes_beyond_tolerance"] == (0 if expected else 1)


def test_the_tolerance_can_be_changed():
    a, b = report(max_cov=0.3), report(max_cov=0.3 + 5e-4)
    assert cer.compare_runs(a, b, tolerance=1e-3)["reproduced"] is True
    assert cer.compare_runs(a, b)["reproduced"] is False


def test_a_difference_beyond_the_tolerance_is_flagged_by_seed():
    r = cer.compare_runs(report(), report(final=0.25))
    assert r["flagged_seeds"] == [10000]
    assert r["max_abs_difference"]["final_coverage"] == pytest.approx(0.05)


def test_a_flipped_success_is_not_reproduced():
    r = cer.compare_runs(report(success=True), report(success=False))
    assert r["reproduced"] is False and r["success_mismatches"] == 1


def test_a_different_number_of_steps_is_not_reproduced():
    r = cer.compare_runs(report(steps=300), report(steps=250))
    assert r["reproduced"] is False and r["steps_mismatches"] == 1


def test_a_start_coverage_missing_in_only_one_run_is_not_reproduced():
    r = cer.compare_runs(report(start=None), report(start=0.1))
    assert r["reproduced"] is False and r["missing_value_mismatches"] == 1


def test_a_start_coverage_missing_in_both_runs_is_fine():
    assert (
        cer.compare_runs(report(start=None), report(start=None))["reproduced"] is True
    )


def test_a_nan_is_not_reproduced_and_is_shown():
    r = cer.compare_runs(report(), report(final=float("nan")))
    assert r["reproduced"] is False and r["episodes_beyond_tolerance"] == 1
    assert math.isnan(r["max_abs_difference"]["final_coverage"])
    assert r["episodes_with_any_exact_difference"] == 1


@pytest.mark.parametrize("nan_at", [0, 2])
def test_a_nan_in_one_episode_is_not_hidden_by_larger_differences_elsewhere(nan_at):
    ea = [episode(10000 + i) for i in range(3)]
    eb = [episode(10000 + i, final=0.9) for i in range(3)]
    eb[nan_at]["final_coverage"] = float("nan")
    r = cer.compare_runs(ev.build_report("p", ea), ev.build_report("p", eb))
    assert math.isnan(r["max_abs_difference"]["final_coverage"])
    assert r["episodes_beyond_tolerance"] == 3


def test_runs_on_different_seeds_are_refused():
    other = ev.build_report("p", [episode(20000 + i) for i in range(4)])
    with pytest.raises(cp.ComparisonError):
        cer.compare_runs(report(), other)


@pytest.mark.parametrize("tolerance", [0.0, -1e-6, float("nan")])
def test_a_bad_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        cer.compare_runs(report(), report(), tolerance=tolerance)


def test_the_means_of_both_runs_are_reported():
    r = cer.compare_runs(report(max_cov=0.3), report(max_cov=0.4))
    assert r["mean_max_coverage"] == pytest.approx([0.3, 0.325])


def test_the_result_is_json_serializable():
    json.dumps(cer.compare_runs(report(), report(final=0.2 + 2.8e-17)))


# ----------------------------------------------------------------------- output


def test_format_says_reproduced_or_not():
    ok = cer.format_report(cer.compare_runs(report(), report()))
    bad = cer.format_report(cer.compare_runs(report(), report(final=0.25)))
    assert "verdict: REPRODUCED" in ok and "NOT REPRODUCED" not in ok
    assert (
        "verdict: NOT REPRODUCED" in bad
        and "first seeds beyond the tolerance: [10000]" in bad
    )
    for expected in (
        "largest differences",
        "episodes beyond the tolerance",
        "tolerance 1e-06",
    ):
        assert expected in ok


def test_format_shows_nan_instead_of_zero():
    text = cer.format_report(cer.compare_runs(report(), report(final=float("nan"))))
    assert "final coverage nan" in text and "verdict: NOT REPRODUCED" in text


def write_pair(tmp_path, second):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(report()))
    b.write_text(json.dumps(second))
    return str(a), str(b)


def test_main_returns_zero_and_writes_the_result_when_reproduced(tmp_path, capsys):
    a, b = write_pair(tmp_path, report())
    out = tmp_path / "r.json"
    assert cer.main([a, b, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["reproduced"] is True
    assert "REPRODUCED" in capsys.readouterr().out


def test_main_returns_one_but_still_writes_the_result_when_not_reproduced(tmp_path):
    a, b = write_pair(tmp_path, report(final=0.25))
    out = tmp_path / "r.json"
    assert cer.main([a, b, "--out", str(out)]) == 1
    assert json.loads(out.read_text())["reproduced"] is False


def test_main_reports_incomparable_runs_as_an_error(tmp_path, capsys):
    a, b = write_pair(
        tmp_path, ev.build_report("p", [episode(20000 + i) for i in range(4)])
    )
    assert cer.main([a, b, "--out", str(tmp_path / "r.json")]) == 1
    assert "error:" in capsys.readouterr().err


def test_main_reports_a_missing_file_as_an_error(tmp_path, capsys):
    assert cer.main([str(tmp_path / "nope.json"), str(tmp_path / "nope2.json")]) == 1
    assert "error:" in capsys.readouterr().err


def test_the_default_output_path_is_named_after_the_policy(tmp_path, monkeypatch):
    a, b = write_pair(tmp_path, report())
    monkeypatch.chdir(tmp_path)
    assert cer.main([a, b]) == 0
    assert (tmp_path / "reports" / "reproducibility_p.json").exists()


# --------------------------------------------------------------- a real saved report

REAL = Path(__file__).resolve().parents[1] / "reports" / "eval_bc_cnn.json"


@pytest.mark.skipif(not REAL.exists(), reason="run evaluate_bc.py first")
def test_a_real_report_compared_with_itself_is_reproduced():
    real = cp.load_report(REAL)
    r = cer.compare_runs(real, real)
    assert r["reproduced"] is True and r["episodes"] == 50
    assert r["episodes_with_any_exact_difference"] == 0
