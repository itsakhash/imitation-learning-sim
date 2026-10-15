import json
from pathlib import Path

import numpy as np
import pytest

import compare_policies as cp
import evaluate as ev
import trace_analysis as ta

# Episode traces used throughout (length 6, evaluated with window=2 where it matters):
A_ABRUPT = [0.0, 0.3, 0.3, 0.0, 0.0, 0.0]  # peak 0.3 at step 2, drops to 0 at once
B_GRADUAL = [0.5, 0.47, 0.44, 0.41, 0.38, 0.35]  # slow slide: giveback 0.15
C_STEADY = [0.2] * 6  # no giveback
D_ZERO = [0.0] * 6  # ends empty, no giveback
E_RISING = [0.0, 0.1, 0.2, 0.3, 0.4, 0.4]  # peak at step 5, no giveback
F_ABRUPT = [0.0, 0.0, 0.6, 0.6, 0.0, 0.0]  # peak 0.6 at step 3, drops to 0 at once
SIX = [A_ABRUPT, B_GRADUAL, C_STEADY, D_ZERO, E_RISING, F_ABRUPT]


def episode(seed, trace):
    trace = list(trace)
    return {
        "seed": seed,
        "steps": len(trace),
        "start_coverage": 0.0,
        "max_coverage": max(trace),
        "final_coverage": trace[-1],
        "success": False,
        "terminated": False,
        "truncated": True,
        "coverage_trace": trace,
    }


def report(name, traces, seed_start=10000):
    return ev.build_report(
        name, [episode(seed_start + i, t) for i, t in enumerate(traces)]
    )


# ------------------------------------------------------------------ peak and drop


def test_peak_step_is_the_first_maximum_and_one_based():
    assert ta.peak_step([0.1, 0.5, 0.5, 0.2]) == 2
    assert ta.peak_step([0.7]) == 1
    assert ta.peak_step(E_RISING) == 5


def test_peak_step_rejects_empty_traces():
    with pytest.raises(ValueError):
        ta.peak_step([])


def test_largest_window_drop_known_values():
    trace = [0.1, 0.5, 0.4, 0.1, 0.1]
    assert ta.largest_window_drop(trace, 1) == pytest.approx(0.3)  # 0.4 -> 0.1
    assert ta.largest_window_drop(trace, 2) == pytest.approx(
        0.4
    )  # 0.5 -> 0.1 within two steps
    assert ta.largest_window_drop(trace, 8) == pytest.approx(0.4)


def test_a_trace_that_never_falls_has_no_drop():
    assert ta.largest_window_drop([0.0, 0.1, 0.2, 0.2], 8) == 0.0
    assert ta.largest_window_drop([0.3], 8) == 0.0


def test_the_drop_window_is_limited():
    trace = [0.5, 0.5, 0.5, 0.0]
    assert ta.largest_window_drop(trace, 1) == pytest.approx(0.5)  # the step 0.5 -> 0.0
    assert ta.largest_window_drop([0.5, 0.45, 0.4, 0.35, 0.0], 2) == pytest.approx(0.4)
    assert ta.largest_window_drop([0.5, 0.45, 0.4, 0.35, 0.3], 1) == pytest.approx(0.05)


@pytest.mark.parametrize("trace, window", [([], 8), ([0.1, 0.2], 0)])
def test_largest_window_drop_rejects_bad_arguments(trace, window):
    with pytest.raises(ValueError):
        ta.largest_window_drop(trace, window)


# ------------------------------------------------------------------ one episode


def test_an_abrupt_episode():
    m = ta.episode_measures(episode(1, A_ABRUPT), window=2)
    assert m["peak_step"] == 2 and m["giveback"] == pytest.approx(0.3)
    assert m["big"] is True and m["abrupt"] is True and m["ended_empty"] is True
    assert m["window_drop"] == pytest.approx(0.3)


def test_a_gradual_episode_is_big_but_not_abrupt():
    m = ta.episode_measures(episode(1, B_GRADUAL), window=2)
    assert m["big"] is True and m["abrupt"] is False and m["ended_empty"] is False
    assert m["giveback"] == pytest.approx(0.15) and m["window_drop"] == pytest.approx(
        0.06
    )


def test_the_default_window_calls_a_fast_slide_abrupt():
    fast = [0.4, 0.36, 0.32, 0.28, 0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.0, 0.0]
    assert (
        ta.episode_measures(episode(1, fast))["abrupt"] is True
    )  # 0.32 within 8 steps
    assert (
        ta.episode_measures(episode(1, fast), window=2)["abrupt"] is False
    )  # 0.08 within 2 steps


def test_no_giveback_means_not_big_and_not_abrupt():
    for trace in (C_STEADY, D_ZERO, E_RISING):
        m = ta.episode_measures(episode(1, trace), window=2)
        assert m["giveback"] == 0.0 and m["big"] is False and m["abrupt"] is False


def test_the_big_threshold_is_strict():
    assert ta.episode_measures(episode(1, [0.1, 0.0]))["big"] is False
    assert ta.episode_measures(episode(1, [0.1, 0.0]), big=0.05)["big"] is True


def test_the_abrupt_threshold_is_inclusive():
    # giveback 0.4, window drop exactly 0.2 = 50% of it
    m = ta.episode_measures(episode(1, [0.5, 0.3, 0.1]), window=1)
    assert m["window_drop"] == pytest.approx(0.2) and m["abrupt"] is True


def test_the_empty_threshold_is_strict():
    assert ta.episode_measures(episode(1, [0.5, 0.001]))["ended_empty"] is False
    assert ta.episode_measures(episode(1, [0.5, 0.0009]))["ended_empty"] is True


def test_a_trace_that_disagrees_with_the_stored_summary_is_refused():
    bad = episode(1, A_ABRUPT)
    bad["max_coverage"] = 0.9
    with pytest.raises(cp.ComparisonError):
        ta.episode_measures(bad)
    bad = episode(1, A_ABRUPT)
    bad["final_coverage"] = 0.5
    with pytest.raises(cp.ComparisonError):
        ta.episode_measures(bad)


@pytest.mark.parametrize("key", ["coverage_trace", "max_coverage", "final_coverage"])
def test_an_episode_missing_a_field_is_refused(key):
    bad = episode(1, A_ABRUPT)
    del bad[key]
    with pytest.raises(cp.ComparisonError):
        ta.episode_measures(bad)


def test_an_empty_trace_is_refused():
    bad = episode(1, A_ABRUPT)
    bad["coverage_trace"] = []
    with pytest.raises(cp.ComparisonError):
        ta.episode_measures(bad)


# ------------------------------------------------------------------ one network


def test_network_measures_known_values():
    n = ta.network_measures(report("net", SIX), window=2)
    assert n["name"] == "net" and n["n_episodes"] == 6 and n["n_big"] == 3
    assert n["mean_max_coverage"] == pytest.approx(2.0 / 6)
    assert n["mean_final_coverage"] == pytest.approx(0.95 / 6)
    assert n["mean_giveback"] == pytest.approx(1.05 / 6)
    assert n["big_giveback_share"] == pytest.approx(0.5)
    assert n["abrupt_share_of_big"] == pytest.approx(2 / 3)
    assert n["median_peak_step_of_big"] == pytest.approx(
        2.0
    )  # peaks at steps 2, 1 and 3
    assert n["ended_empty_share"] == pytest.approx(0.5)  # A, D and F


def test_the_giveback_identity_holds_for_a_network():
    n = ta.network_measures(report("net", SIX), window=2)
    assert n["mean_max_coverage"] - n["mean_final_coverage"] == pytest.approx(
        n["mean_giveback"]
    )


def test_too_few_big_episodes_leave_the_abrupt_share_undefined():
    assert (
        ta.network_measures(report("net", SIX), window=2, min_big=4)[
            "abrupt_share_of_big"
        ]
        is None
    )
    n = ta.network_measures(report("net", [C_STEADY, D_ZERO]), window=2)
    assert (
        n["abrupt_share_of_big"] is None
        and n["median_peak_step_of_big"] is None
        and n["n_big"] == 0
    )


def test_a_report_without_episodes_is_refused():
    with pytest.raises(cp.ComparisonError):
        ta.network_measures({"policy": "x", "episodes": []})


# --------------------------------------------------------------------- comparison


def test_compare_measure_skips_undefined_networks():
    r = ta.compare_measure([0.0, 0.0, None], [1.0, 1.0])
    assert (r["baseline_used"], r["chunk_used"]) == (2, 2)
    assert r["baseline_mean"] == 0.0 and r["chunk_mean"] == 1.0
    assert r["difference"] == pytest.approx(1.0) and r["ci95"] == pytest.approx(
        [1.0, 1.0]
    )
    assert r["baseline_values"] == [0.0, 0.0, None]


def test_compare_measure_needs_two_defined_values_per_group():
    r = ta.compare_measure([0.1, None, None], [1.0, 1.0])
    assert (
        r["difference"] is None
        and r["ci95"] is None
        and r["baseline_mean"] == pytest.approx(0.1)
    )
    assert ta.compare_measure([None, None], [1.0, 1.0])["baseline_mean"] is None


def groups(base_traces, chunk_traces, window=2):
    base = [
        ta.network_measures(report(f"b{i}", base_traces), window=window)
        for i in range(3)
    ]
    chunk = [
        ta.network_measures(report(f"c{i}", chunk_traces), window=window)
        for i in range(3)
    ]
    return base, chunk


def test_the_identity_residual_is_zero_for_any_groups():
    measures = ta.compare_groups(
        *groups([C_STEADY, E_RISING, B_GRADUAL], [A_ABRUPT, F_ABRUPT, D_ZERO])
    )
    assert abs(ta.identity_residual(measures)) < 1e-12
    measures = ta.compare_groups(*groups(SIX, [A_ABRUPT, F_ABRUPT, F_ABRUPT]))
    assert abs(ta.identity_residual(measures)) < 1e-12


def test_the_identity_residual_is_none_without_intervals():
    one = ta.network_measures(report("x", SIX), window=2)
    assert ta.identity_residual(ta.compare_groups([one], [one])) is None


# ------------------------------------------------------------------ reading guide


def measures_with(giveback_ci, abrupt_ci):
    return {
        "mean_giveback": {"ci95": giveback_ci},
        "abrupt_share_of_big": {"ci95": abrupt_ci},
    }


@pytest.mark.parametrize(
    "giveback, abrupt, step1_start, step2_start",
    [
        (
            [0.02, 0.09],
            [0.1, 0.5],
            "chunked networks give back more",
            "the extra loss is more often abrupt",
        ),
        (
            [0.02, 0.09],
            [-0.5, -0.1],
            "chunked networks give back more",
            "the extra loss is more often gradual",
        ),
        (
            [0.02, 0.09],
            [-0.1, 0.5],
            "chunked networks give back more",
            "no clear difference in how",
        ),
        ([0.02, 0.09], None, "chunked networks give back more", "not evaluable"),
        ([-0.01, 0.09], [0.1, 0.5], "no clear difference in coverage", None),
        (
            [0.0, 0.09],
            [0.1, 0.5],
            "no clear difference in coverage",
            None,
        ),  # a lower end of exactly 0 is not above 0
        ([-0.09, -0.02], [0.1, 0.5], "baseline networks give back more", None),
        (None, [0.1, 0.5], "not evaluable", None),
    ],
)
def test_the_reading_guide(giveback, abrupt, step1_start, step2_start):
    g = ta.reading_guide(measures_with(giveback, abrupt))
    assert g["step1"].startswith(step1_start)
    if step2_start is None:
        assert g["step2"] is None
    else:
        assert g["step2"].startswith(step2_start)


# ----------------------------------------------------------------------- analysis


def analysis(base_traces, chunk_traces, n=2, **kw):
    baselines = [report(f"b{i}", base_traces) for i in range(n)]
    chunks = [report(f"c{i}", chunk_traces) for i in range(n)]
    return ta.analyze(baselines, chunks, **kw)


def test_abrupt_chunked_losses_are_reported_as_more_abrupt():
    r = analysis([B_GRADUAL] * 4, [F_ABRUPT] * 4, window=2)
    m = r["measures"]
    assert m["mean_giveback"]["difference"] == pytest.approx(0.6 - 0.15)
    assert (
        m["abrupt_share_of_big"]["baseline_mean"] == 0.0
        and m["abrupt_share_of_big"]["chunk_mean"] == 1.0
    )
    assert r["guide"]["step1"].startswith("chunked networks give back more")
    assert r["guide"]["step2"].startswith("the extra loss is more often abrupt")
    assert abs(r["identity_residual"]) < 1e-12


def test_gradual_chunked_losses_are_reported_as_more_gradual():
    r = analysis([F_ABRUPT] * 4, [B_GRADUAL] * 4, window=2)
    assert r["guide"]["step1"].startswith(
        "baseline networks give back more"
    )  # 0.6 versus 0.15
    assert r["guide"]["step2"] is None
    r = analysis([C_STEADY] * 4, [B_GRADUAL] * 4, window=2)
    assert r["guide"]["step2"].startswith(
        "not evaluable"
    )  # the baseline has no big-giveback episode


def test_equal_networks_show_no_difference():
    r = analysis(SIX, SIX, window=2)
    assert (
        r["guide"]["step1"].startswith("no clear difference")
        and r["measures"]["mean_giveback"]["difference"] == 0.0
    )


def test_the_result_records_settings_names_and_a_pilot_warning():
    r = analysis([B_GRADUAL] * 4, [F_ABRUPT] * 4, window=2)
    assert r["settings"] == {
        "big_giveback": 0.1,
        "window": 2,
        "abrupt_fraction": 0.5,
        "empty_coverage": 0.001,
        "min_big_episodes": 3,
    }
    assert [n["name"] for n in r["baseline_networks"]] == ["b0", "b1"]
    assert r["evaluation"] == {"episodes": 4, "seed_start": 10000, "seed_end": 10003}
    assert any("fewer than 10" in w for w in r["warnings"])
    json.dumps(r)


def test_ten_networks_per_group_carry_no_warning():
    assert analysis(SIX, SIX, n=10, window=2)["warnings"] == []


def test_reports_on_different_seeds_are_refused():
    other = report("c0", SIX, seed_start=20000)
    with pytest.raises(cp.ComparisonError):
        ta.analyze(
            [report("b0", SIX), report("b1", SIX)],
            [other, report("c1", SIX, seed_start=20000)],
        )


def test_the_format_shows_every_measure_the_guide_and_the_note():
    text = ta.format_report(analysis([B_GRADUAL] * 4, [F_ABRUPT] * 4, window=2))
    for expected in (
        "trace analysis:",
        "mean max coverage",
        "mean final coverage",
        "coverage given back",
        "share of episodes giving back more than 0.1",
        "abrupt drop",
        "median step of the peak",
        "ending with coverage below 0.001",
        "identity:",
        "reading guide, step 1",
        "reading guide, step 2",
        "thresholds fixed in advance",
        "WARNING",
        "note:",
    ):
        assert expected in text


def test_the_format_handles_undefined_numbers():
    text = ta.format_report(analysis([C_STEADY] * 4, [C_STEADY] * 4, window=2))
    assert "n/a" in text


# ------------------------------------------------------------------ command line


def write_all(directory, base_traces, chunk_traces, n=2, verified=True):
    paths = {"baseline": [], "chunk": []}
    for kind, traces, prefix in (
        ("baseline", base_traces, "b"),
        ("chunk", chunk_traces, "c"),
    ):
        for i in range(n):
            rep = report(f"{prefix}{i}", traces)
            path = directory / f"eval_{prefix}{i}.json"
            path.write_text(json.dumps(rep))
            paths[kind].append(str(path))
            if verified:
                (directory / f"reproducibility_{prefix}{i}.json").write_text(
                    json.dumps({"reproduced": True})
                )
    return paths


def run_main(directory, paths, *extra):
    out = directory / "out.json"
    code = ta.main(
        [
            "--baseline",
            *paths["baseline"],
            "--chunk",
            *paths["chunk"],
            "--report-dir",
            str(directory),
            "--out",
            str(out),
            *extra,
        ]
    )
    return code, out


def test_main_writes_the_analysis_and_prints_it(tmp_path, capsys):
    paths = write_all(tmp_path, [B_GRADUAL] * 4, [F_ABRUPT] * 4)
    code, out = run_main(tmp_path, paths)
    assert code == 0
    saved = json.loads(out.read_text())
    assert saved["unverified_networks"] == [] and saved["guide"]["step1"]
    assert "trace analysis:" in capsys.readouterr().out


def test_main_refuses_unverified_networks_unless_told_otherwise(tmp_path, capsys):
    paths = write_all(tmp_path, [B_GRADUAL] * 4, [F_ABRUPT] * 4)
    (tmp_path / "reproducibility_c1.json").unlink()
    code, out = run_main(tmp_path, paths)
    assert code == 1 and not out.exists() and "c1" in capsys.readouterr().err
    code, out = run_main(tmp_path, paths, "--allow-unverified")
    saved = json.loads(out.read_text())
    assert (
        code == 0
        and saved["unverified_networks"] == ["c1"]
        and any("c1" in w for w in saved["warnings"])
    )


def test_main_reports_a_missing_file_as_an_error(tmp_path, capsys):
    code = ta.main(
        [
            "--baseline",
            str(tmp_path / "no.json"),
            "--chunk",
            str(tmp_path / "no2.json"),
            "--out",
            str(tmp_path / "o.json"),
        ]
    )
    assert code == 1 and "error:" in capsys.readouterr().err


def test_the_default_paths_follow_the_naming_of_the_seed_study():
    baseline, chunk = ta.default_paths(Path("r"), seeds=range(3))
    assert [p.name for p in baseline] == [
        "eval_bc_cnn.json",
        "eval_bc_cnn_s1.json",
        "eval_bc_cnn_s2.json",
    ]
    assert [p.name for p in chunk] == [
        "eval_chunk_cnn.json",
        "eval_chunk_cnn_s1.json",
        "eval_chunk_cnn_s2.json",
    ]
    assert len(ta.default_paths()[0]) == 10


# --------------------------------------------------------- the real saved reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"
BASELINE_PATHS, CHUNK_PATHS = ta.default_paths(REPORTS)
needs_study = pytest.mark.skipif(
    not all(p.exists() for p in BASELINE_PATHS + CHUNK_PATHS),
    reason="run the seed study first",
)


@pytest.fixture(scope="module")
def real():
    base = [cp.load_report(p) for p in BASELINE_PATHS]
    chunk = [cp.load_report(p) for p in CHUNK_PATHS]
    return base, chunk, ta.analyze(base, chunk)


@needs_study
def test_every_real_trace_agrees_with_its_stored_summary(real):
    base, chunk, _ = real
    for rep in base + chunk:
        for e in rep["episodes"]:
            m = ta.episode_measures(e)  # raises if the trace and the summary disagree
            assert 1 <= m["peak_step"] <= e["steps"] and m["giveback"] >= 0.0


@needs_study
def test_the_real_giveback_equals_max_minus_final_for_every_network(real):
    base, chunk, result = real
    for rep, net in zip(
        base + chunk, result["baseline_networks"] + result["chunk_networks"]
    ):
        s = rep["summary"]
        assert net["mean_giveback"] == pytest.approx(
            s["mean_max_coverage"] - s["mean_final_coverage"], abs=1e-9
        )


@needs_study
def test_the_real_group_means_reproduce_the_seed_study(real):
    m = real[2]["measures"]
    assert m["mean_max_coverage"]["baseline_mean"] == pytest.approx(0.0979, abs=5e-4)
    assert m["mean_max_coverage"]["chunk_mean"] == pytest.approx(0.1474, abs=5e-4)
    assert m["mean_final_coverage"]["baseline_mean"] == pytest.approx(0.0666, abs=5e-4)
    assert m["mean_final_coverage"]["chunk_mean"] == pytest.approx(0.0448, abs=5e-4)
    assert m["mean_max_coverage"]["difference"] == pytest.approx(0.0495, abs=5e-4)
    assert m["mean_final_coverage"]["difference"] == pytest.approx(-0.0218, abs=5e-4)
    assert m["mean_giveback"]["baseline_mean"] == pytest.approx(
        0.0979 - 0.0666, abs=1e-3
    )
    assert m["mean_giveback"]["chunk_mean"] == pytest.approx(0.1474 - 0.0448, abs=1e-3)
    assert abs(real[2]["identity_residual"]) < 1e-9 and real[2]["warnings"] == []


@needs_study
def test_the_do_nothing_reports_give_back_nothing():
    hold = REPORTS / "eval_hold.json"
    if not hold.exists():
        pytest.skip("run evaluate.py --policy hold first")
    n = ta.network_measures(cp.load_report(hold))
    assert n["mean_giveback"] == pytest.approx(0.0, abs=1e-9) and n["n_big"] == 0
