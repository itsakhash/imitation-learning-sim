import json
from pathlib import Path

import numpy as np
import pytest

import compare_policies as cp
import evaluate as ev
import seed_study as ss


def net(name, max_cov, final, start=0.05, n=3, seed_start=10000):
    episodes = [
        {
            "seed": seed_start + i,
            "steps": 300,
            "start_coverage": start,
            "max_coverage": max_cov,
            "final_coverage": final,
            "success": False,
            "terminated": False,
            "truncated": True,
            "coverage_trace": [max_cov],
        }
        for i in range(n)
    ]
    return ev.build_report(name, episodes)


def group(prefix, maxes, finals):
    return [net(f"{prefix}{i}", m, f) for i, (m, f) in enumerate(zip(maxes, finals))]


# ------------------------------------------------------------ bootstrap difference


def test_the_difference_is_second_group_minus_first():
    diff, low, high = ss.bootstrap_difference([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
    assert (
        diff == pytest.approx(-0.5)
        and low == pytest.approx(-0.5)
        and high == pytest.approx(-0.5)
    )


def test_groups_without_spread_have_a_degenerate_interval():
    diff, low, high = ss.bootstrap_difference([0.0, 0.0], [1.0, 1.0])
    assert (diff, low, high) == pytest.approx((1.0, 1.0, 1.0))


def test_the_interval_contains_the_observed_difference():
    a = np.linspace(0.0, 1.0, 10)
    b = np.linspace(0.3, 1.5, 10)
    diff, low, high = ss.bootstrap_difference(a, b)
    assert low < diff < high and diff == pytest.approx(b.mean() - a.mean())


def test_the_bootstrap_is_deterministic_for_a_seed_and_depends_on_it():
    a, b = np.linspace(0.0, 1.0, 8), np.linspace(0.2, 1.4, 8)
    assert ss.bootstrap_difference(a, b, seed=3) == ss.bootstrap_difference(
        a, b, seed=3
    )
    assert ss.bootstrap_difference(a, b, seed=3) != ss.bootstrap_difference(
        a, b, seed=4
    )


def test_more_networks_give_a_narrower_interval():
    rng = np.random.default_rng(0)
    _, lo_a, hi_a = ss.bootstrap_difference(rng.normal(0, 1, 5), rng.normal(0, 1, 5))
    _, lo_b, hi_b = ss.bootstrap_difference(
        rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    )
    assert (hi_b - lo_b) < (hi_a - lo_a)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(a=[1.0], b=[1.0, 2.0]),
        dict(a=[1.0, 2.0], b=[1.0]),
        dict(a=[[1.0, 2.0]], b=[1.0, 2.0]),
        dict(a=[1.0, 2.0], b=[1.0, 2.0], confidence=1.0),
        dict(a=[1.0, 2.0], b=[1.0, 2.0], confidence=0.0),
        dict(a=[1.0, 2.0], b=[1.0, 2.0], n_resamples=10),
    ],
)
def test_bootstrap_difference_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        ss.bootstrap_difference(**kwargs)


# ----------------------------------------------------------------------- verdict


@pytest.mark.parametrize(
    "intervals, expected",
    [
        ({"a": (0.1, 0.3), "b": (0.2, 0.4)}, "supported"),
        ({"a": (-0.1, 0.3), "b": (0.2, 0.4)}, "inconclusive"),
        ({"a": (0.1, 0.3), "b": (-0.5, 0.1)}, "inconclusive"),
        ({"a": (0.1, 0.3), "b": (-0.5, -0.1)}, "contradicted"),
        ({"a": (-0.5, -0.1), "b": (0.2, 0.4)}, "contradicted"),
        ({"a": (-0.5, -0.1), "b": (-0.5, -0.1)}, "contradicted"),
        (
            {"a": (0.0, 0.3), "b": (0.2, 0.4)},
            "inconclusive",
        ),  # a lower bound of exactly 0 is not above 0
        (
            {"a": (-0.3, 0.0), "b": (-0.3, 0.4)},
            "inconclusive",
        ),  # an upper bound of exactly 0 is not below 0
    ],
)
def test_verdict_follows_the_pre_registered_rule(intervals, expected):
    assert ss.verdict(intervals) == expected


def test_verdict_needs_intervals():
    with pytest.raises(ValueError):
        ss.verdict({})


# ------------------------------------------------------------- per-network values


def test_network_values_come_from_the_summary():
    v = ss.network_values(net("x", 0.3, 0.2, start=0.05))
    assert v["name"] == "x"
    assert v["mean_max_coverage"] == pytest.approx(0.3)
    assert v["mean_final_coverage"] == pytest.approx(0.2)
    assert v["mean_start_coverage"] == pytest.approx(0.05)


def test_a_summary_without_start_coverage_is_refused():
    report = net("x", 0.3, 0.2)
    report["summary"]["mean_start_coverage"] = None
    with pytest.raises(cp.ComparisonError):
        ss.network_values(report)


def test_a_report_without_a_summary_is_refused():
    with pytest.raises(cp.ComparisonError):
        ss.network_values({"policy": "x"})


# ---------------------------------------------------------------------- secondary


def test_above_donothing_known_values():
    values = [
        {"mean_final_coverage": 0.10, "mean_start_coverage": 0.05},
        {"mean_final_coverage": 0.00, "mean_start_coverage": 0.05},
        {"mean_final_coverage": 0.06, "mean_start_coverage": 0.05},
    ]
    s = ss.above_donothing(values)
    assert (s["above"], s["networks"]) == (2, 3)
    assert s["above_wilson95"] == list(ev.wilson_ci(2, 3))
    assert s["mean_difference"] == pytest.approx((0.05 - 0.05 + 0.01) / 3)
    lo, hi = s["mean_difference_ci95"]
    assert lo <= s["mean_difference"] <= hi


def test_a_network_exactly_at_the_donothing_level_is_not_above_it():
    s = ss.above_donothing(
        [{"mean_final_coverage": 0.05, "mean_start_coverage": 0.05}] * 2
    )
    assert s["above"] == 0


# ----------------------------------------------------------------------- analysis


def test_clearly_higher_chunked_networks_support_chunking():
    r = ss.analyze(
        group("b", [0.10, 0.11, 0.09], [0.05, 0.06, 0.04]),
        group("c", [0.30, 0.31, 0.29], [0.08, 0.09, 0.07]),
    )
    assert r["verdict"] == "supported"
    assert r["primary"]["mean_max_coverage"]["difference"] == pytest.approx(0.20)
    assert r["primary"]["mean_final_coverage"]["difference"] == pytest.approx(0.03)


def test_a_clearly_lower_final_coverage_contradicts_chunking_whatever_the_peak_is():
    r = ss.analyze(
        group("b", [0.10, 0.11, 0.09], [0.05, 0.06, 0.04]),
        group("c", [0.30, 0.31, 0.29], [0.01, 0.02, 0.00]),
    )
    assert r["verdict"] == "contradicted"


def test_overlapping_groups_are_inconclusive():
    same = [0.1, 0.2, 0.3]
    r = ss.analyze(group("b", same, same), group("c", same, same))
    assert r["verdict"] == "inconclusive"
    assert r["primary"]["mean_max_coverage"]["difference"] == pytest.approx(0.0)


def test_the_result_records_names_ranges_and_the_evaluation():
    r = ss.analyze(
        group("b", [0.1, 0.3], [0.1, 0.2]), group("c", [0.2, 0.6], [0.1, 0.2])
    )
    assert r["baseline_networks"] == ["b0", "b1"] and r["chunk_networks"] == [
        "c0",
        "c1",
    ]
    assert r["evaluation"] == {"episodes": 3, "seed_start": 10000, "seed_end": 10002}
    p = r["primary"]["mean_max_coverage"]
    assert p["baseline_range"] == pytest.approx(0.2) and p[
        "chunk_range"
    ] == pytest.approx(0.4)
    json.dumps(r)


def test_groups_of_different_sizes_are_allowed():
    r = ss.analyze(
        group("b", [0.1, 0.2], [0.1, 0.1]), group("c", [0.2, 0.3, 0.4], [0.1, 0.1, 0.1])
    )
    assert (r["n_baseline"], r["n_chunk"]) == (2, 3)


def test_fewer_than_ten_networks_is_labelled_a_pilot():
    r = ss.analyze(
        group("b", [0.1, 0.2], [0.1, 0.1]), group("c", [0.2, 0.3], [0.1, 0.1])
    )
    assert any("pilot" in w for w in r["warnings"])


def test_ten_networks_per_group_carry_no_pilot_warning():
    values = [0.1 + 0.01 * i for i in range(10)]
    r = ss.analyze(group("b", values, values), group("c", values, values))
    assert r["warnings"] == []


def test_a_single_network_per_group_is_refused():
    with pytest.raises(ValueError):
        ss.analyze([net("b0", 0.1, 0.1)], [net("c0", 0.2, 0.1)])


def test_reports_on_different_seeds_are_refused():
    with pytest.raises(cp.ComparisonError):
        ss.analyze(
            [net("b0", 0.1, 0.1), net("b1", 0.1, 0.1)],
            [
                net("c0", 0.2, 0.1, seed_start=20000),
                net("c1", 0.2, 0.1, seed_start=20000),
            ],
        )


def test_two_networks_with_the_same_name_are_refused():
    with pytest.raises(cp.ComparisonError):
        ss.analyze(
            [net("x", 0.1, 0.1), net("y", 0.1, 0.1)],
            [net("x", 0.2, 0.1), net("z", 0.2, 0.1)],
        )


def test_no_reports_is_refused():
    with pytest.raises(cp.ComparisonError):
        ss.check_all_comparable([])


def test_format_shows_the_numbers_the_verdict_and_the_note():
    r = ss.analyze(
        group("b", [0.1, 0.12], [0.05, 0.06]), group("c", [0.3, 0.32], [0.08, 0.09])
    )
    text = ss.format_study(r)
    for expected in (
        "seed study:",
        "mean_max_coverage",
        "mean_final_coverage",
        "baseline networks:",
        "chunked networks:",
        "verdict:",
        "do-nothing level",
        "WARNING: fewer than 10",
        "note:",
    ):
        assert expected in text


# ----------------------------------------------------------------- command line


def write_group(directory, reports, verified=True):
    paths = []
    for r in reports:
        path = directory / f"eval_{r['policy']}.json"
        path.write_text(json.dumps(r))
        paths.append(str(path))
        if verified:
            (directory / f"reproducibility_{r['policy']}.json").write_text(
                json.dumps({"reproduced": True})
            )
    return paths


def run_main(directory, baselines, chunks, *extra):
    out = directory / "study.json"
    code = ss.main(
        [
            "--baseline",
            *baselines,
            "--chunk",
            *chunks,
            "--report-dir",
            str(directory),
            "--out",
            str(out),
            *extra,
        ]
    )
    return code, out


def test_main_writes_the_study_and_prints_it(tmp_path, capsys):
    b = write_group(tmp_path, group("b", [0.1, 0.2], [0.1, 0.1]))
    c = write_group(tmp_path, group("c", [0.2, 0.4], [0.1, 0.2]))
    code, out = run_main(tmp_path, b, c)
    assert code == 0
    saved = json.loads(out.read_text())
    assert saved["unverified_networks"] == [] and saved["verdict"] in {
        "supported",
        "contradicted",
        "inconclusive",
    }
    assert "seed study:" in capsys.readouterr().out


def test_main_refuses_a_network_without_a_passing_reproducibility_record(
    tmp_path, capsys
):
    b = write_group(tmp_path, group("b", [0.1, 0.2], [0.1, 0.1]))
    c = write_group(tmp_path, group("c", [0.2, 0.4], [0.1, 0.2]))
    (tmp_path / "reproducibility_c1.json").unlink()
    code, out = run_main(tmp_path, b, c)
    assert code == 1 and not out.exists()
    assert "c1" in capsys.readouterr().err


def test_a_failed_reproducibility_record_counts_as_unverified(tmp_path, capsys):
    b = write_group(tmp_path, group("b", [0.1, 0.2], [0.1, 0.1]))
    c = write_group(tmp_path, group("c", [0.2, 0.4], [0.1, 0.2]))
    (tmp_path / "reproducibility_b0.json").write_text(json.dumps({"reproduced": False}))
    assert run_main(tmp_path, b, c)[0] == 1
    assert "b0" in capsys.readouterr().err


def test_allow_unverified_analyses_anyway_and_records_the_warning(tmp_path):
    b = write_group(tmp_path, group("b", [0.1, 0.2], [0.1, 0.1]))
    c = write_group(tmp_path, group("c", [0.2, 0.4], [0.1, 0.2]))
    (tmp_path / "reproducibility_c1.json").unlink()
    code, out = run_main(tmp_path, b, c, "--allow-unverified")
    saved = json.loads(out.read_text())
    assert code == 0 and saved["unverified_networks"] == ["c1"]
    assert any("c1" in w for w in saved["warnings"])


def test_main_reports_a_single_network_per_group_as_an_error(tmp_path, capsys):
    b = write_group(tmp_path, [net("b0", 0.1, 0.1)])
    c = write_group(tmp_path, [net("c0", 0.2, 0.1)])
    assert run_main(tmp_path, b, c)[0] == 1
    assert "error:" in capsys.readouterr().err


def test_main_reports_a_missing_file_as_an_error(tmp_path, capsys):
    code = ss.main(
        [
            "--baseline",
            str(tmp_path / "nope.json"),
            "--chunk",
            str(tmp_path / "nope2.json"),
            "--out",
            str(tmp_path / "o.json"),
        ]
    )
    assert code == 1 and "error:" in capsys.readouterr().err


# ------------------------------------------------------------ the real saved reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"
BASELINES = ["bc_cnn", "bc_cnn_s1", "bc_cnn_s2"]
CHUNKS = ["chunk_cnn", "chunk_cnn_s1", "chunk_cnn_s2"]
needs_reports = pytest.mark.skipif(
    not all((REPORTS / f"eval_{n}.json").exists() for n in BASELINES + CHUNKS),
    reason="run the three-seed evaluations first",
)


@needs_reports
def test_the_three_seed_pilot_reproduces_the_group_means_already_seen():
    r = ss.analyze(
        [cp.load_report(REPORTS / f"eval_{n}.json") for n in BASELINES],
        [cp.load_report(REPORTS / f"eval_{n}.json") for n in CHUNKS],
    )
    assert (r["n_baseline"], r["n_chunk"]) == (3, 3) and any(
        "pilot" in w for w in r["warnings"]
    )
    mx, fin = r["primary"]["mean_max_coverage"], r["primary"]["mean_final_coverage"]
    assert mx["baseline_mean"] == pytest.approx(0.1015, abs=5e-4)
    assert mx["chunk_mean"] == pytest.approx(0.1734, abs=5e-4)
    assert mx["difference"] == pytest.approx(0.0719, abs=5e-4)
    assert fin["baseline_mean"] == pytest.approx(0.0700, abs=5e-4)
    assert fin["chunk_mean"] == pytest.approx(0.0485, abs=5e-4)
    assert fin["difference"] == pytest.approx(-0.0214, abs=5e-4)
    assert (r["secondary"]["baseline"]["above"], r["secondary"]["chunk"]["above"]) == (
        2,
        1,
    )
