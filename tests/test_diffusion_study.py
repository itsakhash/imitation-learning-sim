"""Tests for diffusion_study.py. No torch needed: the reports are made-up evaluation reports."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import arm_train
import compare_policies as cp
import diffusion_study as ds
import evaluate as ev


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


def arm_reports(arm, maxes, finals, seeds=None, start=0.05):
    seeds = list(range(len(maxes))) if seeds is None else seeds
    return [net(arm_train.default_name(arm, s), m, f, start) for s, m, f in zip(seeds, maxes, finals)]


def spread(center, n=10, width=0.02):
    return list(center + np.linspace(-width, width, n))


# ---------------------------------------------------------------- names and values


def test_original_chunked_names_follow_the_project_convention():
    assert ds.original_name(0) == "chunk_cnn" and ds.original_name(12) == "chunk_cnn_s12"
    assert ds.report_path("r", "diff_cnn_s3") == Path("r") / "eval_diff_cnn_s3.json"


def test_blocks_are_the_preregistered_seed_ranges():
    assert ds.BLOCKS[1] == tuple(range(10)) and ds.BLOCKS[2] == tuple(range(10, 20))


def test_giveback_is_max_minus_final():
    v = ds.values(net("x", 0.30, 0.12))
    assert v["mean_giveback"] == pytest.approx(0.18)
    assert v["mean_max_coverage"] == pytest.approx(0.30) and v["mean_final_coverage"] == pytest.approx(0.12)


# ---------------------------------------------------------------- paired bootstrap


def test_paired_bootstrap_is_second_minus_first_and_contains_the_estimate():
    a = np.linspace(0.0, 1.0, 10)
    b = a + 0.5 + np.linspace(-0.05, 0.05, 10)
    d, lo, hi = ds.paired_bootstrap(a, b)
    assert d == pytest.approx(0.5) and lo < d < hi


def test_paired_bootstrap_without_spread_is_degenerate():
    d, lo, hi = ds.paired_bootstrap([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    assert (d, lo, hi) == pytest.approx((-1.0, -1.0, -1.0))


def test_pairing_removes_the_between_network_spread():
    a = np.linspace(0.0, 10.0, 10)  # large differences between networks ...
    b = a + 0.1 + np.linspace(-0.01, 0.01, 10)  # ... but the same shift for each
    _, plo, phi = ds.paired_bootstrap(a, b)
    independent = ds.difference([{"m": x} for x in a], [{"m": x} for x in b], "m")["ci95"]
    assert (phi - plo) < (independent[1] - independent[0])


def test_paired_bootstrap_is_deterministic_and_validates():
    a, b = np.linspace(0, 1, 8), np.linspace(0.2, 1.1, 8)
    assert ds.paired_bootstrap(a, b) == ds.paired_bootstrap(a, b)
    assert ds.paired_bootstrap(a, b, seed=1) != ds.paired_bootstrap(a, b, seed=2)
    for bad in (([1.0], [1.0]), ([1.0, 2.0], [1.0]), ([[1.0, 2.0]], [[1.0, 2.0]])):
        with pytest.raises(ValueError):
            ds.paired_bootstrap(*bad)
    with pytest.raises(ValueError):
        ds.paired_bootstrap(a, b, n_resamples=10)


def test_difference_is_b_minus_a():
    r = ds.difference([{"m": 1.0}, {"m": 1.0}], [{"m": 3.0}, {"m": 3.0}], "m")
    assert r["difference"] == pytest.approx(2.0) and r["a_mean"] == 1.0 and r["b_mean"] == 3.0


# ---------------------------------------------------------------- verdicts


def study(d_max, d_final, c_max=0.15, c_final=0.05, **extra):
    reports = {
        "D": arm_reports("D", spread(d_max), spread(d_final)),
        "C": arm_reports("C", spread(c_max), spread(c_final)),
    }
    reports.update(extra)
    return reports


def test_supported_when_both_intervals_are_above_zero_and_the_direction_is_d_minus_c():
    r = ds.analyze(study(0.30, 0.15))
    assert r["verdict"] == "supported"
    assert r["primary"]["mean_max_coverage"]["difference"] == pytest.approx(0.15)
    assert r["primary"]["mean_final_coverage"]["difference"] == pytest.approx(0.10)
    assert r["primary"]["comparison"] == "D minus C"


def test_contradicted_when_either_interval_is_below_zero():
    assert ds.analyze(study(0.30, 0.01))["verdict"] == "contradicted"  # final below C's
    assert ds.analyze(study(0.05, 0.15))["verdict"] == "contradicted"  # max below C's


def test_inconclusive_when_an_interval_includes_zero():
    assert ds.analyze(study(0.15, 0.15))["verdict"] == "inconclusive"  # max equal, final above
    r = ds.analyze(study(0.30, 0.05))  # final equal
    assert r["verdict"] == "inconclusive"


def test_d_and_c_are_required():
    with pytest.raises(cp.ComparisonError):
        ds.analyze({"D": arm_reports("D", spread(0.3), spread(0.1))})
    with pytest.raises(cp.ComparisonError):
        ds.analyze({"C": arm_reports("C", spread(0.3), spread(0.1)), "D": []})


def test_fewer_than_ten_networks_is_labelled_a_pilot():
    reports = {
        "D": arm_reports("D", spread(0.3, 4), spread(0.1, 4)),
        "C": arm_reports("C", spread(0.2, 4), spread(0.05, 4)),
    }
    assert any("pilot" in w for w in ds.analyze(reports)["warnings"])
    assert ds.analyze(study(0.3, 0.1))["warnings"] == []


def test_reports_must_share_seeds_and_names_must_differ():
    odd = net("C_odd", 0.2, 0.05, seed_start=20000)
    reports = study(0.3, 0.1)
    reports["C"] = reports["C"][:-1] + [odd]
    with pytest.raises(cp.ComparisonError):
        ds.analyze(reports)
    reports = study(0.3, 0.1)
    reports["C"][0] = net("diff_cnn", 0.2, 0.05)  # the same name as a D network
    with pytest.raises(cp.ComparisonError):
        ds.analyze(reports)


# ---------------------------------------------------------------- descriptive parts


def test_arm_summary_counts_networks_above_the_do_nothing_level():
    r = ds.analyze(study(0.30, 0.15, c_final=0.01))  # start coverage 0.05: D above, C below
    assert r["arms"]["D"]["do_nothing"]["above"] == 10 and r["arms"]["C"]["do_nothing"]["above"] == 0
    assert r["arms"]["D"]["mean_giveback"]["mean"] == pytest.approx(0.15)
    assert len(r["arms"]["D"]["mean_final_coverage"]["values"]) == 10


def test_descriptive_comparisons_with_the_baseline_appear_only_when_b_exists():
    assert "D_minus_B" not in ds.analyze(study(0.3, 0.1))["descriptive"]
    b = arm_reports("B", spread(0.10), spread(0.08))
    r = ds.analyze(study(0.30, 0.15, B=b))
    d = r["descriptive"]
    assert d["D_minus_B"]["mean_max_coverage"]["difference"] == pytest.approx(0.20)
    assert d["C_minus_B"]["mean_final_coverage"]["difference"] == pytest.approx(-0.03)
    assert d["D_minus_B"]["mean_giveback"]["difference"] == pytest.approx(0.20 - 0.07)


def test_original_chunked_networks_give_the_confounded_and_the_paired_comparisons():
    originals = [net(ds.original_name(s), 0.12, 0.04) for s in range(10)]
    p = arm_reports("P", [0.14 + 0.001 * s for s in range(10)], [0.05 + 0.001 * s for s in range(10)])
    r = ds.analyze(study(0.30, 0.15, P=p), originals)
    d = r["descriptive"]
    assert d["C_minus_original_chunked"]["mean_max_coverage"]["difference"] == pytest.approx(0.03)
    paired = d["P_minus_original_chunked_paired"]["mean_max_coverage"]
    assert paired["difference"] == pytest.approx(0.0245)
    assert paired["ci95"][0] < paired["difference"] < paired["ci95"][1]
    assert "original_chunked" in d


def test_the_paired_comparison_needs_matching_numbers_of_networks():
    originals = [net(ds.original_name(s), 0.12, 0.04) for s in range(10)]
    p = arm_reports("P", spread(0.14, 5), spread(0.05, 5))
    assert "P_minus_original_chunked_paired" not in ds.analyze(study(0.3, 0.1, P=p), originals)["descriptive"]


def test_format_names_the_verdict_and_the_descriptive_parts():
    b = arm_reports("B", spread(0.10), spread(0.08))
    text = ds.format_block(ds.analyze(study(0.30, 0.15, B=b)), "block 1")
    assert "verdict (D against C): supported" in text and "D minus B (descriptive)" in text
    assert "block 1" in text and "never pooled" in text


# ---------------------------------------------------------------- the command line


def write_block(tmp_path, seeds=range(10), reproduced=True, with_b=True, with_originals=True):
    def put(report):
        (tmp_path / f"eval_{report['policy']}.json").write_text(json.dumps(report))
        (tmp_path / f"reproducibility_{report['policy']}.json").write_text(json.dumps({"reproduced": reproduced}))

    for s in seeds:
        i = s % 10
        put(net(arm_train.default_name("D", s), 0.30 + 0.002 * i, 0.15 + 0.002 * i))
        put(net(arm_train.default_name("C", s), 0.15 + 0.002 * i, 0.05 + 0.002 * i))
        if with_b:
            put(net(arm_train.default_name("B", s), 0.10 + 0.002 * i, 0.08 + 0.002 * i))
        if with_originals:
            put(net(ds.original_name(s), 0.12 + 0.002 * i, 0.04))


def test_main_analyses_a_block_and_saves_the_result(tmp_path, capsys):
    write_block(tmp_path)
    assert ds.main(["--block", "1", "--report-dir", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "diffusion_study_block1.json").read_text())
    assert saved["verdict"] == "supported" and saved["networks_per_arm"] == {"D": 10, "C": 10, "B": 10}
    assert "C_minus_original_chunked" in saved["descriptive"] and saved["unverified_networks"] == []
    assert "verdict (D against C): supported" in capsys.readouterr().out


def test_main_block_2_uses_seeds_10_to_19(tmp_path):
    write_block(tmp_path, seeds=range(10, 20), with_b=False, with_originals=False)
    assert ds.main(["--block", "2", "--report-dir", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "diffusion_study_block2.json").read_text())
    assert saved["networks_per_arm"] == {"D": 10, "C": 10} and saved["seeds"] == list(range(10, 20))
    assert "diff_cnn_s19" in saved["arms"]["D"]["networks"]


def test_main_refuses_unverified_networks_unless_told_otherwise(tmp_path, capsys):
    write_block(tmp_path, reproduced=False, with_b=False, with_originals=False)
    assert ds.main(["--block", "1", "--report-dir", str(tmp_path)]) == 1
    assert "no passing reproducibility record" in capsys.readouterr().err
    assert not (tmp_path / "diffusion_study_block1.json").exists()
    assert ds.main(["--block", "1", "--report-dir", str(tmp_path), "--allow-unverified"]) == 0
    saved = json.loads((tmp_path / "diffusion_study_block1.json").read_text())
    assert len(saved["unverified_networks"]) == 20 and any("reproducibility" in w for w in saved["warnings"])


def test_main_fails_clearly_when_an_evaluation_is_missing(tmp_path, capsys):
    write_block(tmp_path, with_b=False, with_originals=False)
    (tmp_path / "eval_diff_cnn_s4.json").unlink()
    assert ds.main(["--block", "1", "--report-dir", str(tmp_path)]) == 1
    assert "no such report" in capsys.readouterr().err


def test_main_needs_a_block_or_seeds_but_not_both():
    with pytest.raises(SystemExit):
        ds.main([])
    with pytest.raises(SystemExit):
        ds.main(["--block", "1", "--seeds", "1"])


def test_main_refuses_networks_with_no_reproducibility_record_at_all(tmp_path, capsys):
    write_block(tmp_path, with_b=False, with_originals=False)
    (tmp_path / "reproducibility_diff_cnn_s2.json").unlink()
    assert ds.main(["--block", "1", "--report-dir", str(tmp_path)]) == 1
    assert "diff_cnn_s2" in capsys.readouterr().err
