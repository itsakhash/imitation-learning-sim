import json
from pathlib import Path

import numpy as np
import pytest

import ablation_study as ab
import compare_policies as cp
import evaluate as ev
import seed_study as ss


def net(name, max_cov, final, start=0.05, n=3, seed_start=10000):
    episodes = [
        {"seed": seed_start + i, "steps": 300, "start_coverage": start, "max_coverage": max_cov,
         "final_coverage": final, "success": False, "terminated": False, "truncated": True,
         "coverage_trace": [max_cov]}
        for i in range(n)
    ]
    return ev.build_report(name, episodes)


def vals(start, step, n):
    return [(start + step * i, start + step * i) for i in range(n)]


def make_cells(spec, seed_start=10000):
    """spec: {method: {size: [(max, final), ...]}} -> {method: {size: [reports]}}."""
    return {
        method: {
            size: [net(ab.network_name(method, size, i), m, f, seed_start=seed_start)
                   for i, (m, f) in enumerate(rows)]
            for size, rows in by_size.items()
        }
        for method, by_size in spec.items()
    }


def spec_of(base_small, base_full, chunk_small, chunk_full):
    return {"baseline": {20: base_small, 165: base_full}, "chunk": {20: chunk_small, 165: chunk_full}}


SAME_SMALL = [(0.10, 0.10), (0.12, 0.12), (0.14, 0.14), (0.16, 0.16), (0.18, 0.18)]
SAME_FULL = SAME_SMALL * 2


# ----------------------------------------------------------------- names and paths

def test_network_names():
    assert ab.network_name("baseline", 165, 0) == "bc_cnn"
    assert ab.network_name("baseline", 165, 3) == "bc_cnn_s3"
    assert ab.network_name("chunk", 165, 0) == "chunk_cnn"
    assert ab.network_name("chunk", 165, 9) == "chunk_cnn_s9"
    assert ab.network_name("chunk", 20, 0) == "chunk_cnn_n20_s0"
    assert ab.network_name("baseline", 100, 4) == "bc_cnn_n100_s4"


def test_default_paths_have_five_small_and_ten_full_networks_per_method():
    paths = ab.default_paths(Path("r"))
    assert set(paths) == {"baseline", "chunk"}
    for method in paths:
        assert sorted(paths[method]) == [20, 50, 100, 165]
        assert [len(paths[method][s]) for s in (20, 50, 100, 165)] == [5, 5, 5, 10]
    assert paths["baseline"][20][0].name == "eval_bc_cnn_n20_s0.json"
    assert paths["chunk"][100][4].name == "eval_chunk_cnn_n100_s4.json"
    assert paths["baseline"][165][0].name == "eval_bc_cnn.json"
    assert paths["chunk"][165][9].name == "eval_chunk_cnn_s9.json"


def test_default_paths_can_be_restricted():
    paths = ab.default_paths(Path("r"), sizes=(20, 165), small_seeds=2, full_seeds=3)
    assert [len(paths["baseline"][s]) for s in (20, 165)] == [2, 3]
    assert 50 not in paths["baseline"]


# ------------------------------------------------------------------------ verdict

@pytest.mark.parametrize(
    "interval, expected",
    [
        ((0.01, 0.05), "more data helps"),
        ((-0.05, -0.01), "more data hurts"),
        ((-0.01, 0.05), "no clear effect"),
        ((-0.05, 0.01), "no clear effect"),
        ((0.0, 0.05), "no clear effect"),    # a lower bound of exactly 0 is not above 0
        ((-0.05, 0.0), "no clear effect"),   # an upper bound of exactly 0 is not below 0
    ],
)
def test_effect_verdict_follows_the_pre_registered_rule(interval, expected):
    assert ab.effect_verdict(interval) == expected


# --------------------------------------------------------------------- cell summary

def test_cell_summary_known_values():
    reports = [net("a", 0.10, 0.04), net("b", 0.20, 0.08), net("c", 0.30, 0.12)]
    c = ab.cell_summary(reports)
    assert c["networks"] == 3 and c["names"] == ["a", "b", "c"]
    assert c["mean_max_coverage"]["mean"] == pytest.approx(0.20)
    assert c["mean_final_coverage"]["mean"] == pytest.approx(0.08)
    lo, hi = c["mean_max_coverage"]["ci95"]
    assert lo <= 0.20 <= hi
    assert c["mean_final_coverage"]["values"] == pytest.approx([0.04, 0.08, 0.12])
    # the start coverage is 0.05, so two of the three final coverages (0.08 and 0.12) are above it
    assert c["above_donothing"]["above"] == 2 and c["above_donothing"]["networks"] == 3
    assert c["success_episodes"] == 0


def test_cell_summary_counts_successes():
    report = net("a", 0.1, 0.1)
    report["summary"]["success_count"] = 4
    assert ab.cell_summary([report])["success_episodes"] == 4


def test_a_cell_needs_networks():
    with pytest.raises(cp.ComparisonError):
        ab.cell_summary([])


# ------------------------------------------------------------------------- analysis

def test_more_data_helps_when_the_full_cell_is_clearly_higher():
    r = ab.analyze(make_cells(spec_of(vals(0.05, 0.002, 5), vals(0.15, 0.002, 10),
                                      vals(0.05, 0.002, 5), vals(0.15, 0.002, 10))))
    for method in ab.METHODS:
        for metric in ab.PRIMARY:
            p = r["primary"][method][metric]
            assert p["verdict"] == "more data helps"
            assert p["difference"] == pytest.approx(p["full_mean"] - p["small_mean"])
            assert p["ci95"][0] <= p["difference"] <= p["ci95"][1]
            assert (p["small_networks"], p["full_networks"]) == (5, 10)


def test_more_data_hurts_when_the_full_cell_is_clearly_lower():
    r = ab.analyze(make_cells(spec_of(vals(0.30, 0.002, 5), vals(0.10, 0.002, 10),
                                      vals(0.30, 0.002, 5), vals(0.10, 0.002, 10))))
    assert all(r["primary"][m][k]["verdict"] == "more data hurts" for m in ab.METHODS for k in ab.PRIMARY)


def test_identical_cells_show_no_clear_effect_and_zero_difference():
    r = ab.analyze(make_cells(spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL)))
    for method in ab.METHODS:
        for metric in ab.PRIMARY:
            p = r["primary"][method][metric]
            assert p["verdict"] == "no clear effect" and p["difference"] == pytest.approx(0.0)


def test_verdicts_are_per_method_and_per_metric():
    baseline_small = [(0.10 + 0.002 * i, 0.10 + 0.002 * i) for i in range(5)]
    baseline_full = [(0.30 + 0.002 * i, 0.10 + 0.002 * i) for i in range(10)]   # only the peak grows
    r = ab.analyze(make_cells(spec_of(baseline_small, baseline_full, SAME_SMALL, SAME_FULL)))
    assert r["primary"]["baseline"]["mean_max_coverage"]["verdict"] == "more data helps"
    final = r["primary"]["baseline"]["mean_final_coverage"]
    # the mean final coverage is 0.104 for five networks and 0.109 for ten, so the difference is 0.005
    assert final["difference"] == pytest.approx(0.005, abs=1e-9)
    assert final["verdict"] in ("more data helps", "no clear effect")
    assert r["primary"]["chunk"]["mean_max_coverage"]["verdict"] == "no clear effect"


def test_the_difference_is_always_full_minus_smallest():
    r = ab.analyze(make_cells(spec_of(vals(0.10, 0.001, 5), vals(0.20, 0.001, 10),
                                      vals(0.10, 0.001, 5), vals(0.20, 0.001, 10))))
    assert r["primary"]["chunk"]["mean_max_coverage"]["difference"] > 0


def test_middle_sizes_are_descriptive_cells_and_do_not_enter_the_verdicts():
    spec = spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL)
    spec["baseline"][50] = vals(0.9, 0.001, 5)
    spec["baseline"][100] = vals(0.9, 0.001, 5)
    r = ab.analyze(make_cells(spec))
    assert sorted(r["cells"]["baseline"]) == [20, 50, 100, 165]
    assert r["cells"]["baseline"][50]["mean_max_coverage"]["mean"] == pytest.approx(0.902)
    assert r["primary"]["baseline"]["mean_max_coverage"]["verdict"] == "no clear effect"


def test_the_planned_cell_sizes_carry_no_warning():
    assert ab.analyze(make_cells(spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL)))["warnings"] == []


def test_fewer_networks_than_planned_is_labelled_a_pilot():
    r = ab.analyze(make_cells(spec_of(SAME_SMALL[:3], SAME_FULL[:4], SAME_SMALL, SAME_FULL)))
    assert any("baseline, 20 episodes: 3 networks" in w and "pilot" in w for w in r["warnings"])
    assert any("baseline, 165 episodes: 4 networks" in w for w in r["warnings"])
    assert not any(w.startswith("chunk") for w in r["warnings"])


def test_a_missing_smallest_or_full_cell_is_refused():
    for missing in (20, 165):
        spec = spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL)
        del spec["chunk"][missing]
        with pytest.raises(cp.ComparisonError):
            ab.analyze(make_cells(spec))


def test_a_missing_method_is_refused():
    spec = spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL)
    del spec["chunk"]
    with pytest.raises(cp.ComparisonError):
        ab.analyze(make_cells(spec))


def test_a_single_network_in_a_primary_cell_is_refused():
    with pytest.raises(ValueError):
        ab.analyze(make_cells(spec_of(SAME_SMALL[:1], SAME_FULL, SAME_SMALL, SAME_FULL)))


def test_reports_on_different_seeds_are_refused():
    cells = make_cells(spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL))
    cells["chunk"][20][0] = net("chunk_cnn_n20_s0", 0.1, 0.1, seed_start=20000)
    with pytest.raises(cp.ComparisonError):
        ab.analyze(cells)


def test_duplicate_network_names_are_refused():
    cells = make_cells(spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL))
    cells["chunk"][20][1] = cells["chunk"][20][0]
    with pytest.raises(cp.ComparisonError):
        ab.analyze(cells)


def test_the_result_is_json_serializable_and_records_the_evaluation():
    r = ab.analyze(make_cells(spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL)))
    json.dumps(r)
    assert r["evaluation"] == {"episodes": 3, "seed_start": 10000, "seed_end": 10002}


def test_the_format_shows_every_cell_every_verdict_and_the_note():
    spec = spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL)
    spec["baseline"][50] = vals(0.1, 0.01, 5)
    text = ab.format_study(ab.analyze(make_cells(spec)))
    for expected in ("demonstration-count ablation:", "baseline networks:", "chunk networks:",
                     "training episodes", "above do-nothing", "165 episodes", "20 episodes",
                     "baseline, mean_max_coverage:", "chunk, mean_final_coverage:", "no clear effect",
                     "note: four uncorrected verdicts"):
        assert expected in text
    assert "WARNING" not in text


# ----------------------------------------------------------------------- command line

def write_cells(directory, spec, verified=True):
    cells = make_cells(spec)
    for by_size in cells.values():
        for reports in by_size.values():
            for r in reports:
                (directory / f"eval_{r['policy']}.json").write_text(json.dumps(r))
                if verified:
                    (directory / f"reproducibility_{r['policy']}.json").write_text(json.dumps({"reproduced": True}))


def run_main(directory, *extra):
    out = directory / "study.json"
    code = ab.main(["--report-dir", str(directory), "--sizes", "20", "165", "--out", str(out), *extra])
    return code, out


def test_main_writes_the_study_and_prints_it(tmp_path, capsys):
    write_cells(tmp_path, spec_of(vals(0.05, 0.002, 5), vals(0.15, 0.002, 10),
                                  SAME_SMALL, SAME_FULL))
    code, out = run_main(tmp_path)
    assert code == 0
    saved = json.loads(out.read_text())
    assert saved["unverified_networks"] == []
    assert saved["primary"]["baseline"]["mean_max_coverage"]["verdict"] == "more data helps"
    assert saved["primary"]["chunk"]["mean_max_coverage"]["verdict"] == "no clear effect"
    assert "demonstration-count ablation:" in capsys.readouterr().out


def test_main_refuses_a_network_without_a_passing_record(tmp_path, capsys):
    write_cells(tmp_path, spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL))
    (tmp_path / "reproducibility_bc_cnn_n20_s1.json").write_text(json.dumps({"reproduced": False}))
    code, out = run_main(tmp_path)
    assert code == 1 and not out.exists() and "bc_cnn_n20_s1" in capsys.readouterr().err


def test_allow_unverified_analyses_anyway_and_records_it(tmp_path):
    write_cells(tmp_path, spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL))
    (tmp_path / "reproducibility_chunk_cnn_s3.json").unlink()
    code, out = run_main(tmp_path, "--allow-unverified")
    saved = json.loads(out.read_text())
    assert code == 0 and saved["unverified_networks"] == ["chunk_cnn_s3"]
    assert any("chunk_cnn_s3" in w for w in saved["warnings"])


def test_main_requires_the_smallest_and_the_full_size(tmp_path, capsys):
    write_cells(tmp_path, spec_of(SAME_SMALL, SAME_FULL, SAME_SMALL, SAME_FULL))
    code = ab.main(["--report-dir", str(tmp_path), "--sizes", "50", "100", "--out", str(tmp_path / "o.json")])
    assert code == 1 and "--sizes must include" in capsys.readouterr().err


def test_main_reports_a_missing_file_as_an_error(tmp_path, capsys):
    code = ab.main(["--report-dir", str(tmp_path), "--sizes", "20", "165", "--out", str(tmp_path / "o.json")])
    assert code == 1 and "error:" in capsys.readouterr().err


# ------------------------------------------------------------ the real saved reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"
FULL_PATHS = ab.default_paths(REPORTS, sizes=(165,))
needs_seed_study = pytest.mark.skipif(
    not all(p.exists() for by_size in FULL_PATHS.values() for p in by_size[165]),
    reason="run the seed study first",
)


@needs_seed_study
def test_the_full_data_cells_reproduce_the_seed_study_group_means():
    base = ab.cell_summary([cp.load_report(p) for p in FULL_PATHS["baseline"][165]])
    chunk = ab.cell_summary([cp.load_report(p) for p in FULL_PATHS["chunk"][165]])
    assert (base["networks"], chunk["networks"]) == (10, 10)
    assert base["mean_max_coverage"]["mean"] == pytest.approx(0.0979, abs=5e-4)
    assert chunk["mean_max_coverage"]["mean"] == pytest.approx(0.1474, abs=5e-4)
    assert base["mean_final_coverage"]["mean"] == pytest.approx(0.0666, abs=5e-4)
    assert chunk["mean_final_coverage"]["mean"] == pytest.approx(0.0448, abs=5e-4)
    assert (base["above_donothing"]["above"], chunk["above_donothing"]["above"]) == (6, 2)
