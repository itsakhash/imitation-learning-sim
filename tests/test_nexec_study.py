import json
from pathlib import Path

import pytest

import ablation_study as ab
import compare_policies as cp
import evaluate as ev
import nexec_study as nx


def net(name, max_cov, final, start=0.05, n=3, seed_start=10000):
    episodes = [
        {"seed": seed_start + i, "steps": 300, "start_coverage": start, "max_coverage": max_cov,
         "final_coverage": final, "success": False, "terminated": False, "truncated": True,
         "coverage_trace": [max_cov]}
        for i in range(n)
    ]
    return ev.build_report(name, episodes)


def rows(max_start, final_start, step=0.002, n=10):
    return [(max_start + step * i, final_start + step * i) for i in range(n)]


def make_cell(n_exec, cell_rows, seed_start=10000):
    return [net(nx.network_name(i, n_exec), m, f, seed_start=seed_start) for i, (m, f) in enumerate(cell_rows)]


def make_baseline(cell_rows, seed_start=10000):
    return [net(ab.network_name("baseline", ab.FULL_SIZE, i), m, f, seed_start=seed_start)
            for i, (m, f) in enumerate(cell_rows)]


REFERENCE_ROWS = rows(0.15, 0.04)    # giveback 0.11 in every network
BASELINE_ROWS = rows(0.10, 0.06)


# ----------------------------------------------------------------- names and paths

def test_network_names():
    assert nx.network_name(0, 8) == "chunk_cnn"
    assert nx.network_name(3, 8) == "chunk_cnn_s3"
    assert nx.network_name(0, 4) == "chunk_cnn_x4"
    assert nx.network_name(9, 1) == "chunk_cnn_s9_x1"


def test_default_paths_have_ten_networks_per_condition_and_ten_baselines():
    paths = nx.default_paths(Path("r"))
    assert sorted(paths["chunk"]) == [1, 2, 4, 8]
    assert all(len(p) == 10 for p in paths["chunk"].values()) and len(paths["baseline"]) == 10
    assert paths["chunk"][8][0].name == "eval_chunk_cnn.json"
    assert paths["chunk"][1][3].name == "eval_chunk_cnn_s3_x1.json"
    assert paths["chunk"][4][0].name == "eval_chunk_cnn_x4.json"
    assert paths["baseline"][0].name == "eval_bc_cnn.json" and paths["baseline"][9].name == "eval_bc_cnn_s9.json"


def test_default_paths_can_be_restricted():
    paths = nx.default_paths(Path("r"), n_execs=(2, 8), networks=3)
    assert sorted(paths["chunk"]) == [2, 8]
    assert all(len(p) == 3 for p in paths["chunk"].values()) and len(paths["baseline"]) == 3


# ------------------------------------------------------------------------ verdict

@pytest.mark.parametrize(
    "giveback, final, expected",
    [
        ((-0.06, -0.04), (0.02, 0.05), "fewer executed targets help"),
        ((-0.06, -0.04), (-0.01, 0.05), "mixed or no clear effect"),
        ((-0.06, 0.01), (0.02, 0.05), "mixed or no clear effect"),
        ((0.01, 0.05), (0.02, 0.05), "fewer executed targets hurt"),      # giveback entirely above zero
        ((-0.06, -0.04), (-0.05, -0.01), "fewer executed targets hurt"),  # final entirely below zero
        ((0.0, 0.05), (0.02, 0.05), "mixed or no clear effect"),           # a lower bound of 0 is not above 0
        ((-0.06, -0.04), (0.0, 0.05), "mixed or no clear effect"),         # a lower bound of 0 is not above 0
        ((-0.06, -0.04), (-0.05, 0.0), "mixed or no clear effect"),        # an upper bound of 0 is not below 0
        ((0.01, 0.05), (-0.05, -0.01), "fewer executed targets hurt"),
    ],
)
def test_verdict_follows_the_pre_registered_rule(giveback, final, expected):
    assert nx.verdict(giveback, final) == expected


# ------------------------------------------------------------- paired differences

def test_paired_difference_known_values():
    d = nx.paired_difference([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
    assert d["values"] == pytest.approx([0.1, 0.1, 0.1])
    assert d["difference"] == pytest.approx(0.1)
    assert d["ci95"] == pytest.approx([0.1, 0.1])


def test_paired_difference_interval_contains_the_mean():
    d = nx.paired_difference([0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4])
    assert d["difference"] == pytest.approx(0.25)
    assert d["ci95"][0] <= 0.25 <= d["ci95"][1]


# --------------------------------------------------------------------- cell summary

def test_cell_summary_known_values():
    c = nx.cell_summary([net("a", 0.30, 0.10), net("b", 0.20, 0.04), net("c", 0.10, 0.07)])
    assert c["networks"] == 3 and c["names"] == ["a", "b", "c"]
    assert c["mean_max_coverage"]["mean"] == pytest.approx(0.20)
    assert c["mean_final_coverage"]["mean"] == pytest.approx(0.07)
    assert c["mean_giveback"]["values"] == pytest.approx([0.20, 0.16, 0.03])
    assert c["mean_giveback"]["mean"] == pytest.approx(0.13)
    # the start coverage is 0.05, so two of the three final coverages (0.10 and 0.07) are above it
    assert c["above_donothing"]["above"] == 2


def test_a_condition_needs_networks():
    with pytest.raises(cp.ComparisonError):
        nx.cell_summary([])


# ------------------------------------------------------------------------- analysis

def test_executing_fewer_targets_helps_when_giveback_falls_and_final_coverage_rises():
    cells = {1: make_cell(1, rows(0.14, 0.08)), 8: make_cell(8, REFERENCE_ROWS)}
    r = nx.analyze(cells, make_baseline(BASELINE_ROWS))
    p = r["primary"][1]
    assert p["verdict"] == "fewer executed targets help"
    assert p["mean_giveback"]["difference"] == pytest.approx(-0.05)
    assert p["mean_final_coverage"]["difference"] == pytest.approx(0.04)
    assert p["mean_max_coverage"]["difference"] == pytest.approx(-0.01)
    for metric in nx.PAIRED_METRICS:
        low, high = p[metric]["ci95"]
        assert low - 1e-12 <= p[metric]["difference"] <= high + 1e-12
    assert sorted(r["primary"]) == [1]


def test_executing_fewer_targets_hurts_when_final_coverage_and_giveback_get_worse():
    cells = {2: make_cell(2, rows(0.14, 0.01)), 8: make_cell(8, REFERENCE_ROWS)}
    p = nx.analyze(cells, make_baseline(BASELINE_ROWS))["primary"][2]
    assert p["verdict"] == "fewer executed targets hurt"
    assert p["mean_giveback"]["difference"] == pytest.approx(0.02)
    assert p["mean_final_coverage"]["difference"] == pytest.approx(-0.03)


def test_identical_conditions_show_no_clear_effect_and_zero_differences():
    cells = {4: make_cell(4, REFERENCE_ROWS), 8: make_cell(8, REFERENCE_ROWS)}
    p = nx.analyze(cells, make_baseline(BASELINE_ROWS))["primary"][4]
    assert p["verdict"] == "mixed or no clear effect"
    assert all(p[m]["difference"] == pytest.approx(0.0) for m in nx.PAIRED_METRICS)


def test_every_condition_gets_its_own_verdict():
    cells = {
        1: make_cell(1, rows(0.14, 0.08)),
        2: make_cell(2, REFERENCE_ROWS),
        4: make_cell(4, rows(0.14, 0.01)),
        8: make_cell(8, REFERENCE_ROWS),
    }
    r = nx.analyze(cells, make_baseline(BASELINE_ROWS))
    assert sorted(r["primary"]) == [1, 2, 4] and sorted(r["cells"]) == [1, 2, 4, 8]
    assert [r["primary"][k]["verdict"] for k in (1, 2, 4)] == [
        "fewer executed targets help", "mixed or no clear effect", "fewer executed targets hurt"]


def test_descriptive_numbers_and_the_baseline_comparison():
    cells = {1: make_cell(1, rows(0.14, 0.08)), 8: make_cell(8, REFERENCE_ROWS)}
    r = nx.analyze(cells, make_baseline(BASELINE_ROWS))
    assert r["cells"][8]["mean_giveback"]["mean"] == pytest.approx(0.11)
    assert r["cells"][1]["mean_giveback"]["mean"] == pytest.approx(0.06)
    assert r["baseline"]["networks"] == 10
    assert r["baseline"]["mean_max_coverage"] == pytest.approx(0.109)
    assert r["baseline"]["mean_final_coverage"] == pytest.approx(0.069)
    assert r["baseline"]["mean_giveback"] == pytest.approx(0.04)
    # chunked (8 targets) minus baseline: 0.159 - 0.109 on the peak, 0.049 - 0.069 on the end
    assert r["versus_baseline"][8]["mean_max_coverage"]["difference"] == pytest.approx(0.05)
    assert r["versus_baseline"][8]["mean_final_coverage"]["difference"] == pytest.approx(-0.02)


def test_the_planned_number_of_networks_carries_no_warning():
    cells = {1: make_cell(1, rows(0.14, 0.08)), 8: make_cell(8, REFERENCE_ROWS)}
    r = nx.analyze(cells, make_baseline(BASELINE_ROWS))
    assert r["warnings"] == [] and r["networks_per_condition"] == 10


def test_fewer_networks_than_planned_is_labelled_a_pilot():
    cells = {1: make_cell(1, rows(0.14, 0.08, n=5)), 8: make_cell(8, rows(0.15, 0.04, n=5))}
    r = nx.analyze(cells, make_baseline(rows(0.10, 0.06, n=5)))
    assert any("pilot" in w for w in r["warnings"]) and any("baseline" in w for w in r["warnings"])


def test_the_reference_conditon_and_a_second_one_are_required():
    base = make_baseline(BASELINE_ROWS)
    with pytest.raises(cp.ComparisonError):
        nx.analyze({1: make_cell(1, REFERENCE_ROWS), 2: make_cell(2, REFERENCE_ROWS)}, base)
    with pytest.raises(cp.ComparisonError):
        nx.analyze({8: make_cell(8, REFERENCE_ROWS)}, base)


def test_conditions_with_different_numbers_of_networks_are_refused():
    cells = {1: make_cell(1, rows(0.14, 0.08, n=9)), 8: make_cell(8, REFERENCE_ROWS)}
    with pytest.raises(cp.ComparisonError):
        nx.analyze(cells, make_baseline(BASELINE_ROWS))


def test_no_baseline_networks_is_refused():
    cells = {1: make_cell(1, rows(0.14, 0.08)), 8: make_cell(8, REFERENCE_ROWS)}
    with pytest.raises(cp.ComparisonError):
        nx.analyze(cells, [])


def test_networks_in_the_wrong_order_are_refused():
    first = make_cell(1, rows(0.14, 0.08))
    first[0], first[1] = first[1], first[0]
    with pytest.raises(cp.ComparisonError):
        nx.analyze({1: first, 8: make_cell(8, REFERENCE_ROWS)}, make_baseline(BASELINE_ROWS))


def test_reports_on_different_seeds_are_refused():
    cells = {1: make_cell(1, rows(0.14, 0.08), seed_start=20000), 8: make_cell(8, REFERENCE_ROWS)}
    with pytest.raises(cp.ComparisonError):
        nx.analyze(cells, make_baseline(BASELINE_ROWS))


def test_duplicate_network_names_are_refused():
    first = make_cell(1, rows(0.14, 0.08))
    first[1] = first[0]
    with pytest.raises(cp.ComparisonError):
        nx.analyze({1: first, 8: make_cell(8, REFERENCE_ROWS)}, make_baseline(BASELINE_ROWS))


def test_check_pairing_accepts_the_planned_names():
    nx.check_pairing({1: make_cell(1, rows(0.14, 0.08)), 8: make_cell(8, REFERENCE_ROWS)})


def test_the_result_is_json_serializable_and_records_the_evaluation():
    cells = {1: make_cell(1, rows(0.14, 0.08)), 8: make_cell(8, REFERENCE_ROWS)}
    r = nx.analyze(cells, make_baseline(BASELINE_ROWS))
    json.dumps(r)
    assert r["evaluation"] == {"episodes": 3, "seed_start": 10000, "seed_end": 10002}


def test_the_format_shows_every_condition_every_verdict_and_the_note():
    cells = {1: make_cell(1, rows(0.14, 0.08)), 2: make_cell(2, REFERENCE_ROWS), 8: make_cell(8, REFERENCE_ROWS)}
    text = nx.format_study(nx.analyze(cells, make_baseline(BASELINE_ROWS)))
    for expected in ("executed-targets study:", "10 chunked networks", "executed", "given back",
                     "above do-nothing", "baseline networks (10):", "1 executed targets minus 8:",
                     "2 executed targets minus 8:", "-> fewer executed targets help",
                     "-> mixed or no clear effect", "note: three uncorrected verdicts"):
        assert expected in text
    assert "WARNING" not in text


# ----------------------------------------------------------------------- command line

def write_all(directory, cells, baseline):
    for report in [r for reports in cells.values() for r in reports] + baseline:
        (directory / f"eval_{report['policy']}.json").write_text(json.dumps(report))
        (directory / f"reproducibility_{report['policy']}.json").write_text(json.dumps({"reproduced": True}))


def run_main(directory, *extra):
    out = directory / "study.json"
    code = nx.main(["--report-dir", str(directory), "--n-execs", "1", "8", "--out", str(out), *extra])
    return code, out


def good_cells():
    return {1: make_cell(1, rows(0.14, 0.08)), 8: make_cell(8, REFERENCE_ROWS)}


def test_main_writes_the_study_and_prints_it(tmp_path, capsys):
    write_all(tmp_path, good_cells(), make_baseline(BASELINE_ROWS))
    code, out = run_main(tmp_path)
    assert code == 0
    saved = json.loads(out.read_text())
    assert saved["unverified_networks"] == []
    assert saved["primary"]["1"]["verdict"] == "fewer executed targets help"
    assert "executed-targets study:" in capsys.readouterr().out


def test_main_refuses_a_network_without_a_passing_record(tmp_path, capsys):
    write_all(tmp_path, good_cells(), make_baseline(BASELINE_ROWS))
    (tmp_path / "reproducibility_chunk_cnn_s3_x1.json").write_text(json.dumps({"reproduced": False}))
    code, out = run_main(tmp_path)
    assert code == 1 and not out.exists() and "chunk_cnn_s3_x1" in capsys.readouterr().err


def test_allow_unverified_analyses_anyway_and_records_it(tmp_path):
    write_all(tmp_path, good_cells(), make_baseline(BASELINE_ROWS))
    (tmp_path / "reproducibility_chunk_cnn_s2.json").unlink()
    code, out = run_main(tmp_path, "--allow-unverified")
    saved = json.loads(out.read_text())
    assert code == 0 and saved["unverified_networks"] == ["chunk_cnn_s2"]
    assert any("chunk_cnn_s2" in w for w in saved["warnings"])


def test_main_requires_the_reference_condition_and_another_one(tmp_path, capsys):
    write_all(tmp_path, good_cells(), make_baseline(BASELINE_ROWS))
    out = str(tmp_path / "o.json")
    assert nx.main(["--report-dir", str(tmp_path), "--n-execs", "1", "2", "--out", out]) == 1
    assert "--n-execs must include" in capsys.readouterr().err
    assert nx.main(["--report-dir", str(tmp_path), "--n-execs", "8", "--out", out]) == 1
    assert nx.main(["--report-dir", str(tmp_path), "--n-execs", "1", "1", "8", "--out", out]) == 1


def test_main_reports_a_missing_file_as_an_error(tmp_path, capsys):
    assert run_main(tmp_path)[0] == 1
    assert "error:" in capsys.readouterr().err


# ------------------------------------------------------------ the real saved reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REAL = nx.default_paths(REPORTS, n_execs=(nx.REFERENCE,))
needs_seed_study = pytest.mark.skipif(
    not all(p.exists() for p in REAL["chunk"][nx.REFERENCE] + REAL["baseline"]),
    reason="run the seed study first",
)


@needs_seed_study
def test_the_real_eight_target_evaluations_follow_the_naming_and_reproduce_the_seed_study():
    reports = [cp.load_report(p) for p in REAL["chunk"][nx.REFERENCE]]
    nx.check_pairing({nx.REFERENCE: reports})
    c = nx.cell_summary(reports)
    assert c["networks"] == 10
    assert c["mean_max_coverage"]["mean"] == pytest.approx(0.1474, abs=5e-4)
    assert c["mean_final_coverage"]["mean"] == pytest.approx(0.0448, abs=5e-4)
    assert c["mean_giveback"]["mean"] == pytest.approx(0.1026, abs=1e-3)
    assert c["above_donothing"]["above"] == 2
