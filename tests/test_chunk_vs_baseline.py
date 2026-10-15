import json
from pathlib import Path

import pytest

import chunk_vs_baseline as cvb
import compare_policies as cp
import evaluate as ev


def report(name, rows, seed_start=10000):
    """rows: one (max_coverage, final_coverage) pair per seed, in seed order."""
    episodes = [
        {
            "seed": seed_start + i,
            "steps": 300,
            "start_coverage": 0.0,
            "max_coverage": m,
            "final_coverage": f,
            "success": False,
            "terminated": False,
            "truncated": True,
            "coverage_trace": [m],
        }
        for i, (m, f) in enumerate(rows)
    ]
    return ev.build_report(name, episodes)


def base_rows(n=5, max_cov=0.2, final_cov=0.1):
    return [(max_cov, final_cov)] * n


def shifted(rows, d_max, d_final):
    return [(m + d_max, f + d_final) for m, f in rows]


def triple(d_max, d_final, names=("a", "b", "c")):
    baselines = [report(f"bc{n}", base_rows()) for n in names]
    chunks = [report(f"ch{n}", shifted(base_rows(), d_max, d_final)) for n in names]
    return baselines, chunks


# ------------------------------------------------------------------- criterion


def test_improvement_on_both_metrics_in_all_three_pairs_meets_the_criterion():
    results = cvb.compare_pairs(*triple(0.05, 0.03))
    c = cvb.evaluate_criterion(results)
    assert c["met"] is True and c["pairs"] == 3 and c["pairs_improved"] == 3
    assert c["per_pair"] == [True, True, True] and c["verdict"].startswith("met")


def test_one_metric_worse_in_one_pair_means_not_met():
    baselines, chunks = triple(0.05, 0.03)
    chunks[1] = report("ch_bad", shifted(base_rows(), 0.05, -0.02))
    c = cvb.evaluate_criterion(cvb.compare_pairs(baselines, chunks))
    assert (
        c["met"] is False
        and c["pairs_improved"] == 2
        and c["per_pair"] == [True, False, True]
    )
    assert c["verdict"].startswith("not met")


def test_a_tie_is_not_an_improvement():
    c = cvb.evaluate_criterion(cvb.compare_pairs(*triple(0.0, 0.0)))
    assert c["met"] is False and c["pairs_improved"] == 0


def test_a_gain_on_only_one_metric_is_not_an_improvement():
    c = cvb.evaluate_criterion(cvb.compare_pairs(*triple(0.05, 0.0)))
    assert c["met"] is False and c["pairs_improved"] == 0


def test_fewer_than_three_pairs_cannot_meet_the_criterion():
    baselines, chunks = triple(0.05, 0.03)
    c = cvb.evaluate_criterion(cvb.compare_pairs(baselines[:2], chunks[:2]))
    assert c["met"] is False and c["verdict"].startswith("not evaluable")


def test_with_more_than_three_pairs_every_pair_must_improve():
    baselines, chunks = triple(0.05, 0.03, names=("a", "b", "c", "d"))
    assert cvb.evaluate_criterion(cvb.compare_pairs(baselines, chunks))["met"] is True
    chunks[3] = report("ch_bad", base_rows())
    assert cvb.evaluate_criterion(cvb.compare_pairs(baselines, chunks))["met"] is False


# ------------------------------------------------------------------- pairing


def test_pair_results_are_chunk_minus_baseline():
    baselines, chunks = triple(0.05, 0.03)
    r = cvb.compare_pairs(baselines, chunks)[0]
    assert (r["a"], r["b"]) == ("bca", "cha")
    assert r["metrics"]["max_coverage"]["mean_difference"] == pytest.approx(0.05)
    assert r["metrics"]["final_coverage"]["mean_difference"] == pytest.approx(0.03)
    assert r["metrics"]["max_coverage"]["higher"] == 5


def test_mismatched_numbers_of_reports_are_refused():
    baselines, chunks = triple(0.05, 0.03)
    with pytest.raises(cp.ComparisonError):
        cvb.compare_pairs(baselines, chunks[:2])
    with pytest.raises(cp.ComparisonError):
        cvb.compare_pairs([], [])


def test_reports_on_different_seeds_are_refused():
    baselines, chunks = triple(0.05, 0.03)
    chunks[0] = report("other", base_rows(), seed_start=20000)
    with pytest.raises(cp.ComparisonError):
        cvb.compare_pairs(baselines, chunks)


def test_group_means_list_each_network_in_order():
    g = cvb.group_means([report("x", [(0.2, 0.1)] * 3), report("y", [(0.4, 0.3)] * 3)])
    assert g["names"] == ["x", "y"]
    assert g["mean_max_coverage"] == pytest.approx([0.2, 0.4])
    assert g["mean_final_coverage"] == pytest.approx([0.1, 0.3])


# -------------------------------------------------------------------- output


def test_format_shows_the_pairs_the_spread_and_the_verdict():
    baselines, chunks = triple(0.05, 0.03)
    results = cvb.compare_pairs(baselines, chunks)
    text = cvb.format_comparison(
        results,
        cvb.evaluate_criterion(results),
        {"baseline": cvb.group_means(baselines), "chunk": cvb.group_means(chunks)},
    )
    for expected in (
        "pair 1:",
        "pair 3:",
        "chunk higher/equal/lower",
        "baseline networks",
        "chunk networks",
        "range",
        "criterion: met",
        "3 of 3 pairs",
        "note:",
    ):
        assert expected in text


def write_reports(tmp_path, d_max=0.05, d_final=0.03, other_seeds=False):
    baselines, chunks = triple(d_max, d_final)
    if other_seeds:
        chunks[0] = report("other", base_rows(), seed_start=20000)
    paths = {"baseline": [], "chunk": []}
    for kind, items in (("baseline", baselines), ("chunk", chunks)):
        for i, rep in enumerate(items):
            path = tmp_path / f"{kind}{i}.json"
            path.write_text(json.dumps(rep))
            paths[kind].append(str(path))
    return paths


def test_main_writes_the_comparison_and_prints_it(tmp_path, capsys):
    p = write_reports(tmp_path)
    out = tmp_path / "cmp.json"
    assert (
        cvb.main(
            ["--baseline", *p["baseline"], "--chunk", *p["chunk"], "--out", str(out)]
        )
        == 0
    )
    saved = json.loads(out.read_text())
    assert saved["criterion"]["met"] is True and len(saved["pairs"]) == 3
    assert set(saved["groups"]) == {"baseline", "chunk"}
    assert "criterion: met" in capsys.readouterr().out


def test_main_reports_incomparable_runs_as_an_error(tmp_path, capsys):
    p = write_reports(tmp_path, other_seeds=True)
    out = tmp_path / "cmp.json"
    assert (
        cvb.main(
            ["--baseline", *p["baseline"], "--chunk", *p["chunk"], "--out", str(out)]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err and not out.exists()


# ------------------------------------------------------------ real saved report

REAL = Path(__file__).resolve().parents[1] / "reports" / "eval_bc_cnn.json"


@pytest.mark.skipif(not REAL.exists(), reason="run evaluate_bc.py first")
def test_a_policy_compared_with_itself_never_meets_the_criterion():
    real = cp.load_report(REAL)
    results = cvb.compare_pairs([real] * 3, [real] * 3)
    c = cvb.evaluate_criterion(results)
    assert c["met"] is False and c["pairs_improved"] == 0
    for r in results:
        assert r["metrics"]["max_coverage"]["mean_difference"] == 0.0
