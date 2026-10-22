import json
from pathlib import Path

import pytest

import ablation_study as ab
import compare_policies as cp
import failure_categories as fk
import nexec_categories as nc


def net(method, seed, counts, frozen=0):
    return {"method": method, "seed": seed, "policy": f"{method}{seed}", "failed_in_report": sum(counts),
            "categorized": sum(counts), "mismatched": [], "counts": dict(zip(fk.CATEGORIES, counts)),
            "frozen": frozen, "episodes": []}


def block(method, seeds, counts, frozen=0):
    return [net(method, s, counts, frozen) for s in seeds]


BASE = (10, 5, 3, 2)     # shares 0.5, 0.25, 0.15, 0.1 of 20 failed episodes
CHUNK8 = (4, 6, 8, 2)    # shares 0.2, 0.3, 0.4, 0.1
SEEDS = range(10)


def full_networks(seeds=SEEDS):
    return block("baseline", seeds, BASE, frozen=18) + block("chunk", seeds, CHUNK8, frozen=1)


# ------------------------------------------------------------------------------ data

def test_load_full_returns_the_networks_and_names_problems(tmp_path):
    good = tmp_path / "a.json"
    good.write_text(json.dumps({"networks": [net("chunk", 0, BASE)]}))
    assert len(nc.load_full(good)) == 1
    with pytest.raises(nc.CategoryError, match="does not exist"):
        nc.load_full(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(nc.CategoryError, match="not valid JSON"):
        nc.load_full(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"networks": []}))
    with pytest.raises(nc.CategoryError, match="no networks"):
        nc.load_full(empty)


def test_networks_of_filters_by_method_block_and_categorized_episodes():
    networks = (block("baseline", [0, 1, 10], BASE) + block("chunk", [0, 1, 10], CHUNK8)
                + [net("chunk", 2, (0, 0, 0, 0))])
    assert [n["seed"] for n in nc.networks_of(networks, "baseline", "seeds 0-9")] == [0, 1]
    assert [n["seed"] for n in nc.networks_of(networks, "chunk", "seeds 10-19")] == [10]
    assert [n["seed"] for n in nc.networks_of(networks, "chunk", "seeds 0-9")] == [0, 1]


# ---------------------------------------------------------------------------- analysis

def test_paired_rows_are_the_per_network_change_in_each_share():
    default = block("chunk", SEEDS, CHUNK8, frozen=2)    # frozen share 0.1
    setting = block("chunk", SEEDS, BASE, frozen=10)     # frozen share 0.5
    rows = nc.paired_rows(default, setting)
    assert set(rows) == set(fk.KEYS)
    assert rows["never moved"]["mean"] == pytest.approx(0.3)
    assert rows["never moved"]["ci95"] == pytest.approx([0.3, 0.3])
    assert rows["improved then lost"]["mean"] == pytest.approx(0.15 - 0.4)
    assert rows["improved and kept"]["mean"] == pytest.approx(0.0)
    assert rows["frozen"]["mean"] == pytest.approx(0.4)


def test_paired_rows_pair_networks_by_training_seed_not_by_position():
    default = [net("chunk", s, (s + 1, 5, 3, 11 - s)) for s in SEEDS]   # 20 failed episodes each
    rows = nc.paired_rows(default, list(reversed(default)))
    assert all(rows[k]["mean"] == pytest.approx(0.0) for k in fk.KEYS)


def test_paired_rows_need_the_same_seeds_and_at_least_two_networks():
    with pytest.raises(nc.CategoryError, match="different training seeds"):
        nc.paired_rows(block("chunk", range(10), CHUNK8), block("chunk", range(1, 11), BASE))
    with pytest.raises(nc.CategoryError, match="two networks"):
        nc.paired_rows(block("chunk", [0], CHUNK8), block("chunk", [0], BASE))


def test_total_variation_distance_between_the_two_profiles():
    rows = fk.analyze_block(block("baseline", SEEDS, BASE), block("chunk", SEEDS, CHUNK8))["rows"]
    assert nc.tvd(rows) == pytest.approx(0.3)   # 0.5 * (0.3 + 0.05 + 0.25 + 0.0)


def test_analyze_reports_each_setting_against_8_and_against_the_baseline():
    by_setting = {4: block("chunk", SEEDS, (7, 5, 5, 3), frozen=6), 1: block("chunk", SEEDS, BASE, frozen=18)}
    result = nc.analyze(full_networks(), by_setting)
    assert list(result) == ["seeds 0-9"]
    entry = result["seeds 0-9"]
    assert set(entry) == {8, 4, 1, "baseline"}
    assert entry["baseline"]["shares"]["never moved"] == pytest.approx(0.5)
    assert entry["baseline"]["networks"] == 10 and entry["baseline"]["failed_episodes"] == 200
    assert entry[8]["tvd_to_baseline"] == pytest.approx(0.3) and entry[8]["paired_vs_8"] is None
    assert entry[1]["tvd_to_baseline"] == pytest.approx(0.0)
    assert entry[1]["paired_vs_8"]["never moved"]["mean"] == pytest.approx(0.3)
    assert entry[1]["versus_baseline"]["never moved"]["difference"] == pytest.approx(0.0)
    assert entry[4]["paired_vs_8"]["never moved"]["mean"] == pytest.approx(0.35 - 0.2)
    assert entry[4]["tvd_to_baseline"] == pytest.approx(0.15)   # 0.5 * (0.15 + 0.0 + 0.10 + 0.05)
    assert entry[4]["networks"] == 10 and entry[4]["failed_episodes"] == 200


def test_analyze_orders_the_blocks_by_training_seed():
    seeds = list(range(20))
    result = nc.analyze(full_networks(seeds), {1: block("chunk", seeds, BASE)})
    assert list(result) == ["seeds 0-9", "seeds 10-19"]


def test_analyze_refuses_a_block_with_too_few_networks():
    with pytest.raises(nc.CategoryError, match="too few"):
        nc.analyze(full_networks([0]), {1: block("chunk", [0], BASE)})


def result_for():
    by_setting = {4: block("chunk", SEEDS, (7, 5, 5, 3), frozen=6), 1: block("chunk", SEEDS, BASE, frozen=18)}
    return {"settings": {"executed": [4, 1], "default_executed": 8, "seeds": list(SEEDS)},
            "mismatched_episodes": 0,
            "networks": {"4": by_setting[4], "1": by_setting[1]},
            "blocks": nc.analyze(full_networks(), by_setting)}


def test_the_summary_has_the_three_tables_and_the_distances():
    text = nc.format_summary(result_for())
    for expected in ("failure categories by executed targets: 400 failed episodes replayed", "shares of failed episodes",
                     "chunked 8", "chunked 1", "total variation distance", "paired change",
                     "chunked executing n minus baseline", "frozen at the end (flag)", "0.300 at 8",
                     "0.000 at 1", "note: descriptive"):
        assert expected in text


# ------------------------------------------------------------------------ command line

class Env:
    def close(self):
        pass


def run_main(tmp_path, monkeypatch, extra=(), mismatch_in="", missing=""):
    full = tmp_path / "full.json"
    full.write_text(json.dumps({"networks": full_networks([0, 1, 2])}))
    requested, calls = [], []

    def load_report(path):
        requested.append(Path(path).name)
        if Path(path).name == missing:
            raise cp.ComparisonError(f"{path} does not exist")
        return {"policy": Path(path).name}

    def categorize(env, policy, report):
        counts = BASE if report["policy"].endswith("_x1.json") else (7, 5, 5, 3)
        mismatched = [{"seed": 0, "problems": ["steps"]}] if report["policy"] == mismatch_in else []
        return {"policy": report["policy"], "failed_in_report": 20, "categorized": 20, "mismatched": mismatched,
                "counts": dict(zip(fk.CATEGORIES, counts)), "frozen": 4, "episodes": []}

    def builder(kind, checkpoint, env_factory, n_exec):
        calls.append((kind, checkpoint, n_exec))
        return object()

    monkeypatch.setattr(nc.cp, "load_report", load_report)
    monkeypatch.setattr(nc.fk, "categorize_network", categorize)
    code = nc.main(["--seeds", "0", "1", "2", "--full", str(full), "--report-dir", str(tmp_path),
                    "--checkpoint-dir", str(tmp_path), "--out", str(tmp_path / "out.json"), *extra],
                   env_factory=Env, policy_builder=builder, log=lambda *_: None)
    return code, requested, calls


NAMES = ("chunk_cnn", "chunk_cnn_s1", "chunk_cnn_s2")


def test_main_replays_every_chunked_network_at_every_setting_and_writes_the_analysis(tmp_path, monkeypatch, capsys):
    code, requested, calls = run_main(tmp_path, monkeypatch)
    assert code == 0
    assert requested == [f"eval_{name}_x{n}.json" for n in (4, 1) for name in NAMES]
    assert calls == [("chunk_cnn", tmp_path / f"{name}.pt", n) for n in (4, 1) for name in NAMES]
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved["mismatched_episodes"] == 0 and list(saved["networks"]) == ["4", "1"]
    assert saved["settings"]["executed"] == [4, 1] and len(saved["networks"]["1"]) == 3
    entry = saved["blocks"]["seeds 0-9"]
    assert entry["1"]["paired_vs_8"]["never moved"]["mean"] == pytest.approx(0.3)
    assert entry["4"]["paired_vs_8"]["never moved"]["mean"] == pytest.approx(0.15)
    assert entry["8"]["paired_vs_8"] is None
    text = capsys.readouterr().out
    assert "total variation distance" in text and "saved" in text


def test_main_exits_with_one_but_writes_the_analysis_when_a_replay_does_not_match(tmp_path, monkeypatch):
    code, _, _ = run_main(tmp_path, monkeypatch, mismatch_in="eval_chunk_cnn_s1_x4.json")
    assert code == 1
    assert json.loads((tmp_path / "out.json").read_text())["mismatched_episodes"] == 1


def test_main_reports_a_missing_evaluation_report_and_writes_nothing(tmp_path, monkeypatch, capsys):
    code, _, _ = run_main(tmp_path, monkeypatch, missing="eval_chunk_cnn_s2_x1.json")
    assert code == 1 and not (tmp_path / "out.json").exists()
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("settings", [["8"], ["4", "4"], ["0"], ["17"]])
def test_main_rejects_bad_settings(tmp_path, monkeypatch, capsys, settings):
    code, _, _ = run_main(tmp_path, monkeypatch, extra=["--settings", *settings])
    assert code == 1 and "--settings" in capsys.readouterr().err


@pytest.mark.parametrize("seeds", [["1", "1"], ["-1"]])
def test_main_rejects_bad_seeds(tmp_path, monkeypatch, capsys, seeds):
    code, _, _ = run_main(tmp_path, monkeypatch, extra=["--seeds", *seeds])
    assert code == 1 and "--seeds" in capsys.readouterr().err


# ------------------------------------------------------------ the real saved reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"
needs_categories = pytest.mark.skipif(not (REPORTS / "failure_categories.json").exists(),
                                      reason="run failure_categories.py first")


@needs_categories
def test_the_real_evaluation_reports_for_the_variants_exist_under_the_expected_names():
    expected = [f"eval_{ab.network_name('chunk', ab.FULL_SIZE, s)}_x{n}.json" for s in range(20) for n in (4, 1)]
    assert [name for name in expected if not (REPORTS / name).exists()] == []


@needs_categories
def test_the_real_full_categorization_has_forty_networks_in_two_blocks():
    networks = nc.load_full(REPORTS / "failure_categories.json")
    assert len(networks) == 40 and {fk.block_label(n["seed"]) for n in networks} == {"seeds 0-9", "seeds 10-19"}
    assert sum(n["categorized"] for n in networks) == 1998
