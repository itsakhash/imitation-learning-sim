"""Tests for diffusion_pilot.py. Everything but the last two tests runs without torch."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import diffusion_pilot as dpil

NAN = float("nan")


def fake_mae(arm, lr):
    """Made-up errors with a different best rate for each arm."""
    best = {"D": 3e-4, "C": 1e-3, "B": 1e-4}[arm]
    return 10.0 + (0.0 if lr == best else 1.0 + lr)


def write_fake_report(report_dir, arm, lr, mae=None):
    path = dpil.report_path(report_dir, arm, lr)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"final": {"val_mae_px": fake_mae(arm, lr) if mae is None else mae}}))


class FakeTrainer:
    def __init__(self, report_dir):
        self.report_dir, self.calls = report_dir, []

    def __call__(self, arm, lr, name):
        self.calls.append((arm, lr, name))
        write_fake_report(self.report_dir, arm, lr)


# ---------------------------------------------------------------- pure


def test_plan_is_nine_trainings_arm_by_arm_in_the_listed_rate_order():
    p = dpil.plan()
    assert len(p) == 9 and p[0] == ("D", 1e-3) and p[1] == ("D", 3e-4) and p[3] == ("C", 1e-3)
    assert [a for a, _ in p] == list("DDDCCCBBB")
    assert dpil.plan(["B"]) == [("B", 1e-3), ("B", 3e-4), ("B", 1e-4)]


@pytest.mark.parametrize("arms", [["X"], ["D", "D"], ["P"]])
def test_plan_rejects_bad_arms(arms):
    with pytest.raises(ValueError):
        dpil.plan(arms)


def test_the_grid_and_seed_are_the_preregistered_ones():
    assert dpil.GRID == (1e-3, 3e-4, 1e-4) and dpil.PILOT_SEED == 99


def test_names():
    assert dpil.pilot_name("D", 3e-4) == "pilot_D_lr0.0003"
    assert dpil.pilot_name("C", 1e-3) == "pilot_C_lr0.001"
    assert dpil.report_path("r", "B", 1e-4) == Path("r") / "training_pilot_B_lr0.0001.json"


def test_choose_lr_takes_the_lowest_error():
    assert dpil.choose_lr({1e-3: 9.0, 3e-4: 8.0, 1e-4: 8.5}) == 3e-4
    assert dpil.choose_lr({1e-3: 9.0, 3e-4: 9.5, 1e-4: 7.0}) == 1e-4


def test_choose_lr_breaks_ties_towards_the_rate_listed_first():
    assert dpil.choose_lr({1e-3: 8.0, 3e-4: 8.0, 1e-4: 8.0}) == 1e-3
    assert dpil.choose_lr({1e-3: 9.0, 3e-4: 8.0, 1e-4: 8.0}) == 3e-4


def test_choose_lr_ignores_non_finite_errors_and_needs_one_finite():
    assert dpil.choose_lr({1e-3: NAN, 3e-4: float("inf"), 1e-4: 20.0}) == 1e-4
    with pytest.raises(ValueError):
        dpil.choose_lr({1e-3: NAN, 3e-4: NAN, 1e-4: NAN})
    with pytest.raises(ValueError):
        dpil.choose_lr({})


def test_collect_reads_final_errors_of_existing_reports(tmp_path):
    write_fake_report(tmp_path, "D", 1e-3, 12.5)
    write_fake_report(tmp_path, "C", 1e-4, 7.5)
    assert dpil.collect(tmp_path) == {"D": {1e-3: 12.5}, "C": {1e-4: 7.5}}
    assert dpil.collect(tmp_path / "nothing") == {}


def full_collected():
    return {a: {lr: fake_mae(a, lr) for lr in dpil.GRID} for a in dpil.ARMS}


def test_build_result_chooses_per_arm_and_flags_the_edge_of_the_grid():
    result = dpil.build_result(full_collected(), epochs=370)
    chosen = {a: result["arms"][a]["chosen_lr"] for a in dpil.ARMS}
    assert chosen == {"D": 3e-4, "C": 1e-3, "B": 1e-4}
    assert result["arms"]["D"]["at_edge_of_grid"] is False
    assert result["arms"]["C"]["at_edge_of_grid"] is True and result["arms"]["B"]["at_edge_of_grid"] is True
    assert set(result["arms"]["D"]["val_mae_px"]) == {"0.001", "0.0003", "0.0001"}
    assert result["pilot_seed"] == 99 and result["epochs"] == 370


def test_build_result_refuses_an_incomplete_pilot():
    partial = full_collected()
    del partial["B"][1e-4]
    with pytest.raises(ValueError):
        dpil.build_result(partial, epochs=370)


def test_the_result_is_written_once_and_read_back(tmp_path):
    path = tmp_path / "sub" / "pilot_result.json"
    dpil.write_result(path, dpil.build_result(full_collected(), 370))
    assert dpil.load_chosen_lrs(path) == {"D": 3e-4, "C": 1e-3, "B": 1e-4}
    with pytest.raises(FileExistsError):
        dpil.write_result(path, dpil.build_result(full_collected(), 370))


def test_load_chosen_lrs_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        dpil.load_chosen_lrs(tmp_path / "nope.json")
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"arms": {"D": {"chosen_lr": 1e-3}}}))
    with pytest.raises(ValueError):
        dpil.load_chosen_lrs(path)


def test_format_result_mentions_every_arm_and_the_choice():
    text = dpil.format_result(dpil.build_result(full_collected(), 370))
    for arm in "DCB":
        assert f"  {arm}:" in text
    assert "chosen 0.0003" in text and "edge of the grid" in text


# ---------------------------------------------------------------- run_pilot with a fake trainer


def quiet(*_):
    return None


def test_run_pilot_trains_all_nine_in_order_and_writes_the_result(tmp_path):
    trainer, result_path = FakeTrainer(tmp_path / "rep"), tmp_path / "result.json"
    result = dpil.run_pilot(dpil.ARMS, trainer, report_dir=tmp_path / "rep", result_path=result_path, epochs=370, log=quiet)
    assert [c[2] for c in trainer.calls][:4] == ["pilot_D_lr0.001", "pilot_D_lr0.0003", "pilot_D_lr0.0001", "pilot_C_lr0.001"]
    assert len(trainer.calls) == 9
    assert result["arms"]["D"]["chosen_lr"] == 3e-4
    assert dpil.load_chosen_lrs(result_path) == {"D": 3e-4, "C": 1e-3, "B": 1e-4}


def test_run_pilot_resumes_and_does_not_retrain_or_rewrite(tmp_path):
    rep, result_path = tmp_path / "rep", tmp_path / "result.json"
    for lr in dpil.GRID:
        write_fake_report(rep, "D", lr)
    trainer = FakeTrainer(rep)
    dpil.run_pilot(dpil.ARMS, trainer, report_dir=rep, result_path=result_path, epochs=370, log=quiet)
    assert len(trainer.calls) == 6 and all(c[0] != "D" for c in trainer.calls)
    before = result_path.read_text()
    again = FakeTrainer(rep)
    out = dpil.run_pilot(dpil.ARMS, again, report_dir=rep, result_path=result_path, epochs=370, log=quiet)
    assert again.calls == [] and result_path.read_text() == before and out["arms"]["C"]["chosen_lr"] == 1e-3


def test_a_partial_pilot_writes_no_result(tmp_path):
    trainer = FakeTrainer(tmp_path / "rep")
    out = dpil.run_pilot(["D"], trainer, report_dir=tmp_path / "rep", result_path=tmp_path / "result.json", epochs=370, log=quiet)
    assert out is None and len(trainer.calls) == 3 and not (tmp_path / "result.json").exists()


def test_a_failing_training_stops_the_pilot_without_a_result(tmp_path):
    def broken(arm, lr, name):
        raise FloatingPointError("boom")

    with pytest.raises(FloatingPointError):
        dpil.run_pilot(dpil.ARMS, broken, report_dir=tmp_path / "rep", result_path=tmp_path / "r.json", epochs=370, log=quiet)
    assert not (tmp_path / "r.json").exists()


def test_main_dry_run_lists_nine_and_runs_nothing(tmp_path, capsys):
    assert dpil.main(["--dry-run", "--report-dir", str(tmp_path / "rep")]) == 0
    out = capsys.readouterr().out
    assert out.count("to train") == 9 and "pilot_B_lr0.0001" in out and "nothing was run" in out
    write_fake_report(tmp_path / "rep", "D", 1e-3)
    dpil.main(["--dry-run", "--arms", "D", "--report-dir", str(tmp_path / "rep")])
    assert "exists" in capsys.readouterr().out


# ---------------------------------------------------------------- the real trainer (torch)


def test_the_real_pilot_runs_end_to_end_on_a_fake_dataset(tmp_path):
    pytest.importorskip("torch")
    from test_arm_train import make_pusht_arrays

    argv = ["--device", "cpu", "--report-dir", str(tmp_path / "rep"),
            "--checkpoint-dir", str(tmp_path / "ckpt"), "--result", str(tmp_path / "result.json")]
    assert dpil.main(argv, arrays_loader=make_pusht_arrays, epochs=1) == 0
    assert len(list((tmp_path / "rep").glob("training_pilot_*.json"))) == 9
    assert len(list((tmp_path / "ckpt").glob("pilot_*.pt"))) == 9
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["epochs"] == 1 and set(result["arms"]) == {"D", "C", "B"}
    assert all(result["arms"][a]["chosen_lr"] in dpil.GRID for a in "DCB")
    report = json.loads((tmp_path / "rep" / "training_pilot_D_lr0.0003.json").read_text())
    assert report["config"]["seed"] == 99 and report["config"]["lr"] == 3e-4
    assert report["config"]["ema_decay"] == 0.999 and report["config"]["epochs"] == 1


def test_the_real_pilot_resumes_after_an_interruption(tmp_path):
    pytest.importorskip("torch")
    from test_arm_train import make_pusht_arrays

    base = ["--device", "cpu", "--report-dir", str(tmp_path / "rep"),
            "--checkpoint-dir", str(tmp_path / "ckpt"), "--result", str(tmp_path / "result.json")]
    dpil.main(base + ["--arms", "B"], arrays_loader=make_pusht_arrays, epochs=1)
    assert not (tmp_path / "result.json").exists() and len(list((tmp_path / "rep").glob("*.json"))) == 3
    dpil.main(base, arrays_loader=make_pusht_arrays, epochs=1)  # B is not retrained: that would raise
    assert (tmp_path / "result.json").exists() and len(list((tmp_path / "rep").glob("*.json"))) == 9
