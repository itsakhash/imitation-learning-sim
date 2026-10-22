from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import compare_policies as cp
import evaluate as ev
import make_gif as mg


class FakeEnv:
    """A scripted environment: the picture brightens with the step, each step reports the next
    scripted coverage, and the episode is solved from step success_at on."""

    def __init__(self, trace, start=0.1, success_at=None, render_none=False, gray=False):
        self.trace, self.start, self.success_at = list(trace), start, success_at
        self.render_none, self.gray = render_none, gray
        self.t, self.closed, self.seeds = 0, False, []

    @property
    def unwrapped(self):
        return self

    def _get_coverage(self):
        return self.start

    def _picture(self):
        if self.gray:
            return np.zeros((8, 8), dtype=np.uint8)
        return np.full((8, 8, 3), min(self.t * 10, 250), dtype=np.uint8)

    def reset(self, seed=None):
        self.t = 0
        self.seeds.append(seed)
        return {"pixels": self._picture(), "agent_pos": np.zeros(2)}, {}

    def step(self, action):
        coverage = self.trace[self.t]
        self.t += 1
        success = self.success_at is not None and self.t >= self.success_at
        info = {"coverage": coverage, "is_success": success}
        truncated = self.t >= len(self.trace)
        return {"pixels": self._picture(), "agent_pos": np.zeros(2)}, 0.0, success, truncated, info

    def render(self):
        return None if self.render_none else self._picture()

    def close(self):
        self.closed = True


def hold(obs):
    return np.zeros(2, dtype=np.float32)


class RecordingPolicy:
    def __init__(self):
        self.reset_seeds = []

    def reset(self, seed):
        self.reset_seeds.append(seed)

    def __call__(self, obs):
        return np.zeros(2, dtype=np.float32)


KW = dict(trace=[0.1, 0.2, 0.3, 0.4, 0.5], start=0.1)


# ----------------------------------------------------------------------- replay

def test_replay_collects_a_picture_after_the_reset_and_after_every_step():
    env, policy = FakeEnv(**KW), RecordingPolicy()
    r = mg.replay_episode(env, policy, 7)
    assert r["seed"] == 7 and r["steps"] == 5 and len(r["frames"]) == 6
    assert r["coverage_trace"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert (r["start_coverage"], r["max_coverage"], r["final_coverage"]) == (0.1, 0.5, 0.5)
    assert r["success"] is False and r["truncated"] is True and r["terminated"] is False
    assert env.seeds == [7] and policy.reset_seeds == [7]
    assert all(f.dtype == np.uint8 and f.shape == (8, 8, 3) for f in r["frames"])


def test_replay_stops_when_the_episode_is_solved():
    r = mg.replay_episode(FakeEnv(trace=[0.1] * 10, start=0.0, success_at=3), hold, 1)
    assert r["steps"] == 3 and len(r["frames"]) == 4
    assert r["success"] is True and r["terminated"] is True
    assert r["solved"] == [False, False, True]


def test_replay_stops_at_the_step_limit():
    r = mg.replay_episode(FakeEnv(trace=[0.1] * 10, start=0.0), hold, 1, max_steps=4)
    assert r["steps"] == 4 and len(r["frames"]) == 5 and r["truncated"] is False


def test_pictures_come_from_render_or_from_the_observation():
    assert len(mg.replay_episode(FakeEnv(**KW), hold, 1, source="pixels")["frames"]) == 6
    with pytest.raises(RuntimeError, match="--source pixels"):
        mg.replay_episode(FakeEnv(**KW, render_none=True), hold, 1)
    assert len(mg.replay_episode(FakeEnv(**KW, render_none=True), hold, 1, source="pixels")["frames"]) == 6


def test_a_picture_that_is_not_rgb_is_refused():
    with pytest.raises(RuntimeError, match="RGB"):
        mg.replay_episode(FakeEnv(**KW, gray=True), hold, 1)


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"source": "gif"}])
def test_replay_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        mg.replay_episode(FakeEnv(**KW), hold, 1, **kwargs)


# ------------------------------------------------------- comparison with a report

def replay_and_episode(**kw):
    result = mg.replay_episode(FakeEnv(**kw), hold, 7)
    episode = ev.run_episode(FakeEnv(**kw), hold, 7)  # the evaluation harness itself
    return result, episode


def test_a_replay_matches_what_the_evaluation_harness_recorded():
    result, episode = replay_and_episode(**KW)
    assert mg.compare_with_report(result, episode) == []
    result, episode = replay_and_episode(trace=[0.1] * 10, start=0.0, success_at=3)
    assert mg.compare_with_report(result, episode) == []


def test_a_different_number_of_steps_is_reported():
    result, episode = replay_and_episode(**KW)
    problems = mg.compare_with_report(result, dict(episode, steps=episode["steps"] + 1))
    assert any(p.startswith("steps") for p in problems)


def test_a_different_success_is_reported():
    result, episode = replay_and_episode(**KW)
    problems = mg.compare_with_report(result, dict(episode, success=not episode["success"]))
    assert any(p.startswith("success") for p in problems)


def test_a_different_max_coverage_is_reported_but_rounding_is_not():
    result, episode = replay_and_episode(**KW)
    assert any("max_coverage" in p for p in
               mg.compare_with_report(result, dict(episode, max_coverage=episode["max_coverage"] + 1e-3)))
    assert mg.compare_with_report(result, dict(episode, max_coverage=episode["max_coverage"] + 5e-7)) == []


def test_a_different_coverage_trace_is_reported_at_its_first_step():
    result, episode = replay_and_episode(**KW)
    trace = list(episode["coverage_trace"])
    trace[2] += 1e-3
    problems = mg.compare_with_report(result, dict(episode, coverage_trace=trace))
    assert any("step 3" in p for p in problems)
    trace = list(episode["coverage_trace"])
    trace[2] += 5e-5  # the report rounds the trace to 4 decimals
    assert mg.compare_with_report(result, dict(episode, coverage_trace=trace)) == []


def test_a_missing_start_coverage_in_the_report_is_not_compared():
    result, episode = replay_and_episode(**KW)
    assert mg.compare_with_report(result, dict(episode, start_coverage=None)) == []


# ------------------------------------------------------------------------ picking

def ep(seed, success, max_coverage):
    return {"seed": seed, "success": success, "max_coverage": max_coverage}


def test_first_success_is_the_successful_episode_with_the_lowest_seed():
    report = {"episodes": [ep(10003, True, 0.97), ep(10001, True, 0.96), ep(10002, False, 0.3)]}
    assert mg.pick_episode(report, "first-success")["seed"] == 10001


def test_first_success_needs_a_success():
    with pytest.raises(ValueError, match="no successful"):
        mg.pick_episode({"episodes": [ep(1, False, 0.1)]}, "first-success")


def test_median_failure_is_the_lower_median_of_the_failures_by_max_coverage():
    even = {"episodes": [ep(5, False, 0.30), ep(3, False, 0.10), ep(4, False, 0.20), ep(6, False, 0.40)]}
    assert mg.pick_episode(even, "median-failure")["seed"] == 4
    odd = {"episodes": even["episodes"] + [ep(7, False, 0.50)]}
    assert mg.pick_episode(odd, "median-failure")["seed"] == 5


def test_median_failure_breaks_ties_by_seed_and_ignores_successes():
    tied = {"episodes": [ep(9, False, 0.2), ep(8, False, 0.2), ep(7, True, 0.99), ep(10, False, 0.2)]}
    assert mg.pick_episode(tied, "median-failure")["seed"] == 9  # failures ordered 8, 9, 10


def test_median_failure_needs_a_failure_and_unknown_rules_are_refused():
    with pytest.raises(ValueError, match="no failed"):
        mg.pick_episode({"episodes": [ep(1, True, 0.99)]}, "median-failure")
    with pytest.raises(ValueError):
        mg.pick_episode({"episodes": [ep(1, False, 0.1)]}, "best-failure")


# ----------------------------------------------------------------------- frames

@pytest.mark.parametrize(
    "n_frames, every, expected",
    [(10, 3, [0, 3, 6, 9]), (10, 4, [0, 4, 8, 9]), (1, 5, [0]), (5, 1, [0, 1, 2, 3, 4]), (289, 2, list(range(0, 289, 2)))],
)
def test_select_frames_keeps_the_first_and_the_last(n_frames, every, expected):
    assert mg.select_frames(n_frames, every) == expected


@pytest.mark.parametrize("n_frames, every", [(0, 2), (5, 0)])
def test_select_frames_rejects_bad_arguments(n_frames, every):
    with pytest.raises(ValueError):
        mg.select_frames(n_frames, every)


def array(value=100, size=8):
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_draw_frame_puts_a_white_header_above_the_scaled_picture():
    image = mg.draw_frame(array(), 3, 10, 0.5, False, width=32)
    assert image.size == (32, 32 + mg.HEADER_HEIGHT)
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((0, mg.HEADER_HEIGHT)) == (100, 100, 100)
    assert image.getpixel((31, 32 + mg.HEADER_HEIGHT - 1)) == (100, 100, 100)


def test_draw_frame_keeps_or_shrinks_the_size_as_asked():
    assert mg.draw_frame(array(), 0, 1, 0.0, False, width=8).size == (8, 8 + mg.HEADER_HEIGHT)
    assert mg.draw_frame(array(size=64), 0, 1, 0.0, False, width=32).size == (32, 32 + mg.HEADER_HEIGHT)


def test_the_header_carries_text_that_changes_with_the_solved_flag():
    plain = np.array(mg.draw_frame(array(size=64), 3, 10, 0.5, False, width=320))
    solved = np.array(mg.draw_frame(array(size=64), 3, 10, 0.5, True, width=320))
    assert plain[:mg.HEADER_HEIGHT].min() < 128
    assert not np.array_equal(plain[:mg.HEADER_HEIGHT], solved[:mg.HEADER_HEIGHT])
    assert np.array_equal(plain[mg.HEADER_HEIGHT:], solved[mg.HEADER_HEIGHT:])


def test_write_gif_writes_every_frame_and_holds_the_last_one(tmp_path):
    images = [mg.draw_frame(array(value=40 * i + 10), i, 3, 0.1 * i, False, width=48) for i in range(4)]
    path = tmp_path / "sub" / "x.gif"
    mg.write_gif(images, path, ms=60, hold_ms=1500)
    with Image.open(path) as gif:
        assert gif.n_frames == 4 and gif.size == images[0].size
        gif.seek(0)
        assert gif.info["duration"] == 60
        gif.seek(3)
        assert gif.info["duration"] == 1500


def test_write_gif_needs_images(tmp_path):
    with pytest.raises(ValueError):
        mg.write_gif([], tmp_path / "x.gif")


# ------------------------------------------------------------------ command line

def make_report(path, seeds=(10000, 10001, 10002), **kw):
    episodes = [ev.run_episode(FakeEnv(**kw), hold, s) for s in seeds]
    ev.write_report(path, ev.build_report("fake", episodes))
    return path


def factory_for(kw, made):
    def factory():
        env = FakeEnv(**kw)
        made.append(env)
        return env
    return factory


def builder_for(calls):
    def build(kind, checkpoint, env_factory, n_exec):
        calls.append((kind, checkpoint, n_exec))
        return hold
    return build


def run_main(argv, kw=KW, calls=None, made=None):
    return mg.main(argv, env_factory=factory_for(kw, [] if made is None else made),
                   policy_builder=builder_for([] if calls is None else calls))


def test_main_writes_the_gif_and_says_the_replay_matches_the_report(tmp_path, capsys):
    report, out, made, calls = make_report(tmp_path / "r.json", **KW), tmp_path / "x.gif", [], []
    code = run_main(["--policy", "chunk_cnn", "--checkpoint", "ck.pt", "--report", str(report),
                     "--pick", "median-failure", "--width", "32", "--out", str(out)], calls=calls, made=made)
    text = capsys.readouterr().out
    assert code == 0 and out.exists()
    assert "matches the evaluation report: yes" in text and "saved" in text and "success no" in text
    assert calls == [("chunk_cnn", Path("ck.pt"), None)] and made and all(e.closed for e in made)


def test_main_reports_a_mismatch_exits_with_one_and_still_writes_the_gif(tmp_path, capsys):
    other = dict(trace=[0.1, 0.2, 0.3, 0.4, 0.6], start=0.1)
    report, out = make_report(tmp_path / "r.json", **other), tmp_path / "x.gif"
    code = run_main(["--policy", "chunk_cnn", "--checkpoint", "ck.pt", "--report", str(report),
                     "--pick", "median-failure", "--width", "32", "--out", str(out)])
    assert code == 1 and out.exists()
    assert "does NOT match the evaluation report" in capsys.readouterr().out


def test_main_with_a_seed_and_no_report_has_nothing_to_compare(tmp_path, capsys):
    out = tmp_path / "x.gif"
    code = run_main(["--policy", "chunk_cnn", "--checkpoint", "ck.pt", "--seed", "5", "--width", "32",
                     "--out", str(out)])
    assert code == 0 and out.exists() and "not compared" in capsys.readouterr().out


def test_main_refuses_a_seed_that_is_not_in_the_report(tmp_path, capsys):
    report, out = make_report(tmp_path / "r.json", **KW), tmp_path / "x.gif"
    code = run_main(["--policy", "chunk_cnn", "--checkpoint", "ck.pt", "--report", str(report),
                     "--seed", "99", "--out", str(out)])
    assert code == 1 and not out.exists() and "99" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [[], ["--seed", "1", "--pick", "first-success"]])
def test_main_needs_exactly_one_of_seed_and_pick(tmp_path, capsys, extra):
    report = make_report(tmp_path / "r.json", **KW)
    code = run_main(["--policy", "chunk_cnn", "--report", str(report), *extra, "--out", str(tmp_path / "x.gif")])
    assert code == 1 and "exactly one" in capsys.readouterr().err


def test_main_needs_a_report_for_a_pick_and_a_success_for_first_success(tmp_path, capsys):
    assert run_main(["--policy", "chunk_cnn", "--pick", "median-failure"]) == 1
    assert "--pick needs --report" in capsys.readouterr().err
    report = make_report(tmp_path / "r.json", **KW)  # no episode is solved
    assert run_main(["--policy", "chunk_cnn", "--report", str(report), "--pick", "first-success",
                     "--out", str(tmp_path / "x.gif")]) == 1
    assert "no successful episode" in capsys.readouterr().err


def test_main_reports_a_missing_report_as_an_error(tmp_path, capsys):
    code = run_main(["--policy", "chunk_cnn", "--report", str(tmp_path / "nope.json"), "--seed", "1"])
    assert code == 1 and "error:" in capsys.readouterr().err


def test_the_default_output_name_follows_the_checkpoint_the_seed_and_the_executed_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    code = run_main(["--policy", "chunk_cnn", "--checkpoint", "checkpoints/chunk_cnn_s11.pt", "--seed", "10001",
                     "--n-exec", "4", "--width", "32"], calls=calls)
    assert code == 0 and calls == [("chunk_cnn", Path("checkpoints/chunk_cnn_s11.pt"), 4)]
    assert (tmp_path / "reports" / "gifs" / "chunk_cnn_s11_seed10001_x4.gif").exists()


def test_main_keeps_every_nth_frame(tmp_path):
    out = tmp_path / "x.gif"
    assert run_main(["--policy", "chunk_cnn", "--checkpoint", "ck.pt", "--seed", "1", "--every", "3",
                     "--width", "32", "--out", str(out)]) == 0
    with Image.open(out) as gif:
        assert gif.n_frames == 3  # frames 0, 3 and 5 of the 6 pictures of a 5-step episode


# ------------------------------------------------------------ the real saved reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"


@pytest.mark.skipif(not (REPORTS / "eval_chunk_cnn_s11.json").exists(), reason="run the replication first")
def test_the_real_chunked_success_is_the_episode_on_starting_position_10001():
    e = mg.pick_episode(cp.load_report(REPORTS / "eval_chunk_cnn_s11.json"), "first-success")
    assert e["seed"] == 10001 and e["steps"] == 288 and abs(e["start_coverage"]) < 1e-3
    assert e["max_coverage"] == pytest.approx(0.983, abs=5e-4)


@pytest.mark.skipif(not (REPORTS / "eval_bc_cnn_s11.json").exists(), reason="run the replication first")
def test_the_real_baseline_success_is_the_episode_on_starting_position_10000():
    e = mg.pick_episode(cp.load_report(REPORTS / "eval_bc_cnn_s11.json"), "first-success")
    assert e["seed"] == 10000 and e["steps"] == 39
    assert e["start_coverage"] == pytest.approx(0.634, abs=5e-4)
    assert e["max_coverage"] == pytest.approx(0.952, abs=5e-4)


@pytest.mark.skipif(not (REPORTS / "eval_chunk_cnn_s11.json").exists(), reason="run the replication first")
def test_the_real_median_failure_is_a_failed_episode_of_the_report():
    report = cp.load_report(REPORTS / "eval_chunk_cnn_s11.json")
    e = mg.pick_episode(report, "median-failure")
    assert not e["success"] and e in report["episodes"]
