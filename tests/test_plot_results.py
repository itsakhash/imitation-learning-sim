import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

import plot_results as pr  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


REFS = {
    "hold": {"max": 0.0585, "final": 0.0585},
    "random": {"max": 0.0696, "final": 0.0},
}


# ------------------------------------------------------------ synthetic saved reports


def cell(maxes, finals):
    return {
        "networks": len(maxes),
        "mean_max_coverage": {"values": list(maxes), "mean": float(np.mean(maxes))},
        "mean_final_coverage": {"values": list(finals), "mean": float(np.mean(finals))},
    }


def ablation():
    sizes = {"20": 5, "50": 5, "100": 5, "165": 10}

    def method(offset):
        return {
            size: cell(
                [0.05 + offset + 0.01 * i for i in range(n)],
                [0.03 + offset + 0.005 * i for i in range(n)],
            )
            for size, n in sizes.items()
        }

    return {"cells": {"baseline": method(0.0), "chunk": method(0.02)}}


def seed_study(offset=0.0):
    base_max = [0.08 + 0.003 * i for i in range(10)]
    base_final = [0.05 + 0.004 * i for i in range(10)]
    return {
        "primary": {
            "mean_max_coverage": {
                "baseline_values": base_max,
                "chunk_values": [m + 0.04 + offset for m in base_max],
                "difference": 0.04 + offset,
                "ci95": [0.01, 0.07],
            },
            "mean_final_coverage": {
                "baseline_values": base_final,
                "chunk_values": [f - 0.02 + offset for f in base_final],
                "difference": -0.02 + offset,
                "ci95": [-0.04, 0.0],
            },
        }
    }


def nexec(ns=(1, 2, 4, 8), networks=10):
    cells = {}
    for n in ns:
        maxes = [0.05 + 0.01 * n + 0.002 * i for i in range(networks)]
        finals = [0.04 + 0.001 * n + 0.002 * i for i in range(networks)]
        giveback = [m - f for m, f in zip(maxes, finals)]
        cells[str(n)] = {
            "networks": networks,
            "mean_max_coverage": {"values": maxes, "mean": float(np.mean(maxes))},
            "mean_final_coverage": {"values": finals, "mean": float(np.mean(finals))},
            "mean_giveback": {"values": giveback, "mean": float(np.mean(giveback))},
        }
    return {
        "cells": cells,
        "baseline": {
            "networks": networks,
            "mean_max_coverage": 0.09,
            "mean_final_coverage": 0.07,
            "mean_giveback": 0.02,
        },
    }


def summary_report(max_coverage, final_coverage):
    return {
        "summary": {
            "mean_max_coverage": max_coverage,
            "mean_final_coverage": final_coverage,
        }
    }


def write_reports(directory):
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "ablation_study.json": ablation(),
        "seed_study.json": seed_study(),
        "replication_seed_study.json": seed_study(0.01),
        "nexec_study.json": nexec(),
        "nexec_study_s10_s19.json": nexec((1, 4, 8)),
        "eval_hold.json": summary_report(0.0585, 0.0585),
        "eval_random.json": summary_report(0.0696, 0.0),
    }
    for name, data in files.items():
        (directory / name).write_text(json.dumps(data))


# ------------------------------------------------------------------------------- data


def test_group_ci_of_identical_values_is_degenerate():
    mean, low, high = pr.group_ci([0.1] * 5)
    assert (mean, low, high) == pytest.approx((0.1, 0.1, 0.1))


def test_group_ci_contains_the_mean_and_a_single_value_is_its_own_interval():
    mean, low, high = pr.group_ci([0.0, 0.1, 0.2, 0.3])
    assert mean == pytest.approx(0.15) and low <= mean <= high
    assert pr.group_ci([0.4]) == pytest.approx((0.4, 0.4, 0.4))
    with pytest.raises(ValueError):
        pr.group_ci([])


def test_ablation_points_sorts_the_sizes_as_integers():
    points = pr.ablation_points(ablation())
    assert sorted(points) == ["baseline", "chunk"]
    assert list(points["baseline"]) == [20, 50, 100, 165]
    assert [len(points["chunk"][s]["max"]) for s in (20, 50, 100, 165)] == [5, 5, 5, 10]
    assert points["baseline"][20]["final"][0] == pytest.approx(0.03)


def test_seed_block_points_reads_the_values_and_the_difference():
    block = pr.seed_block_points(seed_study())
    assert len(block["baseline"]["max"]) == 10 and len(block["chunk"]["final"]) == 10
    assert block["difference"]["max"] == pytest.approx((0.04, 0.01, 0.07))
    assert block["difference"]["final"] == pytest.approx((-0.02, -0.04, 0.0))


def test_nexec_points_turns_the_keys_into_integers():
    data = pr.nexec_points(nexec((4, 1, 8)))
    assert list(data["cells"]) == [1, 4, 8]
    assert len(data["cells"][4]["giveback"]) == 10
    assert data["baseline"] == pytest.approx(
        {"max": 0.09, "final": 0.07, "giveback": 0.02}
    )


def test_reference_levels_come_from_the_saved_reports(tmp_path):
    write_reports(tmp_path)
    assert pr.reference_levels(tmp_path) == REFS


def test_a_missing_report_is_a_plot_error_that_names_the_file(tmp_path):
    with pytest.raises(pr.PlotError, match="eval_hold.json"):
        pr.reference_levels(tmp_path)


def test_a_report_that_is_not_json_is_a_plot_error(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(pr.PlotError, match="not valid JSON"):
        pr.load_json(tmp_path / "bad.json")


# ----------------------------------------------------------------------------- figures


def test_the_demonstrations_figure_has_two_panels_with_the_planned_sizes():
    fig = pr.figure_demonstrations(ablation(), REFS)
    assert len(fig.axes) == 2
    assert "Peak" in fig.axes[0].get_title() and "Final" in fig.axes[1].get_title()
    for ax in fig.axes:
        assert list(ax.get_xticks()) == [20, 50, 100, 165]
        assert set(ax.get_legend_handles_labels()[1]) == {
            "baseline",
            "chunked",
            "do nothing",
            "random",
        }


def test_the_demonstrations_figure_draws_the_mean_of_every_cell():
    fig = pr.figure_demonstrations(ablation(), REFS)
    containers = {c.get_label(): c for c in fig.axes[0].containers}
    assert list(containers["baseline"].lines[0].get_xdata()) == [16, 46, 96, 161]
    assert list(containers["chunked"].lines[0].get_xdata()) == [24, 54, 104, 169]
    assert list(containers["baseline"].lines[0].get_ydata())[0] == pytest.approx(
        0.05 + 0.02, abs=1e-9
    )


def test_the_methods_figure_shows_both_blocks_and_their_differences():
    fig = pr.figure_methods(seed_study(), seed_study(0.01), REFS)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "baseline",
            "chunked",
            "baseline",
            "chunked",
        ]
        notes = [t.get_text() for t in ax.texts]
        assert sum("chunked minus baseline" in t for t in notes) == 2
        assert sorted(t for t in notes if t.startswith("training seeds")) == [
            "training seeds 0-9",
            "training seeds 10-19",
        ]
    assert "+0.0400 [+0.0100, +0.0700]" in fig.axes[0].texts[0].get_text()
    assert "-0.0200 [-0.0400, +0.0000]" in fig.axes[1].texts[0].get_text()


def test_the_methods_figure_keeps_the_references_and_the_legend():
    fig = pr.figure_methods(seed_study(), seed_study(0.01), REFS)
    assert set(fig.axes[1].get_legend_handles_labels()[1]) == {"do nothing", "random"}


def test_the_executed_targets_figure_has_three_panels_and_both_blocks():
    fig = pr.figure_executed_targets(nexec(), nexec((1, 4, 8)), REFS)
    assert len(fig.axes) == 3
    assert [ax.get_title().split(" (")[0] for ax in fig.axes] == [
        "Peak coverage",
        "Final coverage",
        "Coverage given back",
    ]
    labels = set(fig.axes[0].get_legend_handles_labels()[1])
    assert {
        "chunked, seeds 0-9",
        "baseline, seeds 0-9",
        "chunked, seeds 10-19",
        "baseline, seeds 10-19",
        "do nothing",
    } <= labels
    assert "do nothing" not in set(fig.axes[2].get_legend_handles_labels()[1])


def test_the_executed_targets_figure_places_the_conditions_in_order_and_shifts_the_two_blocks():
    fig = pr.figure_executed_targets(nexec(), nexec((1, 4, 8)), REFS)
    containers = {c.get_label(): c for c in fig.axes[0].containers}
    shift = pr.BLOCK_OFFSET
    assert list(containers["chunked, seeds 0-9"].lines[0].get_xdata()) == pytest.approx(
        [-shift, 1 - shift, 2 - shift, 3 - shift]
    )
    assert list(
        containers["chunked, seeds 10-19"].lines[0].get_xdata()
    ) == pytest.approx([shift, 2 + shift, 3 + shift])
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["1", "2", "4", "8"]


def test_an_unexpected_number_of_executed_targets_is_refused():
    with pytest.raises(pr.PlotError, match="16"):
        pr.figure_executed_targets(nexec((16, 8)), nexec((1, 8)), REFS)


def test_save_figure_writes_a_png_of_the_expected_size_and_closes_the_figure(tmp_path):
    fig = pr.figure_demonstrations(ablation(), REFS)
    number = fig.number
    path = tmp_path / "sub" / "demonstrations.png"
    pr.save_figure(fig, path)
    with Image.open(path) as image:
        assert image.format == "PNG" and image.size == (
            11 * pr.DPI,
            round(4.2 * pr.DPI),
        )
    assert number not in plt.get_fignums()


# ------------------------------------------------------------------------ command line


def test_main_writes_all_three_figures(tmp_path, capsys):
    write_reports(tmp_path / "reports")
    out = tmp_path / "plots"
    assert (
        pr.main(["--report-dir", str(tmp_path / "reports"), "--out-dir", str(out)]) == 0
    )
    text = capsys.readouterr().out
    for name in pr.FIGURES:
        assert (out / f"{name}.png").stat().st_size > 5000 and f"{name}.png" in text


def test_main_can_write_one_figure(tmp_path):
    write_reports(tmp_path / "reports")
    out = tmp_path / "plots"
    assert (
        pr.main(
            [
                "--report-dir",
                str(tmp_path / "reports"),
                "--out-dir",
                str(out),
                "--which",
                "methods",
            ]
        )
        == 0
    )
    assert [p.name for p in out.iterdir()] == ["methods.png"]


def test_main_names_a_missing_report_and_writes_nothing_after_it(tmp_path, capsys):
    write_reports(tmp_path / "reports")
    (tmp_path / "reports" / "nexec_study.json").unlink()
    out = tmp_path / "plots"
    assert (
        pr.main(["--report-dir", str(tmp_path / "reports"), "--out-dir", str(out)]) == 1
    )
    assert "nexec_study.json" in capsys.readouterr().err
    assert not (out / "executed_targets.png").exists()


def test_main_reports_a_report_with_a_missing_field(tmp_path, capsys):
    write_reports(tmp_path / "reports")
    (tmp_path / "reports" / "ablation_study.json").write_text(
        json.dumps({"cells": {"baseline": {}}})
    )
    assert (
        pr.main(
            [
                "--report-dir",
                str(tmp_path / "reports"),
                "--out-dir",
                str(tmp_path / "plots"),
                "--which",
                "demonstrations",
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err


# ------------------------------------------------------------ the real saved reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"
NEEDED = [
    "ablation_study.json",
    "seed_study.json",
    "replication_seed_study.json",
    "nexec_study.json",
    "nexec_study_s10_s19.json",
    "eval_hold.json",
    "eval_random.json",
]
needs_reports = pytest.mark.skipif(
    not all((REPORTS / n).exists() for n in NEEDED), reason="run the studies first"
)


@needs_reports
def test_the_real_reports_have_the_expected_shape_and_numbers():
    points = pr.ablation_points(pr.load_json(REPORTS / "ablation_study.json"))
    assert list(points["baseline"]) == [20, 50, 100, 165]
    assert [len(points["baseline"][s]["max"]) for s in (20, 50, 100, 165)] == [
        5,
        5,
        5,
        10,
    ]
    assert pr.group_ci(points["baseline"][165]["max"])[0] == pytest.approx(
        0.0979, abs=5e-4
    )
    block = pr.seed_block_points(pr.load_json(REPORTS / "seed_study.json"))
    assert len(block["baseline"]["max"]) == 10
    assert block["difference"]["final"][0] == pytest.approx(-0.0218, abs=5e-4)
    replication = pr.nexec_points(pr.load_json(REPORTS / "nexec_study_s10_s19.json"))
    assert list(replication["cells"]) == [1, 4, 8]
    assert replication["cells"][8]["giveback"] and replication["baseline"][
        "max"
    ] == pytest.approx(0.0987, abs=5e-4)


@needs_reports
def test_the_real_reports_produce_all_three_figures(tmp_path):
    assert pr.main(["--report-dir", str(REPORTS), "--out-dir", str(tmp_path)]) == 0
    for name in pr.FIGURES:
        assert (tmp_path / f"{name}.png").stat().st_size > 5000
