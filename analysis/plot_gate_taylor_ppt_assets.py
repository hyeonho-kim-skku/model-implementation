"""Create PPT-ready figures for gate-Taylor pruning slides.

Outputs:
  - gate_taylor_schematic.png
  - gate_taylor_fc1_out_reduction_summary.csv
  - gate_taylor_fc1_out_reduction_summary_table.png

Run:
  python analysis/plot_gate_taylor_ppt_assets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


DATASETS = ("cifar100", "cub200", "fgvc_aircraft", "stanford_cars")
REDUCTIONS = ("sum_abs", "sum_square", "signed_damage")
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
    "overall": "Overall",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-trials-csv",
        default="figures/gate_taylor_sensitivity/gate_taylor_trials.csv",
        help="CSV containing fc1_out reduction sensitivity results.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/gate_taylor_sensitivity/ppt_assets",
        help="Directory for PPT-ready figures and tables.",
    )
    parser.add_argument("--dpi", type=int, default=240, help="Saved figure DPI.")
    return parser


def add_box(ax, xy, width, height, text, facecolor, edgecolor="#344054", fontsize=12):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color="#111827",
    )
    return box


def add_arrow(ax, start, end, color="#475467", text=None, y_text_offset=0.07):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.6,
        color=color,
    )
    ax.add_patch(arrow)
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + y_text_offset,
            text,
            ha="center",
            va="center",
            fontsize=9.5,
            color=color,
        )


def plot_gate_taylor_schematic(output_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    ax.text(
        0.15,
        4.05,
        "Element-wise Gate Taylor for ViT MLP Channel Pruning",
        fontsize=17,
        weight="bold",
        color="#111827",
        va="top",
    )
    ax.text(
        0.15,
        3.72,
        "Gate deletion: g = 1 -> 0,   Taylor damage ~= - gate.grad",
        fontsize=11,
        color="#475467",
        va="top",
    )

    y = 2.35
    h = 0.72
    add_box(ax, (0.35, y), 1.35, h, "Input", "#F9FAFB")
    add_box(ax, (2.05, y), 1.35, h, "fc1", "#E0F2FE")
    add_box(ax, (3.75, y), 1.65, h, "fc1_out\ngate", "#FEF3C7", fontsize=11)
    add_box(ax, (5.75, y), 1.35, h, "GELU", "#ECFDF3")
    add_box(ax, (7.45, y), 1.55, h, "fc2_in\ngate", "#FCE7F3", fontsize=11)
    add_box(ax, (9.35, y), 1.35, h, "fc2", "#E0F2FE")
    add_box(ax, (11.0, y), 0.85, h, "Out", "#F9FAFB")

    centers = [
        (1.7, y + h / 2),
        (2.05, y + h / 2),
        (3.4, y + h / 2),
        (3.75, y + h / 2),
        (5.4, y + h / 2),
        (5.75, y + h / 2),
        (7.1, y + h / 2),
        (7.45, y + h / 2),
        (9.0, y + h / 2),
        (9.35, y + h / 2),
        (10.7, y + h / 2),
        (11.0, y + h / 2),
    ]
    for start, end in zip(centers[0::2], centers[1::2]):
        add_arrow(ax, start, end)

    ax.text(4.575, 1.83, "pre-GELU score", ha="center", fontsize=10, color="#92400E")
    ax.text(8.225, 1.83, "post-GELU score", ha="center", fontsize=10, color="#9D174D")

    add_box(ax, (2.85, 0.45), 6.3, 0.82, "Per element: score = gate * gate.grad", "#F2F4F7", fontsize=12)
    ax.text(
        5.95,
        0.18,
        "Channel score aggregates over batch and token dimensions.",
        ha="center",
        fontsize=10,
        color="#475467",
    )

    output_path = output_dir / "gate_taylor_schematic.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_reduction_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        dataset_frame = frame[(frame["dataset"] == dataset) & (frame["ratio"] > 0)]
        row = {"dataset": dataset}
        for reduction in REDUCTIONS:
            reduction_frame = dataset_frame[dataset_frame["reduction"] == reduction]
            row[f"{reduction}_mean_drop"] = reduction_frame["acc_drop"].mean()
        means = {reduction: row[f"{reduction}_mean_drop"] for reduction in REDUCTIONS}
        row["best_reduction"] = min(means, key=means.get)
        rows.append(row)

    overall = {"dataset": "overall"}
    nonzero = frame[frame["ratio"] > 0]
    for reduction in REDUCTIONS:
        reduction_frame = nonzero[nonzero["reduction"] == reduction]
        overall[f"{reduction}_mean_drop"] = reduction_frame["acc_drop"].mean()
    overall_means = {reduction: overall[f"{reduction}_mean_drop"] for reduction in REDUCTIONS}
    overall["best_reduction"] = min(overall_means, key=overall_means.get)
    rows.append(overall)
    return pd.DataFrame(rows)


def plot_reduction_table(summary: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    display = summary.copy()
    display["Dataset"] = display["dataset"].map(DATASET_LABELS)
    display["sum_abs"] = display["sum_abs_mean_drop"].map(lambda value: f"{value:.3f}")
    display["sum_square"] = display["sum_square_mean_drop"].map(lambda value: f"{value:.3f}")
    display["signed_damage"] = display["signed_damage_mean_drop"].map(lambda value: f"{value:.3f}")
    display["Best"] = display["best_reduction"]
    display = display[["Dataset", "sum_abs", "sum_square", "signed_damage", "Best"]]

    fig, ax = plt.subplots(figsize=(9.6, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.45)

    best_columns = {"sum_abs": 1, "sum_square": 2, "signed_damage": 3}
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#F2F4F7")
            cell.set_text_props(weight="bold", color="#111827")
            continue

        dataset = display.iloc[row - 1]["Dataset"]
        best = display.iloc[row - 1]["Best"]
        if dataset == "Overall":
            cell.set_facecolor("#EEF4FF")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF")

        if col == best_columns.get(best):
            cell.set_facecolor("#D1FADF")
            cell.set_text_props(weight="bold", color="#027A48")

    ax.set_title(
        "fc1_out Gate Taylor Reduction Comparison",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    fig.text(
        0.5,
        0.02,
        "Values are mean accuracy drop (%p) over nonzero pruning ratios. Lower is better.",
        ha="center",
        fontsize=8.5,
        color="#475467",
    )

    output_path = output_dir / "gate_taylor_fc1_out_reduction_summary_table.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schematic_path = plot_gate_taylor_schematic(output_dir, args.dpi)
    print(f"[Plot] saved {schematic_path}")

    frame = pd.read_csv(args.gate_trials_csv)
    summary = build_reduction_summary(frame)
    summary_csv = output_dir / "gate_taylor_fc1_out_reduction_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[Build] wrote {summary_csv}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    table_path = plot_reduction_table(summary, output_dir, args.dpi)
    print(f"[Plot] saved {table_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
