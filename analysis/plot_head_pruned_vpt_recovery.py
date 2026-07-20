"""Create slide-ready tables and plots for head-pruned VPT recovery.

Run:
  python analysis/plot_head_pruned_vpt_recovery.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = ("cifar100", "cub200", "fgvc_aircraft", "stanford_cars")
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "Aircraft",
    "stanford_cars": "Cars",
}
DEEP_TOKENS = (1, 2, 4, 8, 16)
METHOD_ORDER = (
    "Dense source",
    "Pruning-only",
    "LoRA",
    "VPT-Shallow-1",
    "VPT-Deep-1",
    "VPT-Deep-2",
    "VPT-Deep-4",
    "VPT-Deep-8",
    "VPT-Deep-16",
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deep-1-summary",
        default="logs/head_pruned_vpt_recovery_20260715_215638/summary.csv",
    )
    parser.add_argument(
        "--deep-2-summary",
        default="logs/head_pruned_vpt_deep2_20260715_150519/summary.csv",
    )
    parser.add_argument(
        "--deep-4-summary",
        default="logs/head_pruned_vpt_deep4_20260715_150716/summary.csv",
    )
    parser.add_argument(
        "--deep-8-summary",
        default="logs/head_pruned_vpt_deep8_20260715_161503/summary.csv",
    )
    parser.add_argument(
        "--deep-16-summary",
        default="logs/head_pruned_vpt_deep16_20260715_170500/summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/head_pruned_vpt_recovery/meeting",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def validate_summary(frame, tokens, path):
    required = {
        "dataset",
        "prompt_mode",
        "num_prompt_tokens",
        "baseline_acc",
        "pruning_only_acc",
        "reset_lora_best_acc",
        "best_acc",
        "final_acc",
        "trainable_params",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    deep = frame[
        (frame["prompt_mode"] == "deep")
        & (frame["num_prompt_tokens"] == tokens)
    ]
    if set(deep["dataset"]) != set(DATASETS) or len(deep) != len(DATASETS):
        raise ValueError(
            f"Expected one Deep-{tokens} row per dataset in {path}, found {len(deep)}."
        )
    if tokens == 1:
        shallow = frame[
            (frame["prompt_mode"] == "shallow")
            & (frame["num_prompt_tokens"] == 1)
        ]
        if set(shallow["dataset"]) != set(DATASETS) or len(shallow) != len(DATASETS):
            raise ValueError(
                f"Expected one Shallow-1 row per dataset in {path}, found {len(shallow)}."
            )


def load_summaries(args):
    paths = {
        1: Path(args.deep_1_summary),
        2: Path(args.deep_2_summary),
        4: Path(args.deep_4_summary),
        8: Path(args.deep_8_summary),
        16: Path(args.deep_16_summary),
    }
    summaries = {}
    for tokens, path in paths.items():
        frame = pd.read_csv(path)
        validate_summary(frame, tokens, path)
        summaries[tokens] = frame

    reference_columns = ("baseline_acc", "pruning_only_acc", "reset_lora_best_acc")
    reference = summaries[1][summaries[1]["prompt_mode"] == "deep"].set_index(
        "dataset"
    )
    for tokens, frame in summaries.items():
        deep = frame[frame["prompt_mode"] == "deep"].set_index("dataset")
        for column in reference_columns:
            delta = (deep.loc[list(DATASETS), column] - reference.loc[list(DATASETS), column]).abs()
            if (delta > 1e-8).any():
                raise ValueError(
                    f"Reference column {column} differs in Deep-{tokens} summary."
                )
    return summaries, paths


def build_combined_results(summaries, paths):
    reference = summaries[1][summaries[1]["prompt_mode"] == "deep"].set_index(
        "dataset"
    )
    rows = []
    for dataset in DATASETS:
        for method, column in (
            ("Dense source", "baseline_acc"),
            ("Pruning-only", "pruning_only_acc"),
            ("LoRA", "reset_lora_best_acc"),
        ):
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "prompt_mode": "",
                    "num_prompt_tokens": "",
                    "best_acc": float(reference.loc[dataset, column]),
                    "final_acc": "",
                    "trainable_params": "",
                    "source_summary": str(paths[1]),
                }
            )

        shallow = summaries[1][
            (summaries[1]["dataset"] == dataset)
            & (summaries[1]["prompt_mode"] == "shallow")
        ].iloc[0]
        rows.append(
            {
                "method": "VPT-Shallow-1",
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "prompt_mode": "shallow",
                "num_prompt_tokens": 1,
                "best_acc": float(shallow["best_acc"]),
                "final_acc": float(shallow["final_acc"]),
                "trainable_params": int(shallow["trainable_params"]),
                "source_summary": str(paths[1]),
            }
        )

        for tokens in DEEP_TOKENS:
            frame = summaries[tokens]
            deep = frame[
                (frame["dataset"] == dataset)
                & (frame["prompt_mode"] == "deep")
            ].iloc[0]
            rows.append(
                {
                    "method": f"VPT-Deep-{tokens}",
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "prompt_mode": "deep",
                    "num_prompt_tokens": tokens,
                    "best_acc": float(deep["best_acc"]),
                    "final_acc": float(deep["final_acc"]),
                    "trainable_params": int(deep["trainable_params"]),
                    "source_summary": str(paths[tokens]),
                }
            )
    return pd.DataFrame(rows)


def accuracy_matrix(combined):
    matrix = combined.pivot(index="method", columns="dataset", values="best_acc")
    matrix = matrix.loc[list(METHOD_ORDER), list(DATASETS)]
    matrix["average"] = matrix.mean(axis=1)
    return matrix


def save_figure(fig, output_dir, stem, dpi):
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def plot_accuracy_table(matrix, output_dir, dpi):
    columns = ("Method", "CIFAR100", "CUB200", "Aircraft", "Cars", "Average")
    cell_text = []
    for method, row in matrix.iterrows():
        cell_text.append(
            [method]
            + [f"{row[dataset]:.2f}" for dataset in DATASETS]
            + [f"{row['average']:.2f}"]
        )

    fig, ax = plt.subplots(figsize=(13.33, 6.35))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=(0.27, 0.145, 0.145, 0.145, 0.145, 0.15),
        bbox=(0.015, 0.12, 0.97, 0.76),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13.5)
    table.scale(1.0, 1.45)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#203864")
            cell.set_text_props(color="white", weight="bold")
            continue

        method = METHOD_ORDER[row_idx - 1]
        if method == "LoRA":
            cell.set_facecolor("#DCEAF7")
            cell.set_text_props(weight="bold", color="#17365D")
        elif method == "VPT-Deep-16":
            cell.set_facecolor("#E7F6EC")
            cell.set_text_props(weight="bold", color="#027A48")
        else:
            cell.set_facecolor("#FFFFFF")

        if col_idx == 0:
            cell.set_text_props(ha="left", weight=cell.get_text().get_weight())

    ax.set_title(
        "40% Whole-Head Pruning Recovery",
        fontsize=24,
        weight="bold",
        pad=16,
        color="#101828",
    )
    fig.text(
        0.5,
        0.055,
        "Best test accuracy (%) over 20 epochs · classifier reset for LoRA and VPT · seed 42",
        ha="center",
        fontsize=12.5,
        color="#475467",
    )
    return save_figure(fig, output_dir, "accuracy_table", dpi)


def plot_token_scaling(combined, output_dir, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5))
    axes = axes.flatten()
    deep = combined[combined["prompt_mode"] == "deep"].copy()

    for ax, dataset in zip(axes, DATASETS):
        dataset_frame = combined[combined["dataset"] == dataset].set_index("method")
        deep_frame = deep[deep["dataset"] == dataset].sort_values("num_prompt_tokens")
        dense = float(dataset_frame.loc["Dense source", "best_acc"])
        pruning_only = float(dataset_frame.loc["Pruning-only", "best_acc"])
        reset_lora = float(dataset_frame.loc["LoRA", "best_acc"])
        shallow = float(dataset_frame.loc["VPT-Shallow-1", "best_acc"])

        ax.plot(
            deep_frame["num_prompt_tokens"].astype(float),
            deep_frame["best_acc"],
            color="#2E75B6",
            marker="o",
            linewidth=2.6,
            markersize=6.5,
            label="VPT-Deep",
        )
        ax.scatter(
            [1],
            [shallow],
            color="#7030A0",
            marker="X",
            s=70,
            zorder=4,
            label="VPT-Shallow-1",
        )
        ax.axhline(
            reset_lora,
            color="#ED7D31",
            linestyle="--",
            linewidth=2.0,
            label="LoRA",
        )
        ax.axhline(
            dense,
            color="#7F7F7F",
            linestyle=":",
            linewidth=1.8,
            label="Dense source",
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks(DEEP_TOKENS, labels=[str(value) for value in DEEP_TOKENS])
        lower = min(shallow, float(deep_frame["best_acc"].min())) - 1.8
        upper = max(dense, reset_lora, float(deep_frame["best_acc"].max())) + 0.8
        ax.set_ylim(lower, upper)
        ax.set_title(DATASET_LABELS[dataset], fontsize=16, weight="bold", pad=7)
        ax.set_xlabel("Prompt tokens per layer", fontsize=11.5)
        ax.set_ylabel("Best accuracy (%)", fontsize=11.5)
        ax.grid(True, axis="y", alpha=0.28)
        ax.tick_params(labelsize=10.5)
        ax.text(
            0.97,
            0.06,
            f"Pruning-only: {pruning_only:.2f}%",
            transform=ax.transAxes,
            ha="right",
            fontsize=9.8,
            color="#667085",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#D0D5DD",
                "alpha": 0.9,
            },
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=12.5,
    )
    fig.suptitle(
        "Deep VPT Scaling after 40% Whole-Head Pruning",
        fontsize=22,
        weight="bold",
        y=0.99,
        color="#101828",
    )
    fig.text(
        0.5,
        0.025,
        "More prompt tokens consistently improve recovery, but LoRA remains stronger.",
        ha="center",
        fontsize=12.5,
        color="#475467",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.88))
    return save_figure(fig, output_dir, "token_scaling", dpi)


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, paths = load_summaries(args)
    combined = build_combined_results(summaries, paths)
    matrix = accuracy_matrix(combined)

    combined_path = output_dir / "combined_results.csv"
    table_data_path = output_dir / "accuracy_table.csv"
    combined.to_csv(combined_path, index=False)
    matrix.rename(columns={**DATASET_LABELS, "average": "Average"}).to_csv(table_data_path)

    outputs = [combined_path, table_data_path]
    outputs.extend(plot_accuracy_table(matrix, output_dir, args.dpi))
    outputs.extend(plot_token_scaling(combined, output_dir, args.dpi))

    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
