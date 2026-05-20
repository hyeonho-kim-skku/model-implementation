"""Plot Taylor layer-sensitivity results.

Inputs are results.jsonl files from sensitivity_taylor.py. For each dataset,
this script writes:
  - <dataset>_heatmap.png: layer x ratio accuracy-drop heatmap
  - <dataset>_all_layer_curves.png: 12 subplot accuracy-drop curves

Run:
  python analysis/plot_taylor_sensitivity.py

Use --results-jsonl and --output-dir to override the defaults.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS = [
    "pruned/vit_base_cifar100_lora50_taylor_sensitivity/results.jsonl",
    "pruned/vit_base_flowers102_lora50_taylor_sensitivity/results.jsonl",
    "pruned/vit_base_cub200_lora50_taylor_sensitivity/results.jsonl",
]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-jsonl",
        nargs="+",
        default=DEFAULT_RESULTS,
        help="One or more sensitivity results.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/taylor_sensitivity",
        help="Directory to save generated figures.",
    )
    return parser


def load_results(path):
    """Read one JSONL file and split metadata from trial rows."""

    metadata = None
    trials = []
    with open(path, "r") as file:
        for line in file:
            row = json.loads(line)
            if row.get("type") == "metadata":
                metadata = row
            elif row.get("type") == "trial":
                trials.append(row)

    if metadata is None:
        raise ValueError(f"No metadata row found in {path}.")
    if not trials:
        raise ValueError(f"No trial rows found in {path}.")
    return metadata, trials


def build_drop_frame(trials):
    """Convert trial rows into a tidy table with per-layer accuracy drop."""

    rows = []
    # Use each layer's ratio=0.0 no-op trial as its baseline.
    baseline_by_layer = {
        row["layer_idx"]: row["metrics"]["acc"]
        for row in trials
        if float(row["ratio"]) == 0.0
    }

    for row in trials:
        layer_idx = row["layer_idx"]
        ratio = float(row["ratio"])
        acc = row["metrics"]["acc"]
        baseline = baseline_by_layer[layer_idx]
        rows.append(
            {
                "layer_idx": layer_idx,
                "ratio": ratio,
                "acc": acc,
                "acc_drop": baseline - acc,
            }
        )

    frame = pd.DataFrame(rows)
    # Keep rows in plotting order and discard the old shuffled row index.
    return frame.sort_values(["layer_idx", "ratio"]).reset_index(drop=True)


def plot_heatmap(frame, dataset, output_dir):
    """Plot the full layer x ratio sensitivity matrix."""

    # Rows become layers, columns become pruning ratios, values become colors.
    table = frame.pivot(index="layer_idx", columns="ratio", values="acc_drop")
    table = table.sort_index().reindex(sorted(table.columns), axis=1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(
        table,
        ax=ax,
        cmap="magma",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Accuracy drop (%p)"},
    )
    ax.set_title(f"{dataset} Taylor MLP Sensitivity")
    ax.set_xlabel("Pruning ratio")
    ax.set_ylabel("Transformer block")
    fig.tight_layout()

    path = Path(output_dir) / f"{dataset}_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_all_layer_curves(frame, dataset, output_dir):
    """Plot 12 layer-wise curves in a 3x4 grid for detailed inspection."""

    layers = sorted(frame["layer_idx"].unique())
    # Shared y-axis makes layer-to-layer sensitivity visually comparable.
    max_drop = max(0.1, frame["acc_drop"].max())

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
    # Flatten 3x4 axes into a simple list so we can zip with layer indices.
    axes = axes.flatten()

    for ax, layer_idx in zip(axes, layers):
        layer_frame = frame[frame["layer_idx"] == layer_idx].sort_values("ratio")
        ax.plot(layer_frame["ratio"], layer_frame["acc_drop"], marker="o", linewidth=1.8)
        ax.axhline(0.0, color="0.65", linewidth=0.8)
        ax.set_title(f"Layer {layer_idx}")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(min(-0.1, frame["acc_drop"].min() - 0.05), max_drop * 1.08)

    for ax in axes[len(layers):]:
        # If a future model has fewer than 12 layers, hide unused panels.
        ax.axis("off")

    fig.suptitle(f"{dataset} Accuracy Drop by Layer", y=0.995)
    fig.supxlabel("Pruning ratio")
    fig.supylabel("Accuracy drop (%p)")
    fig.tight_layout()

    path = Path(output_dir) / f"{dataset}_all_layer_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_dataset(path, output_dir):
    """Generate both analysis figures for one dataset result file."""

    metadata, trials = load_results(path)
    dataset = metadata["config"]["dataset"]
    frame = build_drop_frame(trials)

    heatmap_path = plot_heatmap(frame, dataset, output_dir)
    curves_path = plot_all_layer_curves(frame, dataset, output_dir)
    return heatmap_path, curves_path


def main(args):
    """Plot every requested results.jsonl file."""

    os.makedirs(args.output_dir, exist_ok=True)
    # Apply a readable default style to all seaborn/matplotlib figures.
    sns.set_theme(style="whitegrid", context="notebook")

    for result_path in args.results_jsonl:
        heatmap_path, curves_path = plot_dataset(result_path, args.output_dir)
        print(f"[Plot] saved {heatmap_path}")
        print(f"[Plot] saved {curves_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
