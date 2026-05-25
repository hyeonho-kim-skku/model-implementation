"""Plot Taylor attention-head sensitivity results.

Inputs are results.jsonl files from sensitivity_taylor.py with
pruning_modules=head. For each dataset, this script writes:
  - <dataset>_head_heatmap.png: layer x pruned-head-count accuracy-drop heatmap
  - <dataset>_head_all_layer_curves.png: 12 subplot accuracy-drop curves

Run:
  python analysis/plot_taylor_head_sensitivity.py
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
    "pruned/vit_base_cifar100_lora50_taylor_head_sensitivity/results.jsonl",
    "pruned/vit_base_cub200_lora50_taylor_head_sensitivity/results.jsonl",
    "pruned/vit_base_fgvc_aircraft_lora50_taylor_head_sensitivity/results.jsonl",
    "pruned/vit_base_stanford_cars_lora50_taylor_head_sensitivity/results.jsonl",
]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-jsonl",
        nargs="+",
        default=DEFAULT_RESULTS,
        help="One or more head-sensitivity results.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/taylor_head_sensitivity",
        help="Directory to save generated figures.",
    )
    return parser


def load_results(path):
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


def build_drop_frame(metadata, trials):
    baseline_acc = metadata["config"]["reference_baseline_metrics"]["acc"]
    rows = []
    for row in trials:
        acc = row["metrics"]["acc"]
        rows.append(
            {
                "layer_idx": row["layer_idx"],
                "pruned_head_count": row["pruned_head_count"],
                "remaining_head_count": row["remaining_head_count"],
                "acc": acc,
                "acc_drop": baseline_acc - acc,
            }
        )
    return pd.DataFrame(rows).sort_values(["layer_idx", "pruned_head_count"]).reset_index(drop=True)


def plot_heatmap(frame, dataset, output_dir):
    table = frame.pivot(index="layer_idx", columns="pruned_head_count", values="acc_drop")
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
    ax.set_title(f"{dataset} Taylor Attention-Head Sensitivity")
    ax.set_xlabel("Pruned attention heads")
    ax.set_ylabel("Transformer block")
    fig.tight_layout()

    path = Path(output_dir) / f"{dataset}_head_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_all_layer_curves(frame, dataset, output_dir):
    layers = sorted(frame["layer_idx"].unique())
    max_drop = max(0.1, frame["acc_drop"].max())

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, layer_idx in zip(axes, layers):
        layer_frame = frame[frame["layer_idx"] == layer_idx].sort_values("pruned_head_count")
        ax.plot(
            layer_frame["pruned_head_count"],
            layer_frame["acc_drop"],
            marker="o",
            linewidth=1.8,
        )
        ax.axhline(0.0, color="0.65", linewidth=0.8)
        ax.set_title(f"Layer {layer_idx}")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(sorted(frame["pruned_head_count"].unique()))
        ax.set_ylim(min(-0.1, frame["acc_drop"].min() - 0.05), max_drop * 1.08)

    for ax in axes[len(layers):]:
        ax.axis("off")

    fig.suptitle(f"{dataset} Accuracy Drop by Pruned Attention Heads", y=0.995)
    fig.supxlabel("Pruned attention heads")
    fig.supylabel("Accuracy drop (%p)")
    fig.tight_layout()

    path = Path(output_dir) / f"{dataset}_head_all_layer_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_dataset(path, output_dir):
    metadata, trials = load_results(path)
    dataset = metadata["config"]["dataset"]
    frame = build_drop_frame(metadata, trials)

    heatmap_path = plot_heatmap(frame, dataset, output_dir)
    curves_path = plot_all_layer_curves(frame, dataset, output_dir)
    return heatmap_path, curves_path


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    for result_path in args.results_jsonl:
        heatmap_path, curves_path = plot_dataset(result_path, args.output_dir)
        print(f"[Plot] saved {heatmap_path}")
        print(f"[Plot] saved {curves_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
