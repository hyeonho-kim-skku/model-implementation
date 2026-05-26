"""Overlay global MLP pruning allocation on Taylor sensitivity heatmaps.

The sensitivity heatmaps show accuracy drop for pruning one MLP block at a
fixed ratio. The global-pruning metrics show the actual ratio selected for each
block. This script highlights the sensitivity cells that bracket each selected
global ratio, making the allocation easier to compare with empirical layer
sensitivity.

Run:
  python analysis/plot_mlp_sensitivity_global_overlay.py
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns


DEFAULT_SENSITIVITY_RESULTS = [
    "pruned/vit_base_cifar100_lora50_taylor_sensitivity/results.jsonl",
    "pruned/vit_base_cub200_lora50_taylor_sensitivity/results.jsonl",
    "pruned/vit_base_fgvc_aircraft_lora50_taylor_sensitivity/results.jsonl",
    "pruned/vit_base_stanford_cars_lora50_taylor_sensitivity/results.jsonl",
]

DEFAULT_GLOBAL_METRICS = [
    "pruned/vit_base_cifar100_lora50_mlp040_global_taylor/eval_metrics.json",
    "pruned/vit_base_cub200_lora50_mlp040_global_taylor/eval_metrics.json",
    "pruned/vit_base_fgvc_aircraft_lora50_mlp040_global_taylor/eval_metrics.json",
    "pruned/vit_base_stanford_cars_lora50_mlp040_global_taylor/eval_metrics.json",
]

DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensitivity-results",
        nargs="+",
        default=DEFAULT_SENSITIVITY_RESULTS,
        help="One or more Taylor MLP sensitivity results.jsonl files.",
    )
    parser.add_argument(
        "--global-metrics",
        nargs="+",
        default=DEFAULT_GLOBAL_METRICS,
        help="One or more global Taylor MLP eval_metrics.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/global_mlp_pruning/sensitivity_overlay",
        help="Directory to save generated overlay heatmaps.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved heatmap images.",
    )
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def load_sensitivity_results(path):
    """Read sensitivity trials and return a tidy accuracy-drop table."""

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

    dataset = metadata["config"]["dataset"]
    baseline_acc = metadata["config"]["reference_baseline_metrics"]["acc"]

    rows = []
    for row in trials:
        rows.append(
            {
                "dataset": dataset,
                "layer_idx": row["layer_idx"],
                "ratio": float(row["ratio"]),
                "acc": row["metrics"]["acc"],
                "acc_drop": baseline_acc - row["metrics"]["acc"],
            }
        )
    return pd.DataFrame(rows).sort_values(["layer_idx", "ratio"]).reset_index(drop=True)


def load_global_ratios(paths):
    """Read each eval_metrics.json and collect actual global MLP ratios."""

    rows = []
    for path in paths:
        with open(path, "r") as file:
            metrics = json.load(file)

        dataset = metrics["dataset"]
        by_layer = metrics["pruning_stats"]["target_pruning_summary"]["by_layer"]
        for layer_name, values in by_layer.items():
            if values.get("type") != "mlp":
                continue
            layer_idx = int(layer_name.split(".")[1])
            rows.append(
                {
                    "dataset": dataset,
                    "layer_idx": layer_idx,
                    "global_pruned_ratio": values["pruned_ratio"],
                }
            )

    if not rows:
        raise ValueError("No global MLP pruning ratios found.")
    return pd.DataFrame(rows)


def bracketing_ratios(value, grid):
    """Return the sensitivity grid ratios that bracket a global pruning ratio."""

    lower = max(ratio for ratio in grid if ratio <= value)
    upper = min(ratio for ratio in grid if ratio >= value)
    return [lower] if lower == upper else [lower, upper]


def draw_bracket_boxes(ax, table, global_frame):
    """Draw red boxes on the cells around each layer's global pruning ratio."""

    ratios = list(table.columns)
    layers = list(table.index)
    ratio_to_col = {ratio: idx for idx, ratio in enumerate(ratios)}
    layer_to_row = {layer: idx for idx, layer in enumerate(layers)}

    for row in global_frame.itertuples(index=False):
        if row.layer_idx not in layer_to_row:
            continue
        for ratio in bracketing_ratios(row.global_pruned_ratio, ratios):
            col_idx = ratio_to_col[ratio]
            row_idx = layer_to_row[row.layer_idx]
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.4,
                )
            )


def plot_overlay(sensitivity_frame, global_frame, dataset, output_dir, dpi):
    table = sensitivity_frame.pivot(index="layer_idx", columns="ratio", values="acc_drop")
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
    draw_bracket_boxes(ax, table, global_frame)

    ax.set_title(f"{dataset_label(dataset)} Taylor MLP Sensitivity + Global Allocation")
    ax.set_xlabel("Single-layer pruning ratio")
    ax.set_ylabel("Transformer block")
    fig.tight_layout()

    path = Path(output_dir) / f"{dataset}_sensitivity_global_overlay.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    global_frame = load_global_ratios(args.global_metrics)
    for result_path in args.sensitivity_results:
        sensitivity_frame = load_sensitivity_results(result_path)
        dataset = sensitivity_frame["dataset"].iloc[0]
        dataset_global = global_frame[global_frame["dataset"] == dataset]
        if dataset_global.empty:
            raise ValueError(f"No global pruning ratios found for dataset {dataset}.")

        output_path = plot_overlay(
            sensitivity_frame=sensitivity_frame,
            global_frame=dataset_global,
            dataset=dataset,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
        print(
            f"[Plot] {dataset_label(dataset)}: "
            f"layers={dataset_global['layer_idx'].nunique()}, saved {output_path}"
        )


if __name__ == "__main__":
    main(build_parser().parse_args())
