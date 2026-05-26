"""Plot global Taylor MLP pruning allocation across ViT blocks.

Inputs are eval_metrics.json files produced after global Taylor MLP pruning.
Each file already contains post-pruning structural statistics, so this script
only reads those summaries and visualizes how the global pruning budget was
allocated across transformer blocks.

Run:
  python analysis/plot_global_mlp_pruning.py
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_METRICS = [
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
        "--metrics-json",
        nargs="+",
        default=DEFAULT_METRICS,
        help="One or more global Taylor MLP eval_metrics.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/global_mlp_pruning",
        help="Directory to save generated figure and CSV.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for the saved heatmap image.",
    )
    return parser


def _dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def _layer_idx(layer_name):
    match = re.fullmatch(r"blocks\.(\d+)\.mlp", layer_name)
    if match is None:
        raise ValueError(f"Unexpected MLP layer name: {layer_name}")
    return int(match.group(1))


def load_metrics(path):
    """Read one eval_metrics.json file and extract per-block MLP pruning rows."""

    with open(path, "r") as file:
        metrics = json.load(file)

    dataset = metrics["dataset"]
    pruning_config = metrics["pruning_config"]
    summary = metrics["pruning_stats"]["target_pruning_summary"]
    overall = summary["overall"]["mlp"]
    by_layer = summary["by_layer"]

    rows = []
    for layer_name, values in by_layer.items():
        if values.get("type") != "mlp":
            continue
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": _dataset_label(dataset),
                "layer_idx": _layer_idx(layer_name),
                "layer_name": layer_name,
                "global_pruning": pruning_config["global_pruning"],
                "configured_pruning_ratio": pruning_config["pruning_ratio"],
                "overall_pruned_ratio": overall["pruned_ratio"],
                "hidden_before": values["hidden_before"],
                "hidden_after": values["hidden_after"],
                "pruned_hidden": values["pruned_hidden"],
                "pruned_ratio": values["pruned_ratio"],
                "source_path": str(path),
            }
        )

    if not rows:
        raise ValueError(f"No MLP pruning rows found in {path}.")

    return rows


def build_frame(paths):
    """Build a tidy table with one row per dataset and transformer block."""

    rows = []
    for path in paths:
        rows.extend(load_metrics(path))

    frame = pd.DataFrame(rows)
    return frame.sort_values(["dataset_label", "layer_idx"]).reset_index(drop=True)


def save_csv(frame, output_dir):
    path = Path(output_dir) / "global_mlp_pruned_ratio.csv"
    frame.to_csv(path, index=False)
    return path


def plot_heatmap(frame, output_dir, dpi):
    """Plot dataset x block pruning ratios.

    Global pruning removes one overall budget, but it does not force each block
    to lose the same ratio. This heatmap shows that layer-wise allocation.
    """

    table = frame.pivot(index="dataset_label", columns="layer_idx", values="pruned_ratio")
    table = table.reindex(frame["dataset_label"].drop_duplicates())
    table = table.reindex(sorted(table.columns), axis=1)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    sns.heatmap(
        table,
        ax=ax,
        cmap="viridis",
        annot=table * 100.0,
        fmt=".1f",
        linewidths=0.5,
        linecolor="white",
        vmin=0.0,
        vmax=max(1.0, float(table.max().max())),
        cbar_kws={"label": "Pruned MLP hidden ratio"},
    )
    ax.set_title("Layer-wise MLP Pruning Ratios under Global Taylor Pruning")
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Dataset")
    fig.tight_layout()

    path = Path(output_dir) / "global_mlp_pruned_ratio_heatmap.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def print_summary(frame):
    for dataset_label, dataset_frame in frame.groupby("dataset_label", sort=False):
        overall_ratio = dataset_frame["overall_pruned_ratio"].iloc[0]
        configured_ratio = dataset_frame["configured_pruning_ratio"].iloc[0]
        print(
            f"[Plot] {dataset_label}: "
            f"blocks={len(dataset_frame)}, "
            f"configured_ratio={configured_ratio:.4f}, "
            f"overall_pruned_ratio={overall_ratio:.4f}"
        )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    frame = build_frame(args.metrics_json)
    print_summary(frame)

    csv_path = save_csv(frame, args.output_dir)
    heatmap_path = plot_heatmap(frame, args.output_dir, args.dpi)

    print(f"[Plot] saved {csv_path}")
    print(f"[Plot] saved {heatmap_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
