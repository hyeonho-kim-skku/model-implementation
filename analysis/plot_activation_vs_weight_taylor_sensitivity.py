"""Compare weight-Taylor and activation-Taylor MLP sensitivity results.

Inputs are paired results.jsonl files from sensitivity_taylor.py. The default
pairs compare the original weight*gradient Taylor sensitivity with the
activation_taylor sum_abs sensitivity for each downstream dataset.

Run:
  python analysis/plot_activation_vs_weight_taylor_sensitivity.py
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


DEFAULT_PAIRS = [
    (
        "cifar100",
        "pruned/vit_base_cifar100_lora50_taylor_sensitivity/results.jsonl",
        "pruned/vit_base_cifar100_lora50_activation_taylor_sum_abs_sensitivity/results.jsonl",
    ),
    (
        "cub200",
        "pruned/vit_base_cub200_lora50_taylor_sensitivity/results.jsonl",
        "pruned/vit_base_cub200_lora50_activation_taylor_sum_abs_sensitivity/results.jsonl",
    ),
    (
        "fgvc_aircraft",
        "pruned/vit_base_fgvc_aircraft_lora50_taylor_sensitivity/results.jsonl",
        "pruned/vit_base_fgvc_aircraft_lora50_activation_taylor_sum_abs_sensitivity/results.jsonl",
    ),
    (
        "stanford_cars",
        "pruned/vit_base_stanford_cars_lora50_taylor_sensitivity/results.jsonl",
        "pruned/vit_base_stanford_cars_lora50_activation_taylor_sum_abs_sensitivity/results.jsonl",
    ),
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
        "--output-dir",
        default="figures/activation_vs_weight_taylor_sensitivity",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved figures.",
    )
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def load_results(path, method):
    """Read one sensitivity results file into a tidy table."""

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
        acc = row["metrics"]["acc"]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "layer_idx": row["layer_idx"],
                "ratio": float(row["ratio"]),
                "acc": acc,
                "acc_drop": baseline_acc - acc,
            }
        )
    return pd.DataFrame(rows).sort_values(["layer_idx", "ratio"]).reset_index(drop=True)


def load_pair(dataset, weight_path, activation_path):
    weight_frame = load_results(weight_path, method="weight_taylor")
    activation_frame = load_results(activation_path, method="activation_taylor")

    if weight_frame["dataset"].iloc[0] != activation_frame["dataset"].iloc[0]:
        raise ValueError(f"Dataset mismatch for pair {dataset}.")

    merged = weight_frame.merge(
        activation_frame,
        on=["dataset", "layer_idx", "ratio"],
        suffixes=("_weight", "_activation"),
    )
    if merged.empty:
        raise ValueError(f"No overlapping layer/ratio rows for {dataset}.")

    merged["acc_diff"] = merged["acc_activation"] - merged["acc_weight"]
    merged["drop_diff"] = merged["acc_drop_activation"] - merged["acc_drop_weight"]
    return merged


def pivot(frame, value):
    table = frame.pivot(index="layer_idx", columns="ratio", values=value)
    return table.sort_index().reindex(sorted(table.columns), axis=1)


def plot_dataset_heatmaps(frame, dataset, output_dir, dpi):
    """Save weight, activation, and activation-minus-weight heatmaps."""

    weight_drop = pivot(frame, "acc_drop_weight")
    activation_drop = pivot(frame, "acc_drop_activation")
    acc_diff = pivot(frame, "acc_diff")

    vmax_drop = max(weight_drop.max().max(), activation_drop.max().max())
    vmax_drop = max(0.1, vmax_drop)
    diff_abs = max(abs(acc_diff.min().min()), abs(acc_diff.max().max()), 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(22, 5.8))
    heatmaps = [
        (weight_drop, "Weight Taylor accuracy drop", "magma", 0.0, vmax_drop),
        (activation_drop, "Activation Taylor accuracy drop", "magma", 0.0, vmax_drop),
        (acc_diff, "Activation - weight accuracy", "coolwarm", -diff_abs, diff_abs),
    ]

    for ax, (table, title, cmap, vmin, vmax) in zip(axes, heatmaps):
        cbar_label = "Accuracy drop (%p)" if "drop" in title else "Accuracy diff (%p)"
        sns.heatmap(
            table,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0.0 if "diff" in cbar_label.lower() else None,
            annot=True,
            fmt=".2f",
            linewidths=0.35,
            cbar_kws={"label": cbar_label},
        )
        ax.set_title(title)
        ax.set_xlabel("Pruning ratio")
        ax.set_ylabel("Transformer block")

    fig.suptitle(f"{dataset_label(dataset)} MLP Sensitivity Comparison", y=1.02)
    fig.tight_layout()

    path = Path(output_dir) / f"{dataset}_weight_vs_activation_taylor_heatmaps.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ratio_mean_diff(all_frame, output_dir, dpi):
    """Plot mean activation-minus-weight accuracy by pruning ratio."""

    ratio_frame = (
        all_frame.groupby(["dataset", "ratio"], as_index=False)["acc_diff"]
        .mean()
        .sort_values(["dataset", "ratio"])
    )

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for dataset, dataset_frame in ratio_frame.groupby("dataset"):
        ax.plot(
            dataset_frame["ratio"],
            dataset_frame["acc_diff"],
            marker="o",
            linewidth=2.0,
            label=dataset_label(dataset),
        )

    ax.axhline(0.0, color="0.25", linewidth=1.0)
    ax.set_title("Mean Accuracy Difference by Pruning Ratio")
    ax.set_xlabel("Pruning ratio")
    ax.set_ylabel("Activation Taylor - weight Taylor accuracy (%p)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Dataset")
    fig.tight_layout()

    path = Path(output_dir) / "ratio_mean_accuracy_diff.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_layer_mean_sensitivity(all_frame, output_dir, dpi):
    """Plot layer-wise mean accuracy drop for both Taylor criteria."""

    ratio_positive = all_frame[all_frame["ratio"] > 0.0].copy()
    rows = []
    for row in ratio_positive.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "layer_idx": row.layer_idx,
                "method": "Weight Taylor",
                "acc_drop": row.acc_drop_weight,
            }
        )
        rows.append(
            {
                "dataset": row.dataset,
                "layer_idx": row.layer_idx,
                "method": "Activation Taylor",
                "acc_drop": row.acc_drop_activation,
            }
        )
    plot_frame = pd.DataFrame(rows)
    plot_frame = (
        plot_frame.groupby(["dataset", "layer_idx", "method"], as_index=False)["acc_drop"]
        .mean()
        .sort_values(["dataset", "layer_idx", "method"])
    )

    datasets = list(dict.fromkeys(plot_frame["dataset"]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()

    for ax, dataset in zip(axes, datasets):
        dataset_frame = plot_frame[plot_frame["dataset"] == dataset]
        for method, method_frame in dataset_frame.groupby("method"):
            ax.plot(
                method_frame["layer_idx"],
                method_frame["acc_drop"],
                marker="o",
                linewidth=2.0,
                label=method,
            )
        ax.set_title(dataset_label(dataset))
        ax.set_xlabel("Transformer block")
        ax.set_ylabel("Mean accuracy drop (%p)")
        ax.grid(True, alpha=0.25)
        ax.legend()

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle("Layer-wise Mean MLP Sensitivity", y=0.995)
    fig.tight_layout()

    path = Path(output_dir) / "layer_mean_accuracy_drop.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_summary_csv(all_frame, output_dir):
    summary = (
        all_frame.groupby(["dataset", "ratio"], as_index=False)
        .agg(
            mean_acc_diff=("acc_diff", "mean"),
            min_acc_diff=("acc_diff", "min"),
            max_acc_diff=("acc_diff", "max"),
            better_layers=("acc_diff", lambda values: int((values > 0).sum())),
            compared_layers=("acc_diff", "count"),
        )
        .sort_values(["dataset", "ratio"])
    )
    path = Path(output_dir) / "ratio_mean_accuracy_diff.csv"
    summary.to_csv(path, index=False)
    return path


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    merged_frames = []
    for dataset, weight_path, activation_path in DEFAULT_PAIRS:
        frame = load_pair(dataset, weight_path, activation_path)
        merged_frames.append(frame)
        path = plot_dataset_heatmaps(frame, dataset, args.output_dir, args.dpi)
        print(f"[Plot] saved {path}")

    all_frame = pd.concat(merged_frames, ignore_index=True)
    ratio_path = plot_ratio_mean_diff(all_frame, args.output_dir, args.dpi)
    layer_path = plot_layer_mean_sensitivity(all_frame, args.output_dir, args.dpi)
    csv_path = write_summary_csv(all_frame, args.output_dir)

    print(f"[Plot] saved {ratio_path}")
    print(f"[Plot] saved {layer_path}")
    print(f"[Plot] saved {csv_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
