"""Plot layer-wise pruning allocation for selected aggregation comparisons.

Run:
  python analysis/plot_gate_taylor_aggregation_layer_distribution.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}

COMPARISONS = (
    {
        "dataset": "cub200",
        "ratio": 0.6,
        "aggregations": ("elementwise", "tokenwise"),
        "title": "CUB200 60%",
    },
    {
        "dataset": "stanford_cars",
        "ratio": 0.6,
        "aggregations": ("elementwise", "tokenwise"),
        "title": "Stanford Cars 60%",
    },
    {
        "dataset": "fgvc_aircraft",
        "ratio": 0.5,
        "aggregations": ("elementwise", "tokenwise"),
        "title": "FGVC-Aircraft 50%",
    },
    {
        "dataset": "cifar100",
        "ratio": 0.5,
        "aggregations": ("elementwise", "samplewise"),
        "title": "CIFAR100 50%",
    },
)

AGGREGATION_LABELS = {
    "elementwise": "Elementwise",
    "samplewise": "Samplewise",
    "tokenwise": "Tokenwise",
    "channelwise": "Channelwise",
}

AGGREGATION_COLORS = {
    "elementwise": "#344054",
    "samplewise": "#039855",
    "tokenwise": "#175CD3",
    "channelwise": "#B42318",
}

AGGREGATION_MARKERS = {
    "elementwise": "o",
    "samplewise": "s",
    "tokenwise": "^",
    "channelwise": "D",
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="pruned")
    parser.add_argument("--output-dir", default="figures/gate_taylor_aggregation_global")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def ratio_tag(ratio):
    return f"global{int(round(ratio * 100)):03d}"


def ratio_label(ratio):
    return f"{int(round(ratio * 100))}%"


def result_path(results_root, dataset, ratio, aggregation):
    aggregation_suffix = "" if aggregation == "elementwise" else f"_{aggregation}"
    folder = (
        f"vit_base_{dataset}_lora50_gate_taylor_"
        f"fc2_in_sum_square{aggregation_suffix}_{ratio_tag(ratio)}"
    )
    return Path(results_root) / folder / "results.jsonl"


def layer_idx(layer_name):
    match = re.fullmatch(r"blocks\.(\d+)\.mlp", layer_name)
    if match is None:
        raise ValueError(f"Unexpected layer name: {layer_name}")
    return int(match.group(1))


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = [row for row in rows if row.get("type") == "metadata"]
    trials = [row for row in rows if row.get("type") == "trial"]
    if len(metadata) != 1 or len(trials) != 1:
        raise ValueError(f"Expected one metadata row and one trial row in {path}.")
    return metadata[0], trials[0]


def rows_from_result(results_root, dataset, ratio, aggregation):
    path = result_path(results_root, dataset, ratio, aggregation)
    metadata, trial = load_jsonl(path)
    config = metadata["config"]
    if config["dataset"] != dataset:
        raise ValueError(f"Dataset mismatch in {path}: {config['dataset']}")
    if abs(float(trial["ratio"]) - ratio) > 1e-12:
        raise ValueError(f"Ratio mismatch in {path}: {trial['ratio']}")
    actual_aggregation = trial["pruning_config"].get("gate_taylor_aggregation", "elementwise")
    if actual_aggregation != aggregation:
        raise ValueError(f"Aggregation mismatch in {path}: {actual_aggregation}")

    baseline_acc = float(config["reference_baseline_metrics"]["acc"])
    pruned_acc = float(trial["metrics"]["acc"])
    summary = trial["pruning_stats"]["target_pruning_summary"]
    by_layer = summary["by_layer"]
    if len(by_layer) != 12:
        raise ValueError(f"Expected 12 MLP layers in {path}, found {len(by_layer)}.")

    rows = []
    for name, values in by_layer.items():
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label(dataset),
                "ratio": ratio,
                "ratio_label": ratio_label(ratio),
                "aggregation": aggregation,
                "aggregation_label": AGGREGATION_LABELS[aggregation],
                "layer_idx": layer_idx(name),
                "layer_name": name,
                "hidden_before": values["hidden_before"],
                "hidden_after": values["hidden_after"],
                "pruned_hidden": values["pruned_hidden"],
                "layer_pruned_ratio": values["pruned_ratio"],
                "baseline_acc": baseline_acc,
                "pruned_acc": pruned_acc,
                "acc_drop": baseline_acc - pruned_acc,
                "artifact_path": trial.get("artifact_path"),
                "results_path": str(path),
            }
        )
    return rows


def build_layer_frame(results_root):
    rows = []
    for comparison in COMPARISONS:
        for aggregation in comparison["aggregations"]:
            rows.extend(
                rows_from_result(
                    results_root,
                    comparison["dataset"],
                    comparison["ratio"],
                    aggregation,
                )
            )
    return pd.DataFrame(rows)


def add_comparison_deltas(layers):
    delta_rows = []
    group_columns = ["dataset", "ratio", "layer_idx"]
    for (_dataset, _ratio, layer), group in layers.groupby(group_columns):
        if "elementwise" not in set(group["aggregation"]):
            continue
        elementwise_ratio = float(group[group["aggregation"] == "elementwise"]["layer_pruned_ratio"].iloc[0])
        for _, row in group.iterrows():
            delta = float(row["layer_pruned_ratio"]) - elementwise_ratio
            delta_rows.append({**row.to_dict(), "delta_vs_elementwise": delta})
    return pd.DataFrame(delta_rows)


def plot_layer_distribution(layers, output_dir, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 7.6), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, comparison in zip(axes, COMPARISONS):
        dataset = comparison["dataset"]
        ratio = comparison["ratio"]
        frame = layers[(layers["dataset"] == dataset) & (layers["ratio"] == ratio)]
        for aggregation in comparison["aggregations"]:
            agg_frame = frame[frame["aggregation"] == aggregation].sort_values("layer_idx")
            label = AGGREGATION_LABELS[aggregation]
            acc = float(agg_frame["pruned_acc"].iloc[0])
            ax.plot(
                agg_frame["layer_idx"],
                agg_frame["layer_pruned_ratio"],
                label=f"{label} ({acc:.2f})",
                color=AGGREGATION_COLORS[aggregation],
                marker=AGGREGATION_MARKERS[aggregation],
                linewidth=2.2,
                markersize=5.0,
            )

        ax.set_title(comparison["title"], fontsize=12.5, weight="bold")
        ax.set_xticks(range(12))
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", color="#EAECF0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9.2, loc="upper left")

    for ax in axes[2:]:
        ax.set_xlabel("Transformer block")
    for ax in axes[::2]:
        ax.set_ylabel("Layer pruning ratio")

    fig.suptitle(
        "Layer-Wise MLP Pruning Allocation by Gate Taylor Aggregation",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.018,
        "Parentheses in legends show pruning-only accuracy for the corresponding artifact.",
        ha="center",
        fontsize=9.8,
        color="#475467",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    output_path = Path(output_dir) / "gate_taylor_aggregation_layer_pruned_ratio.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_delta_distribution(layers, output_dir, dpi):
    deltas = add_comparison_deltas(layers)
    deltas = deltas[deltas["aggregation"] != "elementwise"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 7.6), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, comparison in zip(axes, COMPARISONS):
        dataset = comparison["dataset"]
        ratio = comparison["ratio"]
        frame = deltas[(deltas["dataset"] == dataset) & (deltas["ratio"] == ratio)]
        for aggregation in comparison["aggregations"]:
            if aggregation == "elementwise":
                continue
            agg_frame = frame[frame["aggregation"] == aggregation].sort_values("layer_idx")
            ax.bar(
                agg_frame["layer_idx"],
                agg_frame["delta_vs_elementwise"],
                color=AGGREGATION_COLORS[aggregation],
                alpha=0.86,
                label=f"{AGGREGATION_LABELS[aggregation]} - Elementwise",
            )

        ax.axhline(0.0, color="#344054", linewidth=1.0)
        ax.set_title(comparison["title"], fontsize=12.5, weight="bold")
        ax.set_xticks(range(12))
        ax.grid(axis="y", color="#EAECF0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9.2, loc="upper left")

    for ax in axes[2:]:
        ax.set_xlabel("Transformer block")
    for ax in axes[::2]:
        ax.set_ylabel("Pruned-ratio delta")

    fig.suptitle(
        "Layer Allocation Shift Relative to Elementwise Aggregation",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    output_path = Path(output_dir) / "gate_taylor_aggregation_layer_pruned_ratio_delta.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path, deltas


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = build_layer_frame(Path(args.results_root))
    layers = add_comparison_deltas(layers)
    csv_path = output_dir / "gate_taylor_aggregation_layer_pruned_ratio.csv"
    layers.to_csv(csv_path, index=False)

    distribution_path = plot_layer_distribution(layers, output_dir, args.dpi)
    delta_path, _deltas = plot_delta_distribution(layers, output_dir, args.dpi)

    print(f"[GateTaylorAggregationLayers] layer rows={len(layers)}, saved {csv_path}")
    print(f"[GateTaylorAggregationLayers] saved {distribution_path}")
    print(f"[GateTaylorAggregationLayers] saved {delta_path}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
