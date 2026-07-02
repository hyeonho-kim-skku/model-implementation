"""Summarize head-gate Taylor global attention-head pruning results.

Run:
  python analysis/plot_head_gate_taylor_global_pruning.py
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
import seaborn as sns


DATASET_ORDER = ("cifar100", "cub200", "fgvc_aircraft", "stanford_cars")
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="pruned")
    parser.add_argument("--output-dir", default="figures/head_gate_taylor_global_pruning")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def dataset_sort_key(dataset):
    try:
        return DATASET_ORDER.index(dataset)
    except ValueError:
        return len(DATASET_ORDER), dataset


def discover_result_paths(results_root):
    pattern = "vit_base_*_lora50_head_gate_taylor_proj_in_sum_abs_samplewise_global*/results.jsonl"
    paths = sorted(Path(results_root).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No head-gate Taylor global results found under {results_root!r}.")
    return paths


def load_jsonl(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = [row for row in rows if row.get("type") == "metadata"]
    trials = [row for row in rows if row.get("type") == "trial"]
    if len(metadata) != 1 or len(trials) != 1:
        raise ValueError(f"Expected one metadata row and one trial row in {path}.")
    return metadata[0], trials[0]


def layer_idx(layer_name):
    match = re.fullmatch(r"blocks\.(\d+)\.attn", layer_name)
    if match is None:
        raise ValueError(f"Unexpected attention layer name: {layer_name}")
    return int(match.group(1))


def rows_from_result(path):
    metadata, trial = load_jsonl(path)
    config = metadata["config"]
    dataset = config["dataset"]
    ratio = float(trial["ratio"])
    baseline = config["reference_baseline_metrics"]
    metrics = trial["metrics"]
    pruning_stats = trial["pruning_stats"]
    summary = pruning_stats["target_pruning_summary"]
    overall = summary["overall"]["head"]
    by_layer = summary["by_layer"]
    selected_heads = pruning_stats.get("selected_attention_heads", {})

    summary_row = {
        "dataset": dataset,
        "dataset_label": dataset_label(dataset),
        "ratio": ratio,
        "ratio_label": f"{ratio:.2f}",
        "baseline_acc": float(baseline["acc"]),
        "pruned_acc": float(metrics["acc"]),
        "acc_drop": float(baseline["acc"]) - float(metrics["acc"]),
        "baseline_loss": float(baseline["loss"]),
        "pruned_loss": float(metrics["loss"]),
        "heads_before": int(overall["heads_before"]),
        "heads_after": int(overall["heads_after"]),
        "pruned_heads": int(overall["pruned_heads"]),
        "overall_pruned_ratio": float(overall["pruned_ratio"]),
        "base_params": pruning_stats["base_params"],
        "pruned_params": pruning_stats["pruned_params"],
        "base_macs": pruning_stats["base_macs"],
        "pruned_macs": pruning_stats["pruned_macs"],
        "cache_loaded": bool(config.get("score_cache_loaded")),
        "artifact_path": trial.get("artifact_path"),
        "save_artifacts": bool(config.get("save_artifacts")),
        "results_path": str(path),
    }

    layer_rows = []
    for name, values in by_layer.items():
        if values.get("type") != "head":
            continue
        idx = layer_idx(name)
        selected = selected_heads.get(str(idx), selected_heads.get(idx, []))
        layer_rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label(dataset),
                "ratio": ratio,
                "ratio_label": f"{ratio:.2f}",
                "layer_idx": idx,
                "layer_name": name,
                "heads_before": int(values["heads_before"]),
                "heads_after": int(values["heads_after"]),
                "pruned_heads": int(values["pruned_heads"]),
                "layer_pruned_ratio": float(values["pruned_ratio"]),
                "selected_heads": ",".join(str(head) for head in selected),
                "results_path": str(path),
            }
        )

    if len(layer_rows) != 12:
        raise ValueError(f"Expected 12 attention layers in {path}, found {len(layer_rows)}.")
    return summary_row, layer_rows


def build_frames(results_root):
    summary_rows = []
    layer_rows = []
    for path in discover_result_paths(results_root):
        summary_row, result_layer_rows = rows_from_result(path)
        summary_rows.append(summary_row)
        layer_rows.extend(result_layer_rows)

    summary = pd.DataFrame(summary_rows)
    layers = pd.DataFrame(layer_rows)
    summary = summary.sort_values(
        by=["dataset", "ratio"],
        key=lambda col: col.map(dataset_sort_key) if col.name == "dataset" else col,
    ).reset_index(drop=True)
    layers = layers.sort_values(
        by=["dataset", "ratio", "layer_idx"],
        key=lambda col: col.map(dataset_sort_key) if col.name == "dataset" else col,
    ).reset_index(drop=True)
    return summary, layers


def save_csvs(summary, layers, output_dir):
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.csv"
    layers_path = output_dir / "layers.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    layers.to_csv(layers_path, index=False, float_format="%.4f")
    return summary_path, layers_path


def plot_accuracy(summary, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    sns.lineplot(
        data=summary,
        x="ratio",
        y="pruned_acc",
        hue="dataset_label",
        marker="o",
        linewidth=2.0,
        ax=ax,
    )
    for dataset, frame in summary.groupby("dataset", sort=False):
        baseline = frame["baseline_acc"].iloc[0]
        ax.axhline(
            baseline,
            color=ax.get_lines()[-1].get_color() if ax.get_lines() else "gray",
            linewidth=0.8,
            linestyle="--",
            alpha=0.35,
        )
    ax.set_title("Global Head-Gate Taylor Pruning Accuracy")
    ax.set_xlabel("Global head pruning ratio")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = Path(output_dir) / "accuracy_curve.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_head_counts(summary, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    sns.lineplot(
        data=summary,
        x="ratio",
        y="pruned_heads",
        hue="dataset_label",
        marker="o",
        linewidth=2.0,
        ax=ax,
    )
    ax.set_title("Pruned Attention Heads by Ratio")
    ax.set_xlabel("Global head pruning ratio")
    ax.set_ylabel("Pruned heads")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = Path(output_dir) / "pruned_heads_curve.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_layer_heatmap(layers, output_dir, dpi):
    frame = layers.copy()
    frame["row_label"] = frame["dataset_label"] + " " + frame["ratio_label"]
    frame = frame.sort_values(
        by=["dataset", "ratio", "layer_idx"],
        key=lambda col: col.map(dataset_sort_key) if col.name == "dataset" else col,
    )
    row_order = frame[["dataset", "ratio", "row_label"]].drop_duplicates()["row_label"]
    table = frame.pivot(index="row_label", columns="layer_idx", values="pruned_heads")
    table = table.reindex(row_order).reindex(range(12), axis=1)

    height = max(4.8, 0.32 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(12.2, height))
    sns.heatmap(
        table,
        ax=ax,
        cmap="mako_r",
        vmin=0,
        vmax=11,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Pruned heads in layer"},
    )
    ax.set_title("Global Head-Gate Taylor Pruning Allocation by Layer")
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Dataset / ratio")
    fig.tight_layout()

    path = Path(output_dir) / "layer_pruned_heads_heatmap.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def print_summary(summary):
    for dataset, frame in summary.groupby("dataset", sort=False):
        baseline = frame["baseline_acc"].iloc[0]
        best = frame.loc[frame["pruned_acc"].idxmax()]
        print(
            f"[Plot] {dataset_label(dataset)}: baseline={baseline:.2f}, "
            f"best_ratio={best['ratio']:.2f}, acc={best['pruned_acc']:.2f}, "
            f"drop={baseline - best['pruned_acc']:.2f}, "
            f"pruned_heads={int(best['pruned_heads'])}"
        )


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    summary, layers = build_frames(args.results_root)
    print_summary(summary)
    summary_path, layers_path = save_csvs(summary, layers, output_dir)
    accuracy_path = plot_accuracy(summary, output_dir, args.dpi)
    counts_path = plot_head_counts(summary, output_dir, args.dpi)
    heatmap_path = plot_layer_heatmap(layers, output_dir, args.dpi)

    print(f"[Plot] saved {summary_path}")
    print(f"[Plot] saved {layers_path}")
    print(f"[Plot] saved {accuracy_path}")
    print(f"[Plot] saved {counts_path}")
    print(f"[Plot] saved {heatmap_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
