"""Summarize cached gate-Taylor global pruning results.

Run:
  python analysis/plot_gate_taylor_global_pruning.py
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


DATASETS = ("cifar100", "cub200", "fgvc_aircraft", "stanford_cars")
RATIOS = (0.4, 0.5, 0.6)
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="pruned")
    parser.add_argument("--output-dir", default="figures/gate_taylor_global_pruning")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def ratio_tag(ratio):
    return f"global{int(round(ratio * 100)):03d}"


def result_path(results_root, dataset, ratio):
    folder = (
        f"vit_base_{dataset}_lora50_gate_taylor_"
        f"fc2_in_sum_square_{ratio_tag(ratio)}"
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


def rows_from_result(path, expected_dataset, expected_ratio):
    metadata, trial = load_jsonl(path)
    config = metadata["config"]
    if config["dataset"] != expected_dataset:
        raise ValueError(f"Dataset mismatch in {path}: {config['dataset']}")
    if abs(float(trial["ratio"]) - expected_ratio) > 1e-12:
        raise ValueError(f"Ratio mismatch in {path}: {trial['ratio']}")

    baseline_acc = float(config["reference_baseline_metrics"]["acc"])
    pruned_acc = float(trial["metrics"]["acc"])
    pruning_stats = trial["pruning_stats"]
    summary = pruning_stats["target_pruning_summary"]
    overall = summary["overall"]["mlp"]
    by_layer = summary["by_layer"]
    if len(by_layer) != 12:
        raise ValueError(f"Expected 12 MLP layers in {path}, found {len(by_layer)}.")

    summary_row = {
        "dataset": expected_dataset,
        "dataset_label": dataset_label(expected_dataset),
        "ratio": expected_ratio,
        "ratio_label": f"{int(round(expected_ratio * 100))}%",
        "baseline_acc": baseline_acc,
        "pruned_acc": pruned_acc,
        "acc_drop": baseline_acc - pruned_acc,
        "baseline_loss": float(config["reference_baseline_metrics"]["loss"]),
        "pruned_loss": float(trial["metrics"]["loss"]),
        "base_params": pruning_stats["base_params"],
        "pruned_params": pruning_stats["pruned_params"],
        "base_macs": pruning_stats["base_macs"],
        "pruned_macs": pruning_stats["pruned_macs"],
        "overall_pruned_ratio": overall["pruned_ratio"],
        "cache_loaded": bool(config.get("score_cache_loaded")),
        "artifact_path": trial.get("artifact_path"),
        "results_path": str(path),
    }

    layer_rows = []
    for name, values in by_layer.items():
        layer_rows.append(
            {
                "dataset": expected_dataset,
                "dataset_label": dataset_label(expected_dataset),
                "ratio": expected_ratio,
                "ratio_label": f"{int(round(expected_ratio * 100))}%",
                "layer_idx": layer_idx(name),
                "layer_name": name,
                "hidden_before": values["hidden_before"],
                "hidden_after": values["hidden_after"],
                "pruned_hidden": values["pruned_hidden"],
                "layer_pruned_ratio": values["pruned_ratio"],
            }
        )

    max_ratio = max(row["layer_pruned_ratio"] for row in layer_rows)
    min_ratio = min(row["layer_pruned_ratio"] for row in layer_rows)
    for row in layer_rows:
        row["is_max_pruned_layer"] = row["layer_pruned_ratio"] == max_ratio
        row["is_min_pruned_layer"] = row["layer_pruned_ratio"] == min_ratio

    return summary_row, layer_rows


def build_frames(results_root):
    summary_rows = []
    layer_rows = []
    for dataset in DATASETS:
        for ratio in RATIOS:
            path = result_path(results_root, dataset, ratio)
            summary_row, result_layer_rows = rows_from_result(path, dataset, ratio)
            summary_rows.append(summary_row)
            layer_rows.extend(result_layer_rows)

    summary = pd.DataFrame(summary_rows)
    layers = pd.DataFrame(layer_rows)
    return summary, layers


def plot_accuracy_table(summary, output_dir, dpi):
    rows = []
    for dataset in DATASETS:
        dataset_frame = summary[summary["dataset"] == dataset]
        row = {"Dataset": dataset_label(dataset)}
        baseline = dataset_frame["baseline_acc"].iloc[0]
        row["Baseline"] = f"{baseline:.2f}"
        for ratio in RATIOS:
            ratio_frame = dataset_frame[dataset_frame["ratio"] == ratio].iloc[0]
            tag = f"{int(round(ratio * 100))}%"
            row[f"{tag} Acc"] = f"{ratio_frame['pruned_acc']:.2f}"
            row[f"{tag} Drop"] = f"{ratio_frame['acc_drop']:.2f}"
        rows.append(row)

    table_frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12.2, 2.6))
    ax.axis("off")
    table = ax.table(
        cellText=table_frame.values,
        colLabels=table_frame.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.55)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        if row_idx == 0:
            cell.set_facecolor("#344054")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row_idx % 2 else "#F9FAFB")

    ax.set_title("Gate Taylor Global MLP Pruning Accuracy", fontsize=14, weight="bold", pad=12)
    output_path = Path(output_dir) / "gate_taylor_global_accuracy_table.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_layer_heatmap(layers, output_dir, dpi):
    frame = layers.copy()
    frame["row_label"] = frame["dataset_label"] + " " + frame["ratio_label"]
    order = [
        f"{dataset_label(dataset)} {int(round(ratio * 100))}%"
        for dataset in DATASETS
        for ratio in RATIOS
    ]
    table = frame.pivot(index="row_label", columns="layer_idx", values="layer_pruned_ratio")
    table = table.reindex(order).reindex(range(12), axis=1)

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    sns.heatmap(
        table,
        ax=ax,
        cmap="mako_r",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Layer pruning ratio"},
    )
    ax.set_title("Gate Taylor Global Pruning Allocation by Layer", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("")
    output_path = Path(output_dir) / "gate_taylor_global_layer_pruned_ratio_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dataset_layer_heatmaps(layers, output_dir, dpi):
    output_paths = []
    for dataset in DATASETS:
        frame = layers[layers["dataset"] == dataset].copy()
        order = [f"{int(round(ratio * 100))}%" for ratio in RATIOS]
        table = frame.pivot(index="ratio_label", columns="layer_idx", values="layer_pruned_ratio")
        table = table.reindex(order).reindex(range(12), axis=1)

        fig, ax = plt.subplots(figsize=(10.8, 2.75))
        sns.heatmap(
            table,
            ax=ax,
            cmap="mako_r",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Layer pruning ratio"},
        )
        ax.set_title(
            f"{dataset_label(dataset)} Global Pruning Allocation",
            fontsize=14,
            weight="bold",
            pad=12,
        )
        ax.set_xlabel("Transformer block")
        ax.set_ylabel("Global pruning ratio")
        output_path = Path(output_dir) / f"gate_taylor_global_layer_pruned_ratio_{dataset}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, layers = build_frames(Path(args.results_root))
    summary_csv = output_dir / "gate_taylor_global_summary.csv"
    layers_csv = output_dir / "gate_taylor_global_layer_pruned_ratio.csv"
    summary.to_csv(summary_csv, index=False)
    layers.to_csv(layers_csv, index=False)

    table_path = plot_accuracy_table(summary, output_dir, args.dpi)
    heatmap_path = plot_layer_heatmap(layers, output_dir, args.dpi)
    dataset_heatmap_paths = plot_dataset_layer_heatmaps(layers, output_dir, args.dpi)

    print(f"[GateTaylorGlobal] summary rows={len(summary)}, saved {summary_csv}")
    print(f"[GateTaylorGlobal] layer rows={len(layers)}, saved {layers_csv}")
    print(f"[GateTaylorGlobal] saved {table_path}")
    print(f"[GateTaylorGlobal] saved {heatmap_path}")
    for path in dataset_heatmap_paths:
        print(f"[GateTaylorGlobal] saved {path}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
