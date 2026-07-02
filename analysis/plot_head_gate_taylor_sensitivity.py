"""Plot head-gate Taylor attention-head sensitivity results.

Inputs are results.jsonl files from sensitivity_taylor.py with:
  importance=head_gate_taylor
  pruning_modules=head
  head_gate_taylor_reduction=sum_abs
  head_gate_taylor_aggregation=samplewise

The script writes three analysis artifacts:
  - <dataset>_head_gate_taylor_sensitivity_heatmap.png
  - all_datasets_head_gate_taylor_sensitivity_heatmaps.png
  - safe_budgets.csv
  - summary.csv

Safe budgets are conservative. For a threshold such as 1.0%p, the budget is the
largest pruned-head count k such that every tested count from 1..k stays within
that accuracy-drop threshold. This avoids treating noisy non-monotonic recovery
at larger k as a stable pruning budget.

Run:
  python analysis/plot_head_gate_taylor_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASETS = ("cifar100", "cub200", "fgvc_aircraft", "stanford_cars")
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}
DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="pruned")
    parser.add_argument(
        "--output-dir",
        default="figures/head_gate_taylor_sensitivity/sum_abs",
    )
    parser.add_argument(
        "--thresholds",
        default="0.5,1.0,2.0",
        help="Comma-separated accuracy-drop thresholds in percentage points.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def parse_thresholds(value):
    if value is None:
        return list(DEFAULT_THRESHOLDS)
    thresholds = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if any(threshold < 0.0 for threshold in thresholds):
        raise ValueError(f"Thresholds must be non-negative: {thresholds}")
    return thresholds


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def result_path(results_root, dataset):
    folder = (
        f"vit_base_{dataset}_lora50_"
        "head_gate_taylor_proj_in_sum_abs_samplewise_full_sensitivity"
    )
    return Path(results_root) / folder / "results.jsonl"


def load_jsonl(path):
    metadata = None
    trials = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "metadata":
            metadata = row
        elif row.get("type") == "trial":
            row["_line_no"] = line_no
            trials.append(row)

    if metadata is None:
        raise ValueError(f"Missing metadata row in {path}.")
    if not trials:
        raise ValueError(f"Missing trial rows in {path}.")
    return metadata, trials


def validate_metadata(path, metadata, dataset):
    config = metadata["config"]
    expected = {
        "dataset": dataset,
        "importance": "head_gate_taylor",
        "pruning_modules": "head",
        "head_gate_taylor_reduction": "sum_abs",
        "head_gate_taylor_aggregation": "samplewise",
    }
    mismatches = {
        key: {"expected": value, "found": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Metadata mismatch in {path}: {mismatches}")


def load_rows(path, dataset):
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    metadata, trials = load_jsonl(path)
    validate_metadata(path, metadata, dataset)

    config = metadata["config"]
    reference_acc = float(config["reference_baseline_metrics"]["acc"])
    calibration = config["calibration"]
    rows = []
    seen = set()
    for trial in trials:
        layer_idx = int(trial["layer_idx"])
        pruned_head_count = int(trial["pruned_head_count"])
        key = (layer_idx, pruned_head_count)
        if key in seen:
            raise ValueError(
                f"Duplicate trial in {path}: layer={layer_idx}, "
                f"pruned_head_count={pruned_head_count}"
            )
        seen.add(key)

        acc = float(trial["metrics"]["acc"])
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label(dataset),
                "layer_idx": layer_idx,
                "pruned_head_count": pruned_head_count,
                "remaining_head_count": int(trial["remaining_head_count"]),
                "acc": acc,
                "reference_acc": reference_acc,
                "acc_drop": reference_acc - acc,
                "calibration_split": calibration["split"],
                "calibration_processed_examples": calibration["processed_examples"],
                "results_path": str(path),
            }
        )

    expected = len(config["target_layers"]) * len(config["trials"])
    if len(seen) != expected:
        raise ValueError(f"Expected {expected} trials in {path}, found {len(seen)}.")
    return rows


def build_frame(results_root):
    rows = []
    for dataset in DATASETS:
        rows.extend(load_rows(result_path(results_root, dataset), dataset))
    return pd.DataFrame(rows)


def plot_dataset_heatmap(frame, dataset, output_dir, dpi):
    dataset_frame = frame[frame["dataset"] == dataset]
    table = dataset_frame.pivot(
        index="layer_idx",
        columns="pruned_head_count",
        values="acc_drop",
    )
    table = table.sort_index().reindex(sorted(table.columns), axis=1)
    vmin = min(0.0, float(dataset_frame["acc_drop"].min()))
    vmax = max(0.1, float(dataset_frame["acc_drop"].max()))

    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    sns.heatmap(
        table,
        ax=ax,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Accuracy drop (%p)"},
    )
    ax.set_title(
        f"{dataset_label(dataset)} Head Gate Taylor Sensitivity (sum_abs)",
        weight="bold",
        pad=12,
    )
    ax.set_xlabel("Pruned attention heads in one block")
    ax.set_ylabel("Transformer block")
    output_path = output_dir / f"{dataset}_head_gate_taylor_sensitivity_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_composite_heatmaps(frame, output_dir, dpi):
    """Plot all four dataset heatmaps in one slide-friendly figure.

    This composite version keeps compact cell annotations so the figure can be
    used as a standalone sensitivity slide.
    """
    vmin = min(0.0, float(frame["acc_drop"].min()))
    vmax = max(0.1, float(frame["acc_drop"].max()))

    fig, axes = plt.subplots(2, 2, figsize=(25.0, 14.2), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([0.925, 0.18, 0.014, 0.64])
    axes_flat = axes.ravel()

    for idx, dataset in enumerate(DATASETS):
        ax = axes_flat[idx]
        dataset_frame = frame[frame["dataset"] == dataset]
        table = dataset_frame.pivot(
            index="layer_idx",
            columns="pruned_head_count",
            values="acc_drop",
        )
        table = table.sort_index().reindex(sorted(table.columns), axis=1)
        sns.heatmap(
            table,
            ax=ax,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 13.0},
            linewidths=0.25,
            linecolor="white",
            cbar=idx == 0,
            cbar_ax=cbar_ax if idx == 0 else None,
            cbar_kws={"label": "Accuracy drop (%p)"},
        )
        ax.set_title(dataset_label(dataset), fontsize=20, weight="bold", pad=12)
        ax.set_xlabel("Pruned heads in one block" if idx >= 2 else "", fontsize=18)
        ax.set_ylabel("Transformer block" if idx % 2 == 0 else "", fontsize=18)
        ax.tick_params(axis="both", labelsize=15)

    fig.suptitle(
        "Layer-wise Attention Head Sensitivity (Head Gate Taylor, sum_abs)",
        fontsize=28,
        weight="bold",
        y=0.965,
    )
    cbar_ax.yaxis.label.set_size(18)
    cbar_ax.tick_params(labelsize=16)
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.91, hspace=0.25, wspace=0.12)
    output_path = output_dir / "all_datasets_head_gate_taylor_sensitivity_heatmaps.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def conservative_safe_budget(layer_frame, threshold):
    budget = 0
    for pruned_head_count in sorted(layer_frame["pruned_head_count"].unique()):
        row = layer_frame[layer_frame["pruned_head_count"] == pruned_head_count]
        if row.empty:
            break
        acc_drop = float(row.iloc[0]["acc_drop"])
        if acc_drop <= threshold:
            budget = int(pruned_head_count)
            continue
        break
    return budget


def build_safe_budgets(frame, thresholds):
    rows = []
    for (dataset, layer_idx), layer_frame in frame.groupby(["dataset", "layer_idx"]):
        layer_frame = layer_frame.sort_values("pruned_head_count")
        row = {
            "dataset": dataset,
            "dataset_label": dataset_label(dataset),
            "layer_idx": int(layer_idx),
            "mean_acc_drop": float(layer_frame["acc_drop"].mean()),
            "max_acc_drop": float(layer_frame["acc_drop"].max()),
        }
        for threshold in thresholds:
            label = f"budget_drop_le_{threshold:g}pp"
            row[label] = conservative_safe_budget(layer_frame, threshold)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "layer_idx"]).reset_index(drop=True)


def build_summary(frame, safe_budgets, thresholds):
    rows = []
    for dataset, dataset_frame in frame.groupby("dataset"):
        layer_mean = (
            dataset_frame.groupby("layer_idx")["acc_drop"]
            .mean()
            .sort_values(ascending=False)
        )
        row = {
            "dataset": dataset,
            "dataset_label": dataset_label(dataset),
            "reference_acc": float(dataset_frame["reference_acc"].iloc[0]),
            "num_trials": int(len(dataset_frame)),
            "mean_acc_drop": float(dataset_frame["acc_drop"].mean()),
            "median_acc_drop": float(dataset_frame["acc_drop"].median()),
            "max_acc_drop": float(dataset_frame["acc_drop"].max()),
            "min_acc_drop": float(dataset_frame["acc_drop"].min()),
            "most_sensitive_layer_by_mean_drop": int(layer_mean.index[0]),
            "least_sensitive_layer_by_mean_drop": int(layer_mean.index[-1]),
        }
        dataset_budgets = safe_budgets[safe_budgets["dataset"] == dataset]
        for threshold in thresholds:
            label = f"budget_drop_le_{threshold:g}pp"
            row[f"total_safe_heads_drop_le_{threshold:g}pp"] = int(dataset_budgets[label].sum())
            row[f"mean_safe_heads_drop_le_{threshold:g}pp"] = float(dataset_budgets[label].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def main(args):
    sns.set_theme(style="whitegrid", context="notebook")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)

    frame = build_frame(Path(args.results_root))
    safe_budgets = build_safe_budgets(frame, thresholds)
    summary = build_summary(frame, safe_budgets, thresholds)

    safe_budget_path = output_dir / "safe_budgets.csv"
    summary_path = output_dir / "summary.csv"
    safe_budgets.to_csv(safe_budget_path, index=False, float_format="%.4f")
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"[HeadGateTaylorSensitivity] rows={len(frame)}")
    print(f"[HeadGateTaylorSensitivity] saved {safe_budget_path}")
    print(f"[HeadGateTaylorSensitivity] saved {summary_path}")

    for dataset in DATASETS:
        output_path = plot_dataset_heatmap(frame, dataset, output_dir, args.dpi)
        print(f"[HeadGateTaylorSensitivity] saved {output_path}")

    output_path = plot_composite_heatmaps(frame, output_dir, args.dpi)
    print(f"[HeadGateTaylorSensitivity] saved {output_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
