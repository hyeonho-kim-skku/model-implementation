"""Summarize gate-Taylor global pruning and LoRA recovery results.

Run:
  python analysis/plot_gate_taylor_global_recovery.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch


DATASETS = ("cifar100", "cub200", "fgvc_aircraft", "stanford_cars")
PRUNING_RATIOS = (0.4, 0.5, 0.6)
RECOVERY_RATIOS = (0.5, 0.6)
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="pruned")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="figures/gate_taylor_global_pruning")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def ratio_tag(ratio):
    return f"global{int(round(ratio * 100)):03d}"


def ratio_dir(dataset, ratio):
    return (
        f"vit_base_{dataset}_lora50_gate_taylor_"
        f"fc2_in_sum_square_{ratio_tag(ratio)}"
    )


def result_path(results_root, dataset, ratio):
    return Path(results_root) / ratio_dir(dataset, ratio) / "results.jsonl"


def artifact_path(dataset, ratio):
    tag = int(round(ratio * 100))
    return (
        f"./pruned/{ratio_dir(dataset, ratio)}/"
        f"artifacts/ratio{tag:03d}/pruned_timm_classifier.pth"
    )


def load_pruning_result(path, expected_dataset, expected_ratio):
    if not path.exists():
        raise FileNotFoundError(f"Missing pruning result file: {path}")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = next(row for row in rows if row.get("type") == "metadata")
    trial = next(row for row in rows if row.get("type") == "trial")

    config = metadata["config"]
    if config["dataset"] != expected_dataset:
        raise ValueError(f"Dataset mismatch in {path}: {config['dataset']}")
    if abs(float(trial["ratio"]) - expected_ratio) > 1e-12:
        raise ValueError(f"Ratio mismatch in {path}: {trial['ratio']}")

    baseline_acc = float(config["reference_baseline_metrics"]["acc"])
    pruned_acc = float(trial["metrics"]["acc"])
    return baseline_acc, pruned_acc


def load_best_recovery_checkpoint(runs_root, dataset, ratio):
    expected_artifact = artifact_path(dataset, ratio)
    run_root = Path(runs_root) / f"timm_pruned_lora_{dataset}_supervised"
    candidates = []

    for ckpt_path in run_root.glob("*/best_cls_ckpt.pth"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        if not isinstance(args, dict):
            continue
        if args.get("artifact_path") != expected_artifact:
            continue
        candidates.append(
            {
                "acc": float(ckpt["acc"]),
                "epoch": int(ckpt["epoch"]),
                "run_dir": str(ckpt_path.parent),
                "checkpoint_path": str(ckpt_path),
            }
        )

    if not candidates:
        return None
    return max(candidates, key=lambda row: row["acc"])


def build_summary(results_root, runs_root):
    rows = []
    for dataset in DATASETS:
        row = {"dataset": dataset, "dataset_label": dataset_label(dataset)}
        baseline_acc = None

        for ratio in PRUNING_RATIOS:
            baseline_acc, pruned_acc = load_pruning_result(
                result_path(results_root, dataset, ratio), dataset, ratio
            )
            tag = int(round(ratio * 100))
            row[f"pruned_{tag}_acc"] = pruned_acc
            row[f"pruned_{tag}_drop"] = baseline_acc - pruned_acc

        row["baseline_acc"] = baseline_acc

        for ratio in RECOVERY_RATIOS:
            tag = int(round(ratio * 100))
            recovery = load_best_recovery_checkpoint(runs_root, dataset, ratio)
            if recovery is None:
                row[f"recovered_{tag}_acc"] = None
                row[f"recovered_{tag}_drop"] = None
                row[f"recovered_{tag}_gain"] = None
                row[f"recovered_{tag}_best_epoch"] = None
                row[f"recovered_{tag}_run_dir"] = None
                continue
            row[f"recovered_{tag}_acc"] = recovery["acc"]
            row[f"recovered_{tag}_drop"] = baseline_acc - recovery["acc"]
            row[f"recovered_{tag}_gain"] = recovery["acc"] - row[f"pruned_{tag}_acc"]
            row[f"recovered_{tag}_best_epoch"] = recovery["epoch"]
            row[f"recovered_{tag}_run_dir"] = recovery["run_dir"]

        rows.append(row)
    return pd.DataFrame(rows)


def fmt(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def plot_summary_table(summary, output_dir, dpi):
    table_frame = pd.DataFrame(
        {
            "Dataset": summary["dataset_label"],
            "Dense": summary["baseline_acc"].map(fmt),
            "40% P": summary["pruned_40_acc"].map(fmt),
            "50% P": summary["pruned_50_acc"].map(fmt),
            "50% R": summary["recovered_50_acc"].map(fmt),
            "60% P": summary["pruned_60_acc"].map(fmt),
            "60% R": summary["recovered_60_acc"].map(fmt),
            "60% R Drop": summary["recovered_60_drop"].map(fmt),
        }
    )

    fig, ax = plt.subplots(figsize=(11.7, 2.85))
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

    ax.set_title(
        "Gate Taylor Global MLP Pruning and LoRA Recovery",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    output_path = Path(output_dir) / "gate_taylor_global_pruning_recovery_table.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _style_table(table, font_size=10.5, scale_y=1.55):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        if row_idx == 0:
            cell.set_facecolor("#344054")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row_idx % 2 else "#F9FAFB")


def plot_pruning_only_table(summary, output_dir, dpi):
    table_frame = pd.DataFrame(
        {
            "Dataset": summary["dataset_label"],
            "Dense": summary["baseline_acc"].map(fmt),
            "40% P": summary["pruned_40_acc"].map(fmt),
            "50% P": summary["pruned_50_acc"].map(fmt),
            "60% P": summary["pruned_60_acc"].map(fmt),
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 2.55))
    ax.axis("off")
    table = ax.table(
        cellText=table_frame.values,
        colLabels=table_frame.columns,
        loc="center",
        cellLoc="center",
    )
    _style_table(table)
    ax.set_title("Gate Taylor Global MLP Pruning-Only Accuracy", fontsize=14, weight="bold", pad=12)
    output_path = Path(output_dir) / "gate_taylor_global_pruning_only_table.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_recovery_only_table(summary, output_dir, dpi):
    table_frame = pd.DataFrame(
        {
            "Dataset": summary["dataset_label"],
            "Dense": summary["baseline_acc"].map(fmt),
            "50% R": summary["recovered_50_acc"].map(fmt),
            "50% R Drop": summary["recovered_50_drop"].map(fmt),
            "60% R": summary["recovered_60_acc"].map(fmt),
            "60% R Drop": summary["recovered_60_drop"].map(fmt),
        }
    )

    fig, ax = plt.subplots(figsize=(9.8, 2.55))
    ax.axis("off")
    table = ax.table(
        cellText=table_frame.values,
        colLabels=table_frame.columns,
        loc="center",
        cellLoc="center",
    )
    _style_table(table)
    ax.set_title("Gate Taylor Global MLP Pruning with LoRA Recovery", fontsize=14, weight="bold", pad=12)
    output_path = Path(output_dir) / "gate_taylor_global_recovery_only_table.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(Path(args.results_root), Path(args.runs_root))
    csv_path = output_dir / "gate_taylor_global_pruning_recovery_summary.csv"
    summary.to_csv(csv_path, index=False)
    table_path = plot_summary_table(summary, output_dir, args.dpi)
    pruning_table_path = plot_pruning_only_table(summary, output_dir, args.dpi)
    recovery_table_path = plot_recovery_only_table(summary, output_dir, args.dpi)

    print(f"[GateTaylorGlobalRecovery] summary rows={len(summary)}, saved {csv_path}")
    print(f"[GateTaylorGlobalRecovery] saved {table_path}")
    print(f"[GateTaylorGlobalRecovery] saved {pruning_table_path}")
    print(f"[GateTaylorGlobalRecovery] saved {recovery_table_path}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
