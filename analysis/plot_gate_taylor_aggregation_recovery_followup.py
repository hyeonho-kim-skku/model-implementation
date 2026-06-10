"""Summarize selected recovery follow-ups for gate-Taylor aggregation ablations.

Run:
  python analysis/plot_gate_taylor_aggregation_recovery_followup.py
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


DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}

AGGREGATION_LABELS = {
    "elementwise": "Elementwise",
    "samplewise": "Samplewise",
    "tokenwise": "Tokenwise",
}

SELECTED_FOLLOWUPS = (
    ("cifar100", 0.5, "samplewise", "Samplewise 50%"),
    ("cub200", 0.5, "samplewise", "Samplewise 50%"),
    ("fgvc_aircraft", 0.5, "samplewise", "Samplewise 50%"),
    ("stanford_cars", 0.5, "samplewise", "Samplewise 50%"),
    ("cub200", 0.6, "tokenwise", "Tokenwise selected"),
    ("fgvc_aircraft", 0.5, "tokenwise", "Tokenwise selected"),
    ("stanford_cars", 0.6, "tokenwise", "Tokenwise selected"),
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="pruned")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="figures/gate_taylor_aggregation_global")
    parser.add_argument("--preferred-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def aggregation_label(aggregation):
    return AGGREGATION_LABELS.get(aggregation, aggregation)


def ratio_tag(ratio):
    return f"global{int(round(ratio * 100)):03d}"


def ratio_artifact_tag(ratio):
    return f"ratio{int(round(ratio * 100)):03d}"


def ratio_label(ratio):
    return f"{int(round(ratio * 100))}%"


def result_dir(dataset, ratio, aggregation):
    aggregation_suffix = "" if aggregation == "elementwise" else f"_{aggregation}"
    return (
        f"vit_base_{dataset}_lora50_gate_taylor_"
        f"fc2_in_sum_square{aggregation_suffix}_{ratio_tag(ratio)}"
    )


def result_path(results_root, dataset, ratio, aggregation):
    return Path(results_root) / result_dir(dataset, ratio, aggregation) / "results.jsonl"


def artifact_path(dataset, ratio, aggregation):
    return (
        f"./pruned/{result_dir(dataset, ratio, aggregation)}/"
        f"artifacts/{ratio_artifact_tag(ratio)}/pruned_timm_classifier.pth"
    )


def load_pruning_result(results_root, dataset, ratio, aggregation):
    path = result_path(results_root, dataset, ratio, aggregation)
    if not path.exists():
        raise FileNotFoundError(f"Missing pruning result file: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = next(row for row in rows if row.get("type") == "metadata")
    trial = next(row for row in rows if row.get("type") == "trial")

    config = metadata["config"]
    if config["dataset"] != dataset:
        raise ValueError(f"Dataset mismatch in {path}: {config['dataset']}")
    if abs(float(trial["ratio"]) - ratio) > 1e-12:
        raise ValueError(f"Ratio mismatch in {path}: {trial['ratio']}")
    actual_aggregation = trial["pruning_config"].get("gate_taylor_aggregation", "elementwise")
    if actual_aggregation != aggregation:
        raise ValueError(f"Aggregation mismatch in {path}: {actual_aggregation}")

    return {
        "baseline_acc": float(config["reference_baseline_metrics"]["acc"]),
        "pruned_acc": float(trial["metrics"]["acc"]),
        "artifact_path": trial["artifact_path"],
        "results_path": str(path),
    }


def load_best_recovery(runs_root, dataset, ratio, aggregation, preferred_seed):
    expected_artifact = artifact_path(dataset, ratio, aggregation)
    run_root = Path(runs_root) / f"timm_pruned_lora_{dataset}_supervised"
    candidates = []
    for args_path in run_root.glob("*/args.json"):
        args = json.loads(args_path.read_text())
        if args.get("artifact_path") != expected_artifact:
            continue
        ckpt_path = args_path.parent / "best_cls_ckpt.pth"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        candidates.append(
            {
                "recovered_acc": float(ckpt["acc"]),
                "best_epoch": int(ckpt["epoch"]),
                "run_dir": str(args_path.parent),
                "checkpoint_path": str(ckpt_path),
                "seed": args.get("seed"),
            }
        )
    if not candidates:
        return None
    if preferred_seed is not None:
        seeded_candidates = [
            row for row in candidates if row["seed"] == preferred_seed
        ]
        if seeded_candidates:
            return max(seeded_candidates, key=lambda row: row["recovered_acc"])
    return max(candidates, key=lambda row: row["recovered_acc"])


def build_summary(results_root, runs_root, preferred_seed):
    rows = []
    for dataset, ratio, aggregation, group in SELECTED_FOLLOWUPS:
        selected_pruning = load_pruning_result(results_root, dataset, ratio, aggregation)
        elementwise_pruning = load_pruning_result(results_root, dataset, ratio, "elementwise")
        selected_recovery = load_best_recovery(
            runs_root, dataset, ratio, aggregation, preferred_seed
        )
        elementwise_recovery = load_best_recovery(
            runs_root, dataset, ratio, "elementwise", preferred_seed
        )
        if selected_recovery is None:
            raise FileNotFoundError(
                f"Missing selected recovery run for {dataset} {ratio_label(ratio)} {aggregation}."
            )

        baseline_acc = selected_pruning["baseline_acc"]
        elementwise_recovered_acc = None
        elementwise_recovery_epoch = None
        elementwise_recovery_run_dir = None
        if elementwise_recovery is not None:
            elementwise_recovered_acc = elementwise_recovery["recovered_acc"]
            elementwise_recovery_epoch = elementwise_recovery["best_epoch"]
            elementwise_recovery_run_dir = elementwise_recovery["run_dir"]

        row = {
            "group": group,
            "dataset": dataset,
            "dataset_label": dataset_label(dataset),
            "ratio": ratio,
            "ratio_label": ratio_label(ratio),
            "aggregation": aggregation,
            "aggregation_label": aggregation_label(aggregation),
            "baseline_acc": baseline_acc,
            "selected_pruned_acc": selected_pruning["pruned_acc"],
            "elementwise_pruned_acc": elementwise_pruning["pruned_acc"],
            "selected_pruned_delta_vs_elementwise": (
                selected_pruning["pruned_acc"] - elementwise_pruning["pruned_acc"]
            ),
            "selected_recovered_acc": selected_recovery["recovered_acc"],
            "selected_recovered_drop": baseline_acc - selected_recovery["recovered_acc"],
            "selected_recovery_gain": (
                selected_recovery["recovered_acc"] - selected_pruning["pruned_acc"]
            ),
            "selected_best_epoch": selected_recovery["best_epoch"],
            "selected_seed": selected_recovery["seed"],
            "selected_run_dir": selected_recovery["run_dir"],
            "selected_artifact_path": selected_pruning["artifact_path"],
            "elementwise_recovered_acc": elementwise_recovered_acc,
            "elementwise_recovered_drop": (
                None
                if elementwise_recovered_acc is None
                else baseline_acc - elementwise_recovered_acc
            ),
            "elementwise_recovery_delta": (
                None
                if elementwise_recovered_acc is None
                else selected_recovery["recovered_acc"] - elementwise_recovered_acc
            ),
            "elementwise_recovery_epoch": elementwise_recovery_epoch,
            "elementwise_recovery_run_dir": elementwise_recovery_run_dir,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_acc(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def fmt_delta(value):
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{float(value):.2f}"


def table_frame(summary):
    return pd.DataFrame(
        {
            "Dataset": summary["dataset_label"],
            "Ratio": summary["ratio_label"],
            "Agg.": summary["aggregation_label"],
            "Dense": summary["baseline_acc"].map(fmt_acc),
            "Pruned": summary["selected_pruned_acc"].map(fmt_acc),
            "P Δ vs Elem.": summary["selected_pruned_delta_vs_elementwise"].map(fmt_delta),
            "Recovered": summary["selected_recovered_acc"].map(fmt_acc),
            "Elem. Rec.": summary["elementwise_recovered_acc"].map(fmt_acc),
            "R Δ vs Elem.": summary["elementwise_recovery_delta"].map(fmt_delta),
            "Drop": summary["selected_recovered_drop"].map(fmt_acc),
        }
    )


def style_table(table, summary):
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.58)

    delta_columns = {
        5: "selected_pruned_delta_vs_elementwise",
        8: "elementwise_recovery_delta",
    }
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        if row_idx == 0:
            cell.set_facecolor("#344054")
            cell.set_text_props(color="white", weight="bold")
            continue

        cell.set_facecolor("#FFFFFF" if row_idx % 2 else "#F9FAFB")
        if col_idx in delta_columns:
            value = summary.iloc[row_idx - 1][delta_columns[col_idx]]
            if value is None or pd.isna(value):
                continue
            value = float(value)
            if value > 0.05:
                cell.set_facecolor("#E7F6EC")
                cell.set_text_props(color="#027A48", weight="bold")
            elif value < -0.05:
                cell.set_facecolor("#FDECEC")
                cell.set_text_props(color="#B42318")


def plot_recovery_table(summary, output_dir, dpi):
    frame = table_frame(summary)
    fig, ax = plt.subplots(figsize=(13.7, 4.25))
    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        colLabels=frame.columns,
        loc="center",
        cellLoc="center",
    )
    style_table(table, summary)
    ax.set_title(
        "Selected LoRA Recovery Follow-up for Promising Aggregation Settings",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    ax.text(
        0.5,
        -0.065,
        "Rows are selected from non-elementwise settings with promising pruning-only accuracy. "
        "Delta columns compare against elementwise at the same dataset and pruning ratio.",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9.6,
        color="#475467",
    )
    output_path = Path(output_dir) / "gate_taylor_aggregation_recovery_followup_table.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        Path(args.results_root), Path(args.runs_root), args.preferred_seed
    )
    csv_path = output_dir / "gate_taylor_aggregation_recovery_followup.csv"
    summary.to_csv(csv_path, index=False)
    table_path = plot_recovery_table(summary, output_dir, args.dpi)

    print(f"[GateTaylorAggregationRecovery] rows={len(summary)}, saved {csv_path}")
    print(f"[GateTaylorAggregationRecovery] saved {table_path}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
