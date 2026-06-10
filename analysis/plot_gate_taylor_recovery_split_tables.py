"""Create split recovery tables for samplewise-50 and tokenwise-60 comparisons.

Run:
  python analysis/plot_gate_taylor_recovery_split_tables.py
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
    parser.add_argument("--output-dir", default="figures/gate_taylor_aggregation_global")
    parser.add_argument("--preferred-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def ratio_tag(ratio):
    return f"global{int(round(ratio * 100)):03d}"


def ratio_artifact_tag(ratio):
    return f"ratio{int(round(ratio * 100)):03d}"


def ratio_label(ratio):
    return f"{int(round(ratio * 100))}%"


def result_dir(dataset, ratio, aggregation):
    suffix = "" if aggregation == "elementwise" else f"_{aggregation}"
    return (
        f"vit_base_{dataset}_lora50_gate_taylor_"
        f"fc2_in_sum_square{suffix}_{ratio_tag(ratio)}"
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
    return {
        "baseline_acc": float(metadata["config"]["reference_baseline_metrics"]["acc"]),
        "pruned_acc": float(trial["metrics"]["acc"]),
        "artifact_path": trial["artifact_path"],
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
                "seed": args.get("seed"),
                "run_dir": str(args_path.parent),
            }
        )
    if not candidates:
        return None
    seeded = [row for row in candidates if row["seed"] == preferred_seed]
    if seeded:
        return max(seeded, key=lambda row: row["recovered_acc"])
    return max(candidates, key=lambda row: row["recovered_acc"])


def build_comparison(results_root, runs_root, ratio, method, preferred_seed):
    rows = []
    for dataset in DATASETS:
        base = load_pruning_result(results_root, dataset, ratio, "elementwise")
        other = load_pruning_result(results_root, dataset, ratio, method)
        base_recovery = load_best_recovery(runs_root, dataset, ratio, "elementwise", preferred_seed)
        other_recovery = load_best_recovery(runs_root, dataset, ratio, method, preferred_seed)
        if base_recovery is None or other_recovery is None:
            raise FileNotFoundError(f"Missing recovery for {dataset} {ratio_label(ratio)} {method}.")
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label(dataset),
                "ratio": ratio,
                "ratio_label": ratio_label(ratio),
                "method": method,
                "baseline_acc": base["baseline_acc"],
                "elementwise_pruned_acc": base["pruned_acc"],
                f"{method}_pruned_acc": other["pruned_acc"],
                "pruned_delta": other["pruned_acc"] - base["pruned_acc"],
                "elementwise_recovered_acc": base_recovery["recovered_acc"],
                f"{method}_recovered_acc": other_recovery["recovered_acc"],
                "recovery_delta": other_recovery["recovered_acc"] - base_recovery["recovered_acc"],
                "elementwise_best_epoch": base_recovery["best_epoch"],
                f"{method}_best_epoch": other_recovery["best_epoch"],
                "elementwise_run_dir": base_recovery["run_dir"],
                f"{method}_run_dir": other_recovery["run_dir"],
            }
        )
    return pd.DataFrame(rows)


def fmt_acc(value):
    return f"{float(value):.2f}"


def fmt_delta(value):
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{float(value):.2f}"


def plot_table(frame, method, title, output_path, dpi):
    method_label = method.capitalize()
    table_frame = pd.DataFrame(
        {
            "Dataset": frame["dataset_label"],
            "Dense": frame["baseline_acc"].map(fmt_acc),
            "Elem. P": frame["elementwise_pruned_acc"].map(fmt_acc),
            f"{method_label} P": frame[f"{method}_pruned_acc"].map(fmt_acc),
            "P Delta": frame["pruned_delta"].map(fmt_delta),
            "Elem. R": frame["elementwise_recovered_acc"].map(fmt_acc),
            f"{method_label} R": frame[f"{method}_recovered_acc"].map(fmt_acc),
            "R Delta": frame["recovery_delta"].map(fmt_delta),
        }
    )

    fig, ax = plt.subplots(figsize=(12.4, 2.95))
    ax.axis("off")
    table = ax.table(
        cellText=table_frame.values,
        colLabels=table_frame.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1.0, 1.55)

    delta_cols = {4: "pruned_delta", 7: "recovery_delta"}
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        if row_idx == 0:
            cell.set_facecolor("#344054")
            cell.set_text_props(color="white", weight="bold")
            continue
        cell.set_facecolor("#FFFFFF" if row_idx % 2 else "#F9FAFB")
        if col_idx in delta_cols:
            value = float(frame.iloc[row_idx - 1][delta_cols[col_idx]])
            if value > 0.05:
                cell.set_facecolor("#E7F6EC")
                cell.set_text_props(color="#027A48", weight="bold")
            elif value < -0.05:
                cell.set_facecolor("#FDECEC")
                cell.set_text_props(color="#B42318")

    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)
    runs_root = Path(args.runs_root)

    samplewise50 = build_comparison(results_root, runs_root, 0.5, "samplewise", args.preferred_seed)
    tokenwise60 = build_comparison(results_root, runs_root, 0.6, "tokenwise", args.preferred_seed)

    samplewise_csv = output_dir / "gate_taylor_samplewise50_recovery_comparison.csv"
    tokenwise_csv = output_dir / "gate_taylor_tokenwise60_recovery_comparison.csv"
    samplewise50.to_csv(samplewise_csv, index=False)
    tokenwise60.to_csv(tokenwise_csv, index=False)

    samplewise_png = plot_table(
        samplewise50,
        "samplewise",
        "Seed-Controlled 50% Recovery: Elementwise vs Samplewise",
        output_dir / "gate_taylor_samplewise50_recovery_comparison_table.png",
        args.dpi,
    )
    tokenwise_png = plot_table(
        tokenwise60,
        "tokenwise",
        "Seed-Controlled 60% Recovery: Elementwise vs Tokenwise",
        output_dir / "gate_taylor_tokenwise60_recovery_comparison_table.png",
        args.dpi,
    )

    print(f"[GateTaylorRecoverySplit] saved {samplewise_csv}")
    print(f"[GateTaylorRecoverySplit] saved {samplewise_png}")
    print(f"[GateTaylorRecoverySplit] saved {tokenwise_csv}")
    print(f"[GateTaylorRecoverySplit] saved {tokenwise_png}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
