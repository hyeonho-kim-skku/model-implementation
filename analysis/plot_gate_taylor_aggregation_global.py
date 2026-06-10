"""Plot gate-Taylor aggregation global-pruning accuracy comparisons.

Run:
  python analysis/plot_gate_taylor_aggregation_global.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = ("cifar100", "cub200", "fgvc_aircraft", "stanford_cars")
RATIOS = (0.4, 0.5, 0.6)
AGGREGATIONS = ("elementwise", "samplewise", "tokenwise", "channelwise")
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
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


def result_path(results_root, dataset, ratio, aggregation):
    aggregation_suffix = "" if aggregation == "elementwise" else f"_{aggregation}"
    folder = (
        f"vit_base_{dataset}_lora50_gate_taylor_"
        f"fc2_in_sum_square{aggregation_suffix}_{ratio_tag(ratio)}"
    )
    return Path(results_root) / folder / "results.jsonl"


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = [row for row in rows if row.get("type") == "metadata"]
    trials = [row for row in rows if row.get("type") == "trial"]
    if len(metadata) != 1 or len(trials) != 1:
        raise ValueError(f"Expected one metadata row and one trial row in {path}.")
    return metadata[0], trials[0]


def build_summary(results_root):
    rows = []
    for dataset in DATASETS:
        for ratio in RATIOS:
            baseline_acc = None
            acc_by_aggregation = {}
            for aggregation in AGGREGATIONS:
                path = result_path(results_root, dataset, ratio, aggregation)
                metadata, trial = load_jsonl(path)
                config = metadata["config"]
                if config["dataset"] != dataset:
                    raise ValueError(f"Dataset mismatch in {path}: {config['dataset']}")
                if abs(float(trial["ratio"]) - ratio) > 1e-12:
                    raise ValueError(f"Ratio mismatch in {path}: {trial['ratio']}")

                baseline_acc = float(config["reference_baseline_metrics"]["acc"])
                acc_by_aggregation[aggregation] = float(trial["metrics"]["acc"])

            elementwise_acc = acc_by_aggregation["elementwise"]
            row = {
                "dataset": dataset,
                "dataset_label": dataset_label(dataset),
                "ratio": ratio,
                "ratio_label": f"{int(round(ratio * 100))}%",
                "baseline_acc": baseline_acc,
            }
            for aggregation in AGGREGATIONS:
                acc = acc_by_aggregation[aggregation]
                row[f"{aggregation}_acc"] = acc
                row[f"{aggregation}_delta"] = acc - elementwise_acc
            rows.append(row)
    return pd.DataFrame(rows)


def fmt_acc(value):
    return f"{value:.2f}"


def fmt_acc_delta(acc, delta):
    sign = "+" if delta >= 0 else ""
    return f"{acc:.2f}\n({sign}{delta:.2f})"


def table_frame(summary):
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Dataset": row["dataset_label"],
                "Ratio": row["ratio_label"],
                "Elementwise": fmt_acc(row["elementwise_acc"]),
                "Samplewise": fmt_acc_delta(row["samplewise_acc"], row["samplewise_delta"]),
                "Tokenwise": fmt_acc_delta(row["tokenwise_acc"], row["tokenwise_delta"]),
                "Channelwise": fmt_acc_delta(row["channelwise_acc"], row["channelwise_delta"]),
            }
        )
    return pd.DataFrame(rows)


def style_table(table, summary):
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(1.0, 1.62)

    delta_columns = {
        3: "samplewise_delta",
        4: "tokenwise_delta",
        5: "channelwise_delta",
    }
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        if row_idx == 0:
            cell.set_facecolor("#344054")
            cell.set_text_props(color="white", weight="bold")
            continue

        base_color = "#FFFFFF" if row_idx % 2 else "#F9FAFB"
        cell.set_facecolor(base_color)
        if col_idx in delta_columns:
            delta = float(summary.iloc[row_idx - 1][delta_columns[col_idx]])
            if delta > 0.05:
                cell.set_facecolor("#E7F6EC")
                cell.set_text_props(color="#027A48", weight="bold")
            elif delta < -0.05:
                cell.set_facecolor("#FDECEC")
                cell.set_text_props(color="#B42318")


def plot_accuracy_delta_table(summary, output_dir, dpi):
    frame = table_frame(summary)
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        colLabels=frame.columns,
        loc="center",
        cellLoc="center",
    )
    style_table(table, summary)
    ax.set_title(
        "Gate Taylor Aggregation Ablation: Pruning-Only Accuracy",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    ax.text(
        0.5,
        -0.05,
        "New aggregation cells show accuracy with delta vs elementwise in parentheses.",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9.8,
        color="#475467",
    )
    output_path = Path(output_dir) / "gate_taylor_aggregation_accuracy_delta_table.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(Path(args.results_root))
    csv_path = output_dir / "gate_taylor_aggregation_accuracy_delta.csv"
    summary.to_csv(csv_path, index=False)
    table_path = plot_accuracy_delta_table(summary, output_dir, args.dpi)

    print(f"[GateTaylorAggregation] rows={len(summary)}, saved {csv_path}")
    print(f"[GateTaylorAggregation] saved {table_path}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
