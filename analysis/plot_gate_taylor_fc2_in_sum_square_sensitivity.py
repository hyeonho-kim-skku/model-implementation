"""Plot fc2_in sum_square gate-Taylor sensitivity heatmaps.

Run:
  python analysis/plot_gate_taylor_fc2_in_sum_square_sensitivity.py
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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="pruned")
    parser.add_argument("--output-dir", default="figures/gate_taylor_sensitivity/fc2_in_sum_square")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def result_path(results_root, dataset):
    folder = f"vit_base_{dataset}_lora50_gate_taylor_fc2_in_sum_square_full_sensitivity"
    return Path(results_root) / folder / "results.jsonl"


def load_rows(path, dataset):
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

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
        raise ValueError(f"Missing metadata row in {path}")
    config = metadata["config"]
    if config["dataset"] != dataset:
        raise ValueError(f"Dataset mismatch in {path}: {config['dataset']}")
    if config["gate_taylor_location"] != "fc2_in":
        raise ValueError(f"Location mismatch in {path}: {config['gate_taylor_location']}")
    if config["gate_taylor_reduction"] != "sum_square":
        raise ValueError(f"Reduction mismatch in {path}: {config['gate_taylor_reduction']}")

    reference_acc = float(config["reference_baseline_metrics"]["acc"])
    calibration = config["calibration"]
    rows = []
    seen = set()
    for trial in trials:
        layer_idx = int(trial["layer_idx"])
        ratio = float(trial["ratio"])
        key = (layer_idx, ratio)
        if key in seen:
            raise ValueError(f"Duplicate trial in {path}: layer={layer_idx}, ratio={ratio}")
        seen.add(key)
        acc = float(trial["metrics"]["acc"])
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label(dataset),
                "layer_idx": layer_idx,
                "ratio": ratio,
                "acc": acc,
                "reference_acc": reference_acc,
                "acc_drop": reference_acc - acc,
                "calibration_split": calibration["split"],
                "calibration_processed_examples": calibration["processed_examples"],
                "results_path": str(path),
            }
        )

    if len(seen) != 120:
        raise ValueError(f"Expected 120 trials in {path}, found {len(seen)}")
    return rows


def build_frame(results_root):
    rows = []
    for dataset in DATASETS:
        rows.extend(load_rows(result_path(results_root, dataset), dataset))
    return pd.DataFrame(rows)


def pivot_heatmap(frame):
    table = frame.pivot(index="layer_idx", columns="ratio", values="acc_drop")
    return table.sort_index().reindex(sorted(table.columns), axis=1)


def plot_dataset(frame, dataset, output_dir, dpi):
    dataset_frame = frame[frame["dataset"] == dataset]
    table = pivot_heatmap(dataset_frame)
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
    ax.set_title(f"{dataset_label(dataset)} fc2_in Gate Taylor Sensitivity (sum_square)", weight="bold", pad=12)
    ax.set_xlabel("Layer-wise pruning ratio")
    ax.set_ylabel("Transformer block")
    output_path = Path(output_dir) / f"{dataset}_fc2_in_sum_square_sensitivity_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(args):
    sns.set_theme(style="whitegrid", context="notebook")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_frame(Path(args.results_root))
    csv_path = output_dir / "gate_taylor_fc2_in_sum_square_sensitivity.csv"
    frame.to_csv(csv_path, index=False)
    print(f"[FC2InSensitivity] rows={len(frame)}, saved {csv_path}")
    for dataset in DATASETS:
        output_path = plot_dataset(frame, dataset, output_dir, args.dpi)
        print(f"[FC2InSensitivity] saved {output_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
