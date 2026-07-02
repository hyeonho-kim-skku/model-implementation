"""Plot global head-gate Taylor pruning-only and recovery accuracy.

Run:
  python analysis/plot_head_gate_taylor_recovery.py
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
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}
DEFAULT_RATIOS = (0.1, 0.2, 0.3, 0.4)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="pruned")
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--output-dir", default="figures/head_gate_taylor_recovery")
    parser.add_argument(
        "--ratios",
        default=",".join(str(ratio) for ratio in DEFAULT_RATIOS),
        help="Comma-separated pruning ratios to include.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def parse_ratios(value):
    ratios = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not ratios:
        raise ValueError("At least one ratio is required.")
    return ratios


def ratio_tag(ratio):
    return f"{int(round(float(ratio) * 100)):03d}"


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


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
        raise ValueError(f"Missing trial row in {path}.")
    return metadata, trials


def pruning_result_path(results_root, dataset, ratio):
    tag = ratio_tag(ratio)
    folder = (
        f"vit_base_{dataset}_lora50_"
        f"head_gate_taylor_proj_in_sum_abs_samplewise_global{tag}"
    )
    return Path(results_root) / folder / "results.jsonl"


def load_pruning_row(results_root, dataset, ratio):
    path = pruning_result_path(results_root, dataset, ratio)
    if not path.exists():
        raise FileNotFoundError(f"Missing pruning result: {path}")

    metadata, trials = load_jsonl(path)
    config = metadata["config"]
    baseline = config["reference_baseline_metrics"]
    matches = [trial for trial in trials if abs(float(trial["ratio"]) - float(ratio)) < 1e-9]
    if len(matches) != 1:
        raise ValueError(f"Expected one ratio={ratio} trial in {path}, found {len(matches)}.")

    trial = matches[0]
    head_summary = trial["pruning_stats"]["target_pruning_summary"]["overall"]["head"]
    return {
        "dataset": dataset,
        "dataset_label": dataset_label(dataset),
        "ratio": float(ratio),
        "ratio_tag": ratio_tag(ratio),
        "baseline_acc": float(baseline["acc"]),
        "pruning_only_acc": float(trial["metrics"]["acc"]),
        "pruned_heads": int(head_summary["pruned_heads"]),
        "total_heads": int(head_summary["heads_before"]),
        "results_path": str(path),
    }


def recovery_log_candidates(logs_root, dataset, ratio):
    tag = ratio_tag(ratio)
    pattern = (
        f"taylor_pruned_lora_recovery_head_gate_taylor_global{tag}_*/"
        f"taylor_pruned_lora_recovery_{dataset}.log"
    )
    return sorted(Path(logs_root).glob(pattern), key=lambda path: path.stat().st_mtime)


def parse_recovery_log(path):
    text = path.read_text(errors="ignore")
    matches = re.findall(
        r"\[Epoch\s+(\d+)\].*?Test Acc:\s*([0-9.]+)%.*?Best Acc:\s*([0-9.]+)",
        text,
    )
    if not matches:
        raise ValueError(f"Missing recovery epoch metrics in {path}.")

    parsed = [(int(epoch), float(acc), float(best)) for epoch, acc, best in matches]
    last_epoch, final_acc, best_acc = parsed[-1]
    best_epoch = max(parsed, key=lambda item: item[2])[0]
    return {
        "recovery_final_acc": final_acc,
        "recovery_best_acc": best_acc,
        "recovery_last_epoch": last_epoch,
        "recovery_best_epoch": best_epoch,
        "recovery_log": str(path),
    }


def load_recovery_row(logs_root, dataset, ratio):
    candidates = recovery_log_candidates(logs_root, dataset, ratio)
    if not candidates:
        raise FileNotFoundError(
            f"Missing recovery log for dataset={dataset}, ratio={ratio_tag(ratio)}."
        )
    return parse_recovery_log(candidates[-1])


def build_frame(results_root, logs_root, ratios):
    rows = []
    for dataset in DATASETS:
        for ratio in ratios:
            row = load_pruning_row(results_root, dataset, ratio)
            row.update(load_recovery_row(logs_root, dataset, ratio))
            row["pruning_only_drop"] = row["baseline_acc"] - row["pruning_only_acc"]
            row["recovery_best_drop"] = row["baseline_acc"] - row["recovery_best_acc"]
            row["recovery_gain"] = row["recovery_best_acc"] - row["pruning_only_acc"]
            rows.append(row)
    return pd.DataFrame(rows)


def save_summary(frame, output_dir):
    path = Path(output_dir) / "summary.csv"
    frame.to_csv(path, index=False, float_format="%.4f")
    return path


def setup_grid_axes():
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), sharex=True)
    return fig, axes.flatten()


def plot_accuracy_grid(frame, output_dir, dpi):
    fig, axes = setup_grid_axes()
    for ax, dataset in zip(axes, DATASETS):
        dataset_frame = frame[frame["dataset"] == dataset].sort_values("ratio")
        baseline = dataset_frame["baseline_acc"].iloc[0]
        ax.axhline(
            baseline,
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label="Baseline",
        )
        ax.plot(
            dataset_frame["ratio"],
            dataset_frame["pruning_only_acc"],
            marker="o",
            linewidth=2.0,
            label="Pruning only",
        )
        ax.plot(
            dataset_frame["ratio"],
            dataset_frame["recovery_best_acc"],
            marker="s",
            linewidth=2.0,
            label="Recovery",
        )
        ax.set_title(dataset_label(dataset), weight="bold")
        ax.set_xlabel("Global head pruning ratio")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Global Head-Gate Taylor Pruning: Pruning-Only vs Recovery", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))

    path = Path(output_dir) / "accuracy_grid.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_drop_grid(frame, output_dir, dpi):
    fig, axes = setup_grid_axes()
    for ax, dataset in zip(axes, DATASETS):
        dataset_frame = frame[frame["dataset"] == dataset].sort_values("ratio")
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.2, alpha=0.75)
        ax.plot(
            dataset_frame["ratio"],
            dataset_frame["pruning_only_drop"],
            marker="o",
            linewidth=2.0,
            label="Pruning only",
        )
        ax.plot(
            dataset_frame["ratio"],
            dataset_frame["recovery_best_drop"],
            marker="s",
            linewidth=2.0,
            label="Recovery",
        )
        ax.set_title(dataset_label(dataset), weight="bold")
        ax.set_xlabel("Global head pruning ratio")
        ax.set_ylabel("Accuracy drop (%p)")
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Accuracy Drop from Baseline after Head Pruning", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))

    path = Path(output_dir) / "drop_grid.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary(frame):
    for dataset in DATASETS:
        dataset_frame = frame[frame["dataset"] == dataset].sort_values("ratio")
        baseline = dataset_frame["baseline_acc"].iloc[0]
        best_ratio = dataset_frame.loc[dataset_frame["recovery_best_drop"].idxmin()]
        high_ratio = dataset_frame.iloc[-1]
        print(
            f"[Plot] {dataset_label(dataset)}: baseline={baseline:.2f}, "
            f"best_recovery_drop={best_ratio['recovery_best_drop']:.2f} "
            f"at ratio={best_ratio['ratio']:.1f}, "
            f"ratio={high_ratio['ratio']:.1f} recovery={high_ratio['recovery_best_acc']:.2f} "
            f"drop={high_ratio['recovery_best_drop']:.2f}"
        )


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    ratios = parse_ratios(args.ratios)
    frame = build_frame(args.results_root, args.logs_root, ratios)
    summary_path = save_summary(frame, output_dir)
    accuracy_path = plot_accuracy_grid(frame, output_dir, args.dpi)
    drop_path = plot_drop_grid(frame, output_dir, args.dpi)
    print_summary(frame)
    print(f"[Plot] saved {summary_path}")
    print(f"[Plot] saved {accuracy_path}")
    print(f"[Plot] saved {drop_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
