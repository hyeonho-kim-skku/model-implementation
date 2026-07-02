"""Build slide-friendly figures for head-gate Taylor global pruning.

Run:
  python analysis/plot_head_gate_taylor_meeting_assets.py
"""

from __future__ import annotations

import argparse
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
PRUNING_ONLY_RATIOS = tuple(round(0.1 * idx, 1) for idx in range(1, 10))
RECOVERY_RATIOS = (0.1, 0.2, 0.3, 0.4)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global-summary",
        default="figures/head_gate_taylor_global_pruning/summary.csv",
    )
    parser.add_argument(
        "--recovery-summary",
        default="figures/head_gate_taylor_recovery/summary.csv",
    )
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--output-dir", default="figures/head_gate_taylor_meeting")
    parser.add_argument("--dpi", type=int, default=240)
    return parser


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def is_ratio_in(values, targets):
    rounded = values.round(4)
    target_values = {round(float(target), 4) for target in targets}
    return rounded.isin(target_values)


def load_global_summary(path):
    frame = pd.read_csv(path)
    frame = frame[frame["dataset"].isin(DATASETS)].copy()
    frame = frame[is_ratio_in(frame["ratio"], PRUNING_ONLY_RATIOS)]
    expected = len(DATASETS) * len(PRUNING_ONLY_RATIOS)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} pruning-only rows, found {len(frame)}.")
    return frame


def load_recovery_summary(path):
    frame = pd.read_csv(path)
    frame = frame[frame["dataset"].isin(DATASETS)].copy()
    frame = frame[is_ratio_in(frame["ratio"], RECOVERY_RATIOS)]
    expected = len(DATASETS) * len(RECOVERY_RATIOS)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} recovery rows, found {len(frame)}.")
    return frame


def ratio_tag(ratio):
    return f"{int(round(float(ratio) * 100)):03d}"


def reset_recovery_log_candidates(logs_root, dataset, ratio):
    tag = ratio_tag(ratio)
    pattern = (
        f"taylor_pruned_lora_recovery_head_gate_taylor_reset_cls_global{tag}_*/"
        f"taylor_pruned_lora_recovery_{dataset}.log"
    )
    return sorted(Path(logs_root).glob(pattern), key=lambda path: path.stat().st_mtime)


def parse_reset_recovery_log(path):
    text = path.read_text(errors="ignore")
    reset_match = re.search(r"\[TIMMPrunedLoRA\] reset_classifier: (\w+)", text)
    if not reset_match or reset_match.group(1) != "True":
        raise ValueError(f"Expected reset_classifier=True in {path}.")

    matches = re.findall(
        r"\[Epoch\s+(\d+)\].*?Test Acc:\s*([0-9.]+)%.*?Best Acc:\s*([0-9.]+)",
        text,
    )
    if not matches:
        raise ValueError(f"Missing recovery metrics in {path}.")

    parsed = [(int(epoch), float(acc), float(best)) for epoch, acc, best in matches]
    last_epoch, final_acc, best_acc = parsed[-1]
    best_epoch = max(parsed, key=lambda item: item[2])[0]
    return {
        "reset_recovery_final_acc": final_acc,
        "reset_recovery_best_acc": best_acc,
        "reset_recovery_last_epoch": last_epoch,
        "reset_recovery_best_epoch": best_epoch,
        "reset_recovery_log": str(path),
    }


def load_reset_recovery_rows(logs_root):
    rows = []
    for dataset in DATASETS:
        for ratio in RECOVERY_RATIOS:
            candidates = reset_recovery_log_candidates(logs_root, dataset, ratio)
            if not candidates:
                raise FileNotFoundError(
                    f"Missing reset recovery log for dataset={dataset}, ratio={ratio_tag(ratio)}."
                )
            row = {"dataset": dataset, "ratio": float(ratio)}
            row.update(parse_reset_recovery_log(candidates[-1]))
            rows.append(row)
    return pd.DataFrame(rows)


def build_recovery_comparison(recovery_summary, reset_recovery):
    frame = recovery_summary.merge(
        reset_recovery,
        on=("dataset", "ratio"),
        how="left",
        validate="one_to_one",
    )
    if frame["reset_recovery_best_acc"].isna().any():
        missing = frame[frame["reset_recovery_best_acc"].isna()][["dataset", "ratio"]]
        raise ValueError(f"Missing reset recovery rows:\n{missing}")

    frame["reset_recovery_best_drop"] = frame["baseline_acc"] - frame["reset_recovery_best_acc"]
    frame["reset_vs_no_reset_gain"] = frame["reset_recovery_best_acc"] - frame["recovery_best_acc"]
    return frame


def setup_slide_grid():
    fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5), sharex=True)
    return fig, axes.flatten()


def style_axis(ax, title, ylabel):
    ax.set_title(title, weight="bold", fontsize=17, pad=7)
    ax.set_xlabel("Global head pruning ratio", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, axis="y", alpha=0.28)
    ax.tick_params(axis="both", labelsize=11)
    ax.tick_params(axis="x", labelbottom=True)


def plot_pruning_only_slide(global_summary, output_dir, dpi):
    fig, axes = setup_slide_grid()
    for ax, dataset in zip(axes, DATASETS):
        frame = global_summary[global_summary["dataset"] == dataset].sort_values("ratio")
        baseline = float(frame["baseline_acc"].iloc[0])
        ax.axhline(
            baseline,
            color="0.35",
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            label="Baseline",
        )
        ax.plot(
            frame["ratio"],
            frame["pruned_acc"],
            color=sns.color_palette()[0],
            marker="o",
            markersize=5.6,
            linewidth=2.4,
            label="Pruning only",
        )
        style_axis(ax, dataset_label(dataset), "Top-1 accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    fig.suptitle(
        "Global Head Pruning Only Sweep (0.1-0.9)",
        fontsize=23,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "Head pruning without recovery quickly degrades accuracy at high pruning ratios.",
        ha="center",
        fontsize=13,
        color="0.25",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.88))

    path = Path(output_dir) / "slide_pruning_only_sweep.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recovery_slide(recovery_summary, output_dir, dpi):
    fig, axes = setup_slide_grid()
    colors = sns.color_palette()
    for ax, dataset in zip(axes, DATASETS):
        frame = recovery_summary[recovery_summary["dataset"] == dataset].sort_values("ratio")
        baseline = float(frame["baseline_acc"].iloc[0])
        ax.axhline(
            baseline,
            color="0.35",
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            label="Baseline",
        )
        ax.plot(
            frame["ratio"],
            frame["pruning_only_acc"],
            color=colors[0],
            marker="o",
            markersize=6.0,
            linewidth=2.4,
            label="Pruning only",
        )
        ax.plot(
            frame["ratio"],
            frame["recovery_best_acc"],
            color=colors[1],
            marker="s",
            markersize=6.0,
            linewidth=2.4,
            label="Recovery (no reset)",
        )
        ax.plot(
            frame["ratio"],
            frame["reset_recovery_best_acc"],
            color=colors[2],
            marker="^",
            markersize=6.3,
            linewidth=2.4,
            label="Recovery (reset cls)",
        )
        ratio_030 = frame[frame["ratio"].round(4) == 0.3].iloc[0]
        ax.text(
            0.98,
            0.08,
            f"30% reset drop: {ratio_030['reset_recovery_best_drop']:.2f}pp",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="0.2",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.85"},
        )
        style_axis(ax, dataset_label(dataset), "Top-1 accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
        fontsize=13,
    )
    fig.suptitle(
        "Classifier Reset Improves Recovery after Global Head Pruning",
        fontsize=23,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "Resetting the classifier helps LoRA recovery adapt to the pruned representation.",
        ha="center",
        fontsize=13,
        color="0.25",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.88))

    path = Path(output_dir) / "slide_pruning_only_vs_recovery.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reset_gain_slide(recovery_summary, output_dir, dpi):
    fig, axes = setup_slide_grid()
    color = sns.color_palette()[2]
    max_gain = float(recovery_summary["reset_vs_no_reset_gain"].max())
    y_max = max(0.8, max_gain + 0.45)

    for ax, dataset in zip(axes, DATASETS):
        frame = recovery_summary[recovery_summary["dataset"] == dataset].sort_values("ratio")
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.2, alpha=0.75)
        ax.plot(
            frame["ratio"],
            frame["reset_vs_no_reset_gain"],
            color=color,
            marker="^",
            markersize=7.0,
            linewidth=2.6,
        )
        for _, row in frame.iterrows():
            ax.text(
                row["ratio"],
                row["reset_vs_no_reset_gain"] + 0.08,
                f"+{row['reset_vs_no_reset_gain']:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.2,
                color="0.2",
            )
        ax.set_ylim(-0.08, y_max)
        ax.set_xticks(list(RECOVERY_RATIOS))
        style_axis(ax, dataset_label(dataset), "Reset gain over no-reset (%p)")

    fig.suptitle(
        "Classifier Reset Gain over No-Reset Recovery",
        fontsize=23,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "Classifier reset consistently improves recovery, with larger gains on fine-grained datasets.",
        ha="center",
        fontsize=13,
        color="0.25",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.91))

    path = Path(output_dir) / "slide_reset_gain_over_no_reset.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_recovery_drop_table(recovery_summary, output_dir):
    table = recovery_summary.pivot(
        index="dataset_label",
        columns="ratio",
        values="reset_recovery_best_drop",
    )
    table = table.reindex([dataset_label(dataset) for dataset in DATASETS])
    table.columns = [f"{int(round(float(col) * 100))}%" for col in table.columns]
    path = Path(output_dir) / "recovery_drop_table.csv"
    table.to_csv(path, float_format="%.2f")
    return path


def save_recovery_comparison_table(recovery_summary, output_dir):
    columns = [
        "dataset",
        "dataset_label",
        "ratio",
        "baseline_acc",
        "pruning_only_acc",
        "recovery_best_acc",
        "reset_recovery_best_acc",
        "pruning_only_drop",
        "recovery_best_drop",
        "reset_recovery_best_drop",
        "reset_vs_no_reset_gain",
        "recovery_log",
        "reset_recovery_log",
    ]
    path = Path(output_dir) / "recovery_reset_comparison.csv"
    recovery_summary[columns].to_csv(path, index=False, float_format="%.4f")
    return path


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    global_summary = load_global_summary(args.global_summary)
    recovery_summary = load_recovery_summary(args.recovery_summary)
    reset_recovery = load_reset_recovery_rows(args.logs_root)
    recovery_summary = build_recovery_comparison(recovery_summary, reset_recovery)
    pruning_path = plot_pruning_only_slide(global_summary, output_dir, args.dpi)
    recovery_path = plot_recovery_slide(recovery_summary, output_dir, args.dpi)
    reset_gain_path = plot_reset_gain_slide(recovery_summary, output_dir, args.dpi)
    table_path = save_recovery_drop_table(recovery_summary, output_dir)
    comparison_path = save_recovery_comparison_table(recovery_summary, output_dir)

    print(f"[Meeting] saved {pruning_path}")
    print(f"[Meeting] saved {recovery_path}")
    print(f"[Meeting] saved {reset_gain_path}")
    print(f"[Meeting] saved {table_path}")
    print(f"[Meeting] saved {comparison_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
