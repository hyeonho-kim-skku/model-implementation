"""Create recovery comparison tables for the completed pruning pipelines.

Run:
  python analysis/plot_progressive_pruning_recovery_tables.py
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
RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}
METHODS = (
    ("adapted_ce_oneshot", "Adapted CE One-shot"),
    ("ce_progressive", "CE-Guided Progressive"),
    ("prototype_progressive", "Prototype-Guided Progressive"),
)
EXTENDED_METHODS = (
    ("adapted_ce_oneshot", "Adapted CE One-shot"),
    ("ce_progressive", "CE-Guided Progressive"),
    ("prototype_progressive", "Prototype-Guided Progressive"),
    ("adapted_ce_progressive", "Adapted CE Progressive"),
)
INTERMEDIATE_RECOVERY_METHODS = (
    ("adapted_ce_oneshot", "Adapted CE One-shot"),
    ("ce_progressive", "CE-Guided Progressive"),
    ("prototype_progressive", "Prototype-Guided Progressive"),
    ("adapted_ce_progressive", "Adapted CE Progressive"),
    ("adapted_ce_prune_recover", "Adapted CE Progressive + 1ep Recovery"),
)
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="figures/progressive_pruning")
    parser.add_argument("--preferred-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only save the combined all-dataset figure and long CSV files.",
    )
    return parser


def ratio_tag(ratio):
    return f"{int(round(float(ratio) * 100)):03d}"


def ratio_label(ratio):
    return f"{int(round(float(ratio) * 100))}%"


def expected_artifact(method, dataset, ratio):
    tag = ratio_tag(ratio)
    if method == "adapted_ce_oneshot":
        return (
            f"./pruned/vit_base_{dataset}_lora50_gate_taylor_"
            f"fc2_in_sum_square_samplewise_global{tag}/"
            f"artifacts/ratio{tag}/pruned_timm_classifier.pth"
        )
    if method == "adapted_ce_progressive":
        return (
            f"./pruned/progressive_adapted_ce_{dataset}/"
            f"target{tag}/pruned_timm_classifier.pth"
        )
    if method == "adapted_ce_prune_recover":
        return (
            f"./pruned/progressive_adapted_ce_prune_recover_{dataset}/"
            f"target{tag}/pruned_timm_classifier.pth"
        )
    if method == "ce_progressive":
        return (
            f"./pruned/progressive_baseline_{dataset}/"
            f"target{tag}/pruned_timm_classifier.pth"
        )
    if method == "prototype_progressive":
        return (
            f"./pruned/progressive_prototype_{dataset}/"
            f"target{tag}/pruned_timm_classifier.pth"
        )
    raise ValueError(f"Unknown method: {method}")


def load_recovery(runs_root, method, dataset, ratio, preferred_seed):
    artifact = expected_artifact(method, dataset, ratio)
    run_root = Path(runs_root) / f"timm_pruned_lora_{dataset}_supervised"
    candidates = []

    for args_path in run_root.glob("*/args.json"):
        args = json.loads(args_path.read_text())
        if args.get("artifact_path") != artifact:
            continue
        if args.get("reset_classifier") is not True:
            continue
        if preferred_seed is not None and args.get("seed") != preferred_seed:
            continue

        checkpoint_path = args_path.parent / "best_cls_ckpt.pth"
        if not checkpoint_path.exists():
            continue
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        candidates.append(
            {
                "accuracy": float(checkpoint["acc"]),
                "best_epoch": int(checkpoint["epoch"]),
                "seed": args.get("seed"),
                "run_dir": str(args_path.parent),
                "checkpoint_path": str(checkpoint_path),
                "artifact_path": artifact,
            }
        )

    if not candidates:
        raise FileNotFoundError(
            f"Missing reset-classifier recovery for {method}, {dataset}, "
            f"{ratio_label(ratio)}: {artifact}"
        )

    # Match the existing analysis convention: if a run was retried under the
    # same controlled setup, retain the best completed checkpoint.
    return max(candidates, key=lambda row: row["accuracy"])


def build_results(runs_root, preferred_seed, methods=METHODS):
    records = []
    for dataset in DATASETS:
        for method, method_label in methods:
            for ratio in RATIOS:
                recovery = load_recovery(
                    runs_root,
                    method,
                    dataset,
                    ratio,
                    preferred_seed,
                )
                records.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "method": method,
                        "method_label": method_label,
                        "ratio": ratio,
                        "ratio_label": ratio_label(ratio),
                        "recovery_accuracy": recovery["accuracy"],
                        "best_epoch": recovery["best_epoch"],
                        "seed": recovery["seed"],
                        "artifact_path": recovery["artifact_path"],
                        "run_dir": recovery["run_dir"],
                        "checkpoint_path": recovery["checkpoint_path"],
                    }
                )
    return pd.DataFrame(records)


def dataset_table_frame(results, dataset, methods=METHODS):
    subset = results[results["dataset"] == dataset]
    rows = []
    for method, method_label in methods:
        method_rows = subset[subset["method"] == method].set_index("ratio")
        values = [float(method_rows.loc[ratio, "recovery_accuracy"]) for ratio in RATIOS]
        row = {
            "Method": method_label,
            **{ratio_label(ratio): value for ratio, value in zip(RATIOS, values)},
            "Avg.": sum(values) / len(values),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _format_table_values(frame):
    formatted = frame.copy()
    for column in frame.columns[1:]:
        formatted[column] = frame[column].map(lambda value: f"{float(value):.2f}")
    return formatted


def _draw_table(ax, frame, dataset, methods=METHODS, compact=False):
    formatted = _format_table_values(frame)
    ax.axis("off")
    if len(methods) > 4:
        column_widths = [0.34] + [0.094] * (len(formatted.columns) - 1)
    elif len(methods) > 3:
        column_widths = [0.31] + [0.0985] * (len(formatted.columns) - 1)
    else:
        column_widths = [0.25] + [0.107] * (len(formatted.columns) - 1)
    table = ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=column_widths,
    )
    table.auto_set_font_size(False)
    if compact and len(methods) > 4:
        table.set_fontsize(7.0)
    elif compact and len(methods) > 3:
        table.set_fontsize(8.0)
    else:
        table.set_fontsize(8.5 if compact else 10.2)
    table.scale(1.0, 1.34 if compact and len(methods) > 4 else 1.45 if compact else 1.65)

    numeric_columns = list(frame.columns[1:])
    column_maxima = {
        column: float(frame[column].max())
        for column in numeric_columns
    }

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#344054")
            cell.set_text_props(color="white", weight="bold")
            continue

        cell.set_facecolor("white")
        if col_idx == 0:
            cell.set_text_props(weight="bold", color="#1D2939", ha="left")
        else:
            column = frame.columns[col_idx]
            value = float(frame.iloc[row_idx - 1][column])
            if abs(value - column_maxima[column]) < 1e-9:
                cell.set_text_props(weight="bold", color="#101828")
                cell.set_facecolor("#FFF2CC")
            if column == "Avg.":
                cell.set_edgecolor("#98A2B3")
                cell.set_linewidth(1.4)

    ax.set_title(
        DATASET_LABELS[dataset],
        fontsize=12 if compact else 14,
        weight="bold",
        pad=8,
        color="#101828",
    )
    return table


def plot_dataset_table(frame, dataset, output_path, dpi, methods=METHODS):
    height = 3.05 if len(methods) > 3 else 2.75
    fig, ax = plt.subplots(figsize=(12.8, height))
    _draw_table(ax, frame, dataset, methods=methods, compact=False)
    fig.text(
        0.5,
        0.04,
        "One-shot: task-adapted source  |  Progressive: pruning before backbone adaptation",
        ha="center",
        fontsize=8.8,
        color="#475467",
    )
    fig.tight_layout(rect=(0.01, 0.10, 0.99, 0.98))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_combined_tables(frames, output_path, dpi, methods=METHODS):
    height = 6.25 if len(methods) > 4 else 6.0 if len(methods) > 3 else 5.4
    fig, axes = plt.subplots(2, 2, figsize=(15.2, height))
    for ax, dataset in zip(axes.flat, DATASETS):
        _draw_table(ax, frames[dataset], dataset, methods=methods, compact=True)

    fig.suptitle(
        "Recovery Accuracy by Pruning Ratio",
        fontsize=16,
        weight="bold",
        color="#101828",
        y=0.97,
    )
    fig.text(
        0.5,
        0.025,
        "All methods: global MLP pruning, fc2_in, sum_square, samplewise, "
        "classifier reset, 20-epoch LoRA recovery, seed 42",
        ha="center",
        fontsize=9,
        color="#475467",
    )
    fig.text(
        0.5,
        0.003,
        "Adapted methods use a task-adapted source; CE/Prototype progressive methods prune before backbone adaptation.",
        ha="center",
        fontsize=8.5,
        color="#667085",
    )
    fig.tight_layout(rect=(0.015, 0.09, 0.985, 0.91), h_pad=0.2, w_pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = build_results(Path(args.runs_root), args.preferred_seed)
    results_path = output_dir / "recovery_comparison_long.csv"
    results.to_csv(results_path, index=False)

    frames = {}
    for dataset in DATASETS:
        frame = dataset_table_frame(results, dataset)
        frames[dataset] = frame
        if not args.combined_only:
            frame.to_csv(
                output_dir / f"recovery_comparison_{dataset}.csv",
                index=False,
                float_format="%.2f",
            )
            plot_dataset_table(
                frame,
                dataset,
                output_dir / f"recovery_comparison_{dataset}.png",
                args.dpi,
            )

    combined_path = output_dir / "recovery_comparison_all.png"
    plot_combined_tables(frames, combined_path, args.dpi)

    extended_results = build_results(
        Path(args.runs_root),
        args.preferred_seed,
        methods=EXTENDED_METHODS,
    )
    extended_results_path = (
        output_dir / "recovery_comparison_with_adapted_progressive_long.csv"
    )
    extended_results.to_csv(extended_results_path, index=False)

    extended_frames = {}
    for dataset in DATASETS:
        frame = dataset_table_frame(
            extended_results,
            dataset,
            methods=EXTENDED_METHODS,
        )
        extended_frames[dataset] = frame
        if not args.combined_only:
            frame.to_csv(
                output_dir
                / f"recovery_comparison_with_adapted_progressive_{dataset}.csv",
                index=False,
                float_format="%.2f",
            )
            plot_dataset_table(
                frame,
                dataset,
                output_dir
                / f"recovery_comparison_with_adapted_progressive_{dataset}.png",
                args.dpi,
                methods=EXTENDED_METHODS,
            )

    extended_combined_path = (
        output_dir / "recovery_comparison_with_adapted_progressive_all.png"
    )
    plot_combined_tables(
        extended_frames,
        extended_combined_path,
        args.dpi,
        methods=EXTENDED_METHODS,
    )

    intermediate_results = build_results(
        Path(args.runs_root),
        args.preferred_seed,
        methods=INTERMEDIATE_RECOVERY_METHODS,
    )
    intermediate_results_path = (
        output_dir / "recovery_comparison_with_intermediate_recovery_long.csv"
    )
    intermediate_results.to_csv(intermediate_results_path, index=False)

    intermediate_frames = {}
    for dataset in DATASETS:
        frame = dataset_table_frame(
            intermediate_results,
            dataset,
            methods=INTERMEDIATE_RECOVERY_METHODS,
        )
        intermediate_frames[dataset] = frame
        if not args.combined_only:
            frame.to_csv(
                output_dir
                / f"recovery_comparison_with_intermediate_recovery_{dataset}.csv",
                index=False,
                float_format="%.2f",
            )
            plot_dataset_table(
                frame,
                dataset,
                output_dir
                / f"recovery_comparison_with_intermediate_recovery_{dataset}.png",
                args.dpi,
                methods=INTERMEDIATE_RECOVERY_METHODS,
            )

    intermediate_combined_path = (
        output_dir / "recovery_comparison_with_intermediate_recovery_all.png"
    )
    plot_combined_tables(
        intermediate_frames,
        intermediate_combined_path,
        args.dpi,
        methods=INTERMEDIATE_RECOVERY_METHODS,
    )

    print(f"[ProgressiveRecoveryTables] saved {results_path}")
    if not args.combined_only:
        for dataset in DATASETS:
            print(
                "[ProgressiveRecoveryTables] saved "
                f"{output_dir / f'recovery_comparison_{dataset}.png'}"
            )
    print(f"[ProgressiveRecoveryTables] saved {combined_path}")
    print(f"[ProgressiveRecoveryTables] saved {extended_results_path}")
    print(f"[ProgressiveRecoveryTables] saved {extended_combined_path}")
    print(f"[ProgressiveRecoveryTables] saved {intermediate_results_path}")
    print(f"[ProgressiveRecoveryTables] saved {intermediate_combined_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
