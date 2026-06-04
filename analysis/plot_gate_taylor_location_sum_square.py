"""Compare fc1_out and fc2_in gate-Taylor sum_square sensitivity.

This analysis keeps the question narrow: does moving the element-wise gate from
the pre-GELU fc1 output to the post-GELU fc2 input change layer-wise sensitivity?

Run:
  python analysis/plot_gate_taylor_location_sum_square.py
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
LOCATIONS = ("fc1_out", "fc2_in")
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="pruned",
        help="Root directory containing gate-Taylor sensitivity result folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/gate_taylor_sensitivity/location_sum_square_comparison",
        help="Directory for comparison CSVs and heatmaps.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Saved figure DPI.")
    return parser


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def result_path(results_root: Path, dataset: str, location: str) -> Path:
    folder = f"vit_base_{dataset}_lora50_gate_taylor_{location}_sum_square_full_sensitivity"
    return results_root / folder / "results.jsonl"


def load_result(path: Path, dataset: str, location: str) -> tuple[dict, list[dict]]:
    metadata = None
    trials = []
    with path.open("r") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "metadata":
                metadata = row
            elif row.get("type") == "trial":
                row["_line_no"] = line_no
                trials.append(row)

    if metadata is None:
        raise ValueError(f"No metadata row found in {path}.")
    if not trials:
        raise ValueError(f"No trial rows found in {path}.")

    config = metadata["config"]
    if config.get("dataset") != dataset:
        raise ValueError(f"Dataset mismatch in {path}: expected {dataset}, got {config.get('dataset')}.")
    if config.get("gate_taylor_location") != location:
        raise ValueError(
            f"Location mismatch in {path}: expected {location}, "
            f"got {config.get('gate_taylor_location')}."
        )
    if config.get("gate_taylor_reduction") != "sum_square":
        raise ValueError(
            f"Reduction mismatch in {path}: expected sum_square, "
            f"got {config.get('gate_taylor_reduction')}."
        )
    return metadata, trials


def rows_from_result(path: Path, dataset: str, location: str) -> list[dict]:
    metadata, trials = load_result(path, dataset, location)
    config = metadata["config"]
    calibration = config["calibration"]
    reference = config["reference_baseline_metrics"]

    seen = set()
    rows = []
    for trial in trials:
        layer_idx = int(trial["layer_idx"])
        ratio = float(trial["ratio"])
        key = (layer_idx, ratio)
        if key in seen:
            raise ValueError(f"Duplicate trial for {dataset}/{location}: layer={layer_idx}, ratio={ratio}.")
        seen.add(key)

        metrics = trial["metrics"]
        pruning_stats = trial.get("pruning_stats", {})
        rows.append(
            {
                "dataset": dataset,
                "location": location,
                "reduction": "sum_square",
                "layer_idx": layer_idx,
                "ratio": ratio,
                "acc": float(metrics["acc"]),
                "loss": float(metrics["loss"]),
                "reference_acc": float(reference["acc"]),
                "reference_loss": float(reference["loss"]),
                "acc_drop": float(reference["acc"]) - float(metrics["acc"]),
                "loss_increase": float(metrics["loss"]) - float(reference["loss"]),
                "calibration_requested_batches": calibration.get("requested_batches"),
                "calibration_processed_examples": calibration.get("processed_examples"),
                "gate_taylor_score_mode": calibration.get("gate_taylor_score_mode"),
                "trial_index": trial.get("trial_index"),
                "total_trials": trial.get("total_trials"),
                "base_macs": pruning_stats.get("base_macs"),
                "pruned_macs": pruning_stats.get("pruned_macs"),
                "base_params": pruning_stats.get("base_params"),
                "pruned_params": pruning_stats.get("pruned_params"),
                "results_path": str(path),
            }
        )

    if len(seen) != 120:
        raise ValueError(f"Expected 120 unique trials for {dataset}/{location}, found {len(seen)}.")
    return rows


def pivot_heatmap(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    table = frame.pivot(index="layer_idx", columns="ratio", values=value)
    return table.sort_index().reindex(sorted(table.columns), axis=1)


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        dataset_frame = frame[frame["dataset"] == dataset]
        summary = {"dataset": dataset}
        for location in LOCATIONS:
            loc_frame = dataset_frame[dataset_frame["location"] == location]
            nonzero = loc_frame[loc_frame["ratio"] > 0]
            worst = nonzero.loc[nonzero["acc_drop"].idxmax()]
            summary[f"{location}_mean_drop"] = nonzero["acc_drop"].mean()
            summary[f"{location}_median_drop"] = nonzero["acc_drop"].median()
            summary[f"{location}_max_drop"] = nonzero["acc_drop"].max()
            summary[f"{location}_worst_layer"] = int(worst["layer_idx"])
            summary[f"{location}_worst_ratio"] = float(worst["ratio"])
        summary["mean_drop_diff_fc2_in_minus_fc1_out"] = (
            summary["fc2_in_mean_drop"] - summary["fc1_out_mean_drop"]
        )
        summary["max_drop_diff_fc2_in_minus_fc1_out"] = (
            summary["fc2_in_max_drop"] - summary["fc1_out_max_drop"]
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def build_pairwise_winrate(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(
        index=["dataset", "layer_idx", "ratio"],
        columns="location",
        values="acc_drop",
    ).reset_index()
    wide = wide[wide["ratio"] > 0].copy()
    wide["diff_fc2_minus_fc1"] = wide["fc2_in"] - wide["fc1_out"]
    tie_tolerance = 1e-12
    wide["tie"] = wide["diff_fc2_minus_fc1"].abs() < tie_tolerance
    wide["fc2_win"] = wide["diff_fc2_minus_fc1"] < -tie_tolerance
    wide["fc1_win"] = wide["diff_fc2_minus_fc1"] > tie_tolerance

    rows = []
    for dataset in DATASETS:
        dataset_frame = wide[wide["dataset"] == dataset]
        rows.append(_pairwise_row(dataset, dataset_frame))
    rows.append(_pairwise_row("overall", wide))
    return pd.DataFrame(rows)


def _pairwise_row(dataset: str, frame: pd.DataFrame) -> dict:
    return {
        "dataset": dataset,
        "n_pairs": len(frame),
        "mean_diff_fc2_minus_fc1": frame["diff_fc2_minus_fc1"].mean(),
        "fc2_win_rate": frame["fc2_win"].mean(),
        "fc2_wins": int(frame["fc2_win"].sum()),
        "fc1_wins": int(frame["fc1_win"].sum()),
        "ties": int(frame["tie"].sum()),
    }


def plot_dataset(dataset_frame: pd.DataFrame, dataset: str, output_dir: Path, dpi: int) -> Path:
    fc1 = dataset_frame[dataset_frame["location"] == "fc1_out"]
    fc2 = dataset_frame[dataset_frame["location"] == "fc2_in"]
    merged = fc1.merge(
        fc2,
        on=["dataset", "layer_idx", "ratio"],
        suffixes=("_fc1_out", "_fc2_in"),
        validate="one_to_one",
    )
    merged["acc_drop_diff"] = merged["acc_drop_fc2_in"] - merged["acc_drop_fc1_out"]

    vmin = min(0.0, float(dataset_frame["acc_drop"].min()))
    vmax = max(0.1, float(dataset_frame["acc_drop"].max()))
    diff_abs = max(0.1, float(merged["acc_drop_diff"].abs().max()))

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8), sharey=True)
    panels = (
        ("fc1_out", pivot_heatmap(fc1, "acc_drop"), "magma", vmin, vmax, "Accuracy drop (%p)"),
        ("fc2_in", pivot_heatmap(fc2, "acc_drop"), "magma", vmin, vmax, "Accuracy drop (%p)"),
        (
            "fc2_in - fc1_out",
            pivot_heatmap(merged, "acc_drop_diff"),
            "coolwarm",
            -diff_abs,
            diff_abs,
            "Drop difference (%p)",
        ),
    )
    for ax, (title, table, cmap, panel_vmin, panel_vmax, cbar_label) in zip(axes, panels):
        sns.heatmap(
            table,
            ax=ax,
            cmap=cmap,
            vmin=panel_vmin,
            vmax=panel_vmax,
            center=0 if title == "fc2_in - fc1_out" else None,
            annot=True,
            fmt=".2f",
            linewidths=0.35,
            cbar=(ax is axes[-1]),
            cbar_kws={"label": cbar_label},
        )
        ax.set_title(title)
        ax.set_xlabel("Pruning ratio")
        ax.set_ylabel("Transformer block" if ax is axes[0] else "")

    fig.suptitle(f"{dataset_label(dataset)} Gate Location Sensitivity (sum_square)", y=1.02)
    fig.tight_layout()
    output_path = output_dir / f"{dataset}_fc1_out_vs_fc2_in_sum_square_heatmaps.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pairwise_table(pairwise: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    display = pairwise.copy()
    display["dataset"] = display["dataset"].map(
        {
            "cifar100": "CIFAR100",
            "cub200": "CUB200",
            "fgvc_aircraft": "FGVC-Aircraft",
            "stanford_cars": "Stanford Cars",
            "overall": "Overall",
        }
    )
    display["Mean Diff\n(fc2_in - fc1_out)"] = display["mean_diff_fc2_minus_fc1"].map(
        lambda value: f"{value:.4f}"
    )
    display["fc2_in\nWin Rate"] = display["fc2_win_rate"].map(lambda value: f"{value * 100:.2f}%")
    display["W / L / T"] = display.apply(
        lambda row: f"{int(row['fc2_wins'])} / {int(row['fc1_wins'])} / {int(row['ties'])}",
        axis=1,
    )
    display = display[
        [
            "dataset",
            "Mean Diff\n(fc2_in - fc1_out)",
            "fc2_in\nWin Rate",
            "W / L / T",
        ]
    ]
    display = display.rename(columns={"dataset": "Dataset"})

    fig, ax = plt.subplots(figsize=(8.8, 2.6))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#F2F4F7")
            cell.set_text_props(weight="bold", color="#111827")
        elif display.iloc[row - 1]["Dataset"] == "Overall":
            cell.set_facecolor("#EEF4FF")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF")

    ax.set_title(
        "Gate Location Comparison (sum_square)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    fig.text(
        0.5,
        0.02,
        "Mean Diff is computed over nonzero pruning ratios. Negative values indicate lower accuracy drop for fc2_in.",
        ha="center",
        fontsize=8.5,
        color="#475467",
    )

    output_path = output_dir / "gate_taylor_location_sum_square_pairwise_winrate_table.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_mean_drop_table(summary: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    display = summary.copy()
    overall = {
        "dataset": "overall",
        "fc1_out_mean_drop": summary["fc1_out_mean_drop"].mean(),
        "fc2_in_mean_drop": summary["fc2_in_mean_drop"].mean(),
        "mean_drop_diff_fc2_in_minus_fc1_out": summary[
            "mean_drop_diff_fc2_in_minus_fc1_out"
        ].mean(),
    }
    display = pd.concat([display, pd.DataFrame([overall])], ignore_index=True)
    display["dataset"] = display["dataset"].map(
        {
            "cifar100": "CIFAR100",
            "cub200": "CUB200",
            "fgvc_aircraft": "FGVC-Aircraft",
            "stanford_cars": "Stanford Cars",
            "overall": "Overall",
        }
    )
    display["fc1_out\nMean Drop"] = display["fc1_out_mean_drop"].map(lambda value: f"{value:.3f}")
    display["fc2_in\nMean Drop"] = display["fc2_in_mean_drop"].map(lambda value: f"{value:.3f}")
    display["Diff\n(fc2_in - fc1_out)"] = display[
        "mean_drop_diff_fc2_in_minus_fc1_out"
    ].map(lambda value: f"{value:.3f}")
    display = display[
        [
            "dataset",
            "fc1_out\nMean Drop",
            "fc2_in\nMean Drop",
            "Diff\n(fc2_in - fc1_out)",
        ]
    ].rename(columns={"dataset": "Dataset"})

    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#F2F4F7")
            cell.set_text_props(weight="bold", color="#111827")
        elif display.iloc[row - 1]["Dataset"] == "Overall":
            cell.set_facecolor("#EEF4FF")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF")

    ax.set_title(
        "Gate Location Mean Accuracy Drop (sum_square)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    fig.text(
        0.5,
        0.02,
        "Values are averaged over 12 layers and 9 nonzero pruning ratios. Negative diff favors fc2_in.",
        ha="center",
        fontsize=8.5,
        color="#475467",
    )

    output_path = output_dir / "gate_taylor_location_sum_square_mean_drop_table.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main(args: argparse.Namespace) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset in DATASETS:
        for location in LOCATIONS:
            path = result_path(results_root, dataset, location)
            if not path.exists():
                raise FileNotFoundError(f"Missing result file: {path}")
            all_rows.extend(rows_from_result(path, dataset, location))

    frame = pd.DataFrame(all_rows)
    frame = frame.sort_values(["dataset", "location", "layer_idx", "ratio"]).reset_index(drop=True)
    trials_csv = output_dir / "gate_taylor_location_sum_square_trials.csv"
    frame.to_csv(trials_csv, index=False)
    print(f"[Build] wrote {len(frame)} rows to {trials_csv}")

    summary = build_summary(frame)
    summary_csv = output_dir / "gate_taylor_location_sum_square_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[Build] wrote summary to {summary_csv}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    mean_drop_table_path = plot_mean_drop_table(summary, output_dir, args.dpi)
    print(f"[Plot] saved {mean_drop_table_path}")

    pairwise = build_pairwise_winrate(frame)
    pairwise_csv = output_dir / "gate_taylor_location_sum_square_pairwise_winrate.csv"
    pairwise.to_csv(pairwise_csv, index=False)
    print(f"[Build] wrote pairwise win-rate to {pairwise_csv}")
    print(pairwise.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    pairwise_table_path = plot_pairwise_table(pairwise, output_dir, args.dpi)
    print(f"[Plot] saved {pairwise_table_path}")

    for dataset in DATASETS:
        dataset_frame = frame[frame["dataset"] == dataset]
        output_path = plot_dataset(dataset_frame, dataset, output_dir, args.dpi)
        print(f"[Plot] saved {output_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
