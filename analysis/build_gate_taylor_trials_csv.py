"""Build a tidy CSV from gate-Taylor layer-sensitivity results.

The output table is the starting point for reduction and layer-sensitivity
analysis. It intentionally avoids plotting so the first analysis artifact is a
simple, inspectable CSV.

Run:
  python analysis/build_gate_taylor_trials_csv.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASETS = ("cifar100", "cub200", "fgvc_aircraft", "stanford_cars")
REDUCTIONS = ("sum_abs", "sum_square", "signed_damage")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="pruned",
        help="Root directory containing gate-Taylor sensitivity result folders.",
    )
    parser.add_argument(
        "--output-csv",
        default="figures/gate_taylor_sensitivity/gate_taylor_trials.csv",
        help="Path for the combined trial CSV.",
    )
    return parser


def result_path(results_root: Path, dataset: str, reduction: str) -> Path:
    folder = f"vit_base_{dataset}_lora50_gate_taylor_fc1_out_{reduction}_full_sensitivity"
    return results_root / folder / "results.jsonl"


def load_result(path: Path, dataset: str, reduction: str) -> tuple[dict, list[dict]]:
    metadata = None
    trials = []
    with path.open("r") as file:
        for line_no, line in enumerate(file, start=1):
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
    if config.get("gate_taylor_reduction") != reduction:
        raise ValueError(
            f"Reduction mismatch in {path}: expected {reduction}, "
            f"got {config.get('gate_taylor_reduction')}."
        )
    return metadata, trials


def rows_from_result(path: Path, dataset: str, reduction: str) -> list[dict]:
    metadata, trials = load_result(path, dataset, reduction)
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
            raise ValueError(f"Duplicate trial for {dataset}/{reduction}: layer={layer_idx}, ratio={ratio}.")
        seen.add(key)

        metrics = trial["metrics"]
        pruning_stats = trial.get("pruning_stats", {})
        rows.append(
            {
                "dataset": dataset,
                "reduction": reduction,
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
        raise ValueError(f"Expected 120 unique trials for {dataset}/{reduction}, found {len(seen)}.")
    return rows


def main(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    all_rows = []
    for dataset in DATASETS:
        for reduction in REDUCTIONS:
            path = result_path(results_root, dataset, reduction)
            if not path.exists():
                raise FileNotFoundError(f"Missing result file: {path}")
            all_rows.extend(rows_from_result(path, dataset, reduction))

    frame = pd.DataFrame(all_rows)
    frame = frame.sort_values(["dataset", "reduction", "layer_idx", "ratio"]).reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    print(f"[Build] wrote {len(frame)} rows to {output_csv}")
    print(f"[Build] datasets={frame['dataset'].nunique()}, reductions={frame['reduction'].nunique()}")


if __name__ == "__main__":
    main(build_parser().parse_args())
