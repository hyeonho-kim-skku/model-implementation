"""Build a CSV comparison of 40% head-pruned VPT and reset-LoRA recovery."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DATASETS = ("cifar100", "cub200", "fgvc_aircraft", "stanford_cars")
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}
MODES = ("shallow", "deep")
EPOCH_PATTERN = re.compile(
    r"\[Epoch (?P<epoch>\d+)\].*Test Acc: (?P<acc>[0-9.]+)%, "
    r"Best Acc: (?P<best>[0-9.]+)"
)
PARAM_PATTERN = re.compile(r"\[TIMMPrunedVPT\] trainable params: ([0-9,]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Recovery log directory; defaults to the newest matching directory.",
    )
    parser.add_argument(
        "--reference-csv",
        default="figures/head_gate_taylor_meeting/recovery_reset_comparison.csv",
    )
    parser.add_argument(
        "--output",
        default="figures/head_pruned_vpt_recovery/summary.csv",
    )
    parser.add_argument("--num-prompt-tokens", type=int, default=1)
    parser.add_argument(
        "--prompt-modes",
        default=",".join(MODES),
        help="Comma-separated prompt modes to summarize (shallow, deep).",
    )
    return parser.parse_args()


def newest_log_dir():
    candidates = sorted(Path("logs").glob("head_pruned_vpt_recovery_*"))
    if not candidates:
        raise FileNotFoundError("No head_pruned_vpt_recovery log directory was found.")
    return candidates[-1]


def load_reference(path):
    rows = {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            if abs(float(row["ratio"]) - 0.4) > 1e-8:
                continue
            label = row["dataset_label"]
            rows[label] = {
                "baseline_acc": float(row["baseline_acc"]),
                "pruning_only_acc": float(row["pruning_only_acc"]),
                "reset_lora_best_acc": float(row["reset_recovery_best_acc"]),
            }
    return rows


def parse_log(path):
    text = path.read_text()
    epochs = [
        {
            "epoch": int(match.group("epoch")),
            "acc": float(match.group("acc")),
            "best": float(match.group("best")),
        }
        for match in EPOCH_PATTERN.finditer(text)
    ]
    if not epochs:
        raise ValueError(f"No completed evaluation epoch found in {path}.")
    parameter_match = PARAM_PATTERN.search(text)
    if parameter_match is None:
        raise ValueError(f"No trainable parameter count found in {path}.")
    return {
        "best_acc": max(item["best"] for item in epochs),
        "final_acc": epochs[-1]["acc"],
        "final_epoch": epochs[-1]["epoch"],
        "trainable_params": int(parameter_match.group(1).replace(",", "")),
    }


def parse_prompt_modes(value):
    modes = tuple(mode.strip() for mode in value.split(",") if mode.strip())
    if not modes:
        raise ValueError("--prompt-modes must include at least one mode.")

    invalid_modes = sorted(set(modes) - set(MODES))
    if invalid_modes:
        raise ValueError(
            f"Unsupported prompt mode(s): {', '.join(invalid_modes)}. "
            f"Choose from: {', '.join(MODES)}."
        )
    return modes


def main():
    args = parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else newest_log_dir()
    prompt_modes = parse_prompt_modes(args.prompt_modes)
    reference = load_reference(args.reference_csv)
    rows = []
    for dataset in DATASETS:
        dataset_label = DATASET_LABELS[dataset]
        if dataset_label not in reference:
            raise ValueError(f"Missing 40% reference row for {dataset_label}.")
        for mode in prompt_modes:
            log_path = log_dir / (
                f"head_pruned_vpt_{mode}{args.num_prompt_tokens}_{dataset}.log"
            )
            metrics = parse_log(log_path)
            ref = reference[dataset_label]
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label,
                    "prompt_mode": mode,
                    "num_prompt_tokens": args.num_prompt_tokens,
                    **ref,
                    **metrics,
                    "recovery_gain": metrics["best_acc"] - ref["pruning_only_acc"],
                    "vs_reset_lora": metrics["best_acc"] - ref["reset_lora_best_acc"],
                    "log_path": str(log_path),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output}")
    for row in rows:
        print(
            f"{row['dataset_label']:16s} {row['prompt_mode']:7s} "
            f"best={row['best_acc']:.2f} final={row['final_acc']:.2f} "
            f"gain={row['recovery_gain']:+.2f} vs_lora={row['vs_reset_lora']:+.2f}"
        )


if __name__ == "__main__":
    main()
