"""Summarize allocation-aware VPT recovery logs into a comparison CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"\[Epoch (?P<epoch>\d+)\].*Test Acc: (?P<acc>[0-9.]+)%, "
    r"Best Acc: (?P<best>[0-9.]+)"
)
PARAM_PATTERN = re.compile(r"\[TIMMPrunedVPT\] trainable params: ([0-9,]+)")
MODE_PATTERN = re.compile(r"\[TIMMPrunedVPT\] prompt mode: (\S+)")
COUNTS_PATTERN = re.compile(
    r"\[TIMMPrunedVPT\] prompt tokens per layer: \[([^]]*)\]"
)
TOTAL_PATTERN = re.compile(r"\[TIMMPrunedVPT\] total prompt tokens: (\d+)")
LABEL_PATTERN = re.compile(r"\[TIMMPrunedVPT\] allocation label: (.+)")
MACS_PATTERN = re.compile(r"\[(?:ModelProfile|TIMMPrunedVPT)\] MACs: ([0-9,]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument(
        "--reference-csv",
        default="figures/head_gate_taylor_meeting/recovery_reset_comparison.csv",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_reference(path, dataset):
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            if row["dataset"] == dataset and abs(float(row["ratio"]) - 0.4) < 1e-8:
                return {
                    "baseline_acc": float(row["baseline_acc"]),
                    "pruning_only_acc": float(row["pruning_only_acc"]),
                    "reset_lora_best_acc": float(row["reset_recovery_best_acc"]),
                }
    raise ValueError(f"No 40% pruning reference found for dataset={dataset!r}.")


def required_match(pattern, text, path, description):
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"No {description} found in {path}.")
    return match


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

    counts_text = required_match(
        COUNTS_PATTERN, text, path, "layer-wise prompt allocation"
    ).group(1)
    counts = tuple(
        int(value.strip()) for value in counts_text.split(",") if value.strip()
    )
    label_match = LABEL_PATTERN.search(text)
    macs_match = MACS_PATTERN.search(text)
    return {
        "allocation_label": (
            label_match.group(1).strip() if label_match else path.stem
        ),
        "prompt_mode": required_match(
            MODE_PATTERN, text, path, "prompt mode"
        ).group(1),
        "prompt_tokens_per_layer": ",".join(str(count) for count in counts),
        "total_prompt_tokens": int(
            required_match(TOTAL_PATTERN, text, path, "total prompt count").group(1)
        ),
        "trainable_params": int(
            required_match(PARAM_PATTERN, text, path, "trainable parameter count")
            .group(1)
            .replace(",", "")
        ),
        "macs": (
            int(macs_match.group(1).replace(",", "")) if macs_match else ""
        ),
        "best_acc": max(item["best"] for item in epochs),
        "final_acc": epochs[-1]["acc"],
        "final_epoch": epochs[-1]["epoch"],
    }


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_paths = sorted(log_dir.glob("*.log"))
    if not log_paths:
        raise FileNotFoundError(f"No .log files found in {log_dir}.")

    reference = load_reference(args.reference_csv, args.dataset)
    rows = []
    for log_path in log_paths:
        metrics = parse_log(log_path)
        rows.append(
            {
                "dataset": args.dataset,
                **metrics,
                **reference,
                "recovery_gain": metrics["best_acc"]
                - reference["pruning_only_acc"],
                "vs_reset_lora": metrics["best_acc"]
                - reference["reset_lora_best_acc"],
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
            f"{row['allocation_label']:24s} tokens={row['total_prompt_tokens']:3d} "
            f"best={row['best_acc']:.2f} final={row['final_acc']:.2f} "
            f"gain={row['recovery_gain']:+.2f} vs_lora={row['vs_reset_lora']:+.2f}"
        )


if __name__ == "__main__":
    main()
