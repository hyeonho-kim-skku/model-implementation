"""Summarize VPT and KV prompt recovery logs into a comparison CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"\[Epoch (?P<epoch>\d+)\].*Test Acc: (?P<acc>[0-9.]+)%, "
    r"Best Acc: (?P<best>[0-9.]+)"
)
PARAM_PATTERN = re.compile(
    r"\[(?:TIMMPrunedVPT|TIMMPrunedPrompt)\] trainable params: ([0-9,]+)"
)
MODE_PATTERN = re.compile(r"\[TIMMPrunedVPT\] prompt mode: (\S+)")
COUNTS_PATTERN = re.compile(
    r"\[TIMMPrunedVPT\] prompt tokens per layer: \[([^]]*)\]"
)
TOTAL_PATTERN = re.compile(r"\[TIMMPrunedVPT\] total prompt tokens: (\d+)")
LABEL_PATTERN = re.compile(
    r"\[(?:TIMMPrunedVPT|TIMMPrunedPrompt)\] allocation label: (.+)"
)
MACS_PATTERN = re.compile(r"\[(?:ModelProfile|TIMMPrunedVPT)\] MACs: ([0-9,]+)")
COMPONENTS_PATTERN = re.compile(r"\[TIMMPrunedPrompt\] components: (.+)")
VPT_MODE_PATTERN = re.compile(r"\[TIMMPrunedPrompt\] VPT mode: (\S+)")
VPT_COUNTS_PATTERN = re.compile(
    r"\[TIMMPrunedPrompt\] VPT tokens per layer: \[([^]]*)\]"
)
KV_COUNTS_PATTERN = re.compile(
    r"\[TIMMPrunedPrompt\] KV tokens per layer: \[([^]]*)\]"
)
KV_SHARING_PATTERN = re.compile(
    r"\[TIMMPrunedPrompt\] KV prompt sharing: (none|shared|separate)"
)
TOTAL_VPT_PATTERN = re.compile(r"\[TIMMPrunedPrompt\] total VPT tokens: (\d+)")
TOTAL_KV_PATTERN = re.compile(r"\[TIMMPrunedPrompt\] total KV tokens: (\d+)")
VPT_PROMPT_PARAMS_PATTERN = re.compile(
    r"\[TIMMPrunedPrompt\] VPT prompt params: ([0-9,]+)"
)
KV_PROMPT_PARAMS_PATTERN = re.compile(
    r"\[TIMMPrunedPrompt\] KV prompt params: ([0-9,]+)"
)
LORA_ENABLED_PATTERN = re.compile(
    r"\[TIMMPrunedPrompt\] LoRA enabled: (true|false)"
)
LORA_RANK_PATTERN = re.compile(r"\[TIMMPrunedPrompt\] LoRA rank: (\d+|none)")
LORA_PARAMS_PATTERN = re.compile(
    r"\[TIMMPrunedPrompt\] LoRA params: ([0-9,]+)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated datasets for <dataset>__<experiment>.log files.",
    )
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

    components_match = COMPONENTS_PATTERN.search(text)
    if components_match:
        components = components_match.group(1).strip()
        prompt_mode = required_match(
            VPT_MODE_PATTERN, text, path, "VPT prompt mode"
        ).group(1)
        vpt_counts = parse_counts(
            required_match(
                VPT_COUNTS_PATTERN, text, path, "layer-wise VPT allocation"
            ).group(1)
        )
        kv_counts = parse_counts(
            required_match(
                KV_COUNTS_PATTERN, text, path, "layer-wise KV allocation"
            ).group(1)
        )
        total_vpt_tokens = int(
            required_match(
                TOTAL_VPT_PATTERN, text, path, "total VPT prompt count"
            ).group(1)
        )
        total_kv_tokens = int(
            required_match(
                TOTAL_KV_PATTERN, text, path, "total KV prompt count"
            ).group(1)
        )
        vpt_prompt_params = parse_parameter_count(
            required_match(
                VPT_PROMPT_PARAMS_PATTERN, text, path, "VPT prompt parameters"
            ).group(1)
        )
        kv_prompt_params = parse_parameter_count(
            required_match(
                KV_PROMPT_PARAMS_PATTERN, text, path, "KV prompt parameters"
            ).group(1)
        )
        sharing_match = KV_SHARING_PATTERN.search(text)
        kv_prompt_sharing = (
            sharing_match.group(1)
            if sharing_match
            else ("shared" if "kv" in components.split(",") else "none")
        )
    else:
        components = "vpt"
        prompt_mode = required_match(
            MODE_PATTERN, text, path, "prompt mode"
        ).group(1)
        vpt_counts = parse_counts(
            required_match(
                COUNTS_PATTERN, text, path, "layer-wise prompt allocation"
            ).group(1)
        )
        kv_counts = ()
        total_vpt_tokens = int(
            required_match(TOTAL_PATTERN, text, path, "total prompt count").group(1)
        )
        total_kv_tokens = 0
        vpt_prompt_params = ""
        kv_prompt_params = 0
        kv_prompt_sharing = "none"

    label_match = LABEL_PATTERN.search(text)
    macs_match = MACS_PATTERN.search(text)
    lora_enabled_match = LORA_ENABLED_PATTERN.search(text)
    lora_rank_match = LORA_RANK_PATTERN.search(text)
    lora_params_match = LORA_PARAMS_PATTERN.search(text)
    return {
        "allocation_label": (
            label_match.group(1).strip() if label_match else path.stem
        ),
        "prompt_components": components,
        "vpt_prompt_mode": prompt_mode,
        "vpt_prompt_tokens_per_layer": ",".join(str(count) for count in vpt_counts),
        "kv_prompt_tokens_per_layer": ",".join(str(count) for count in kv_counts),
        "kv_prompt_sharing": kv_prompt_sharing,
        "total_vpt_prompt_tokens": total_vpt_tokens,
        "total_kv_prompt_tokens": total_kv_tokens,
        "vpt_prompt_params": vpt_prompt_params,
        "kv_prompt_params": kv_prompt_params,
        "lora_enabled": (
            lora_enabled_match.group(1) == "true" if lora_enabled_match else False
        ),
        "lora_rank": (
            int(lora_rank_match.group(1))
            if lora_rank_match and lora_rank_match.group(1) != "none"
            else ""
        ),
        "lora_params": (
            parse_parameter_count(lora_params_match.group(1))
            if lora_params_match
            else 0
        ),
        "trainable_params": parse_parameter_count(
            required_match(PARAM_PATTERN, text, path, "trainable parameter count").group(1)
        ),
        "macs": (
            int(macs_match.group(1).replace(",", "")) if macs_match else ""
        ),
        "best_acc": max(item["best"] for item in epochs),
        "final_acc": epochs[-1]["acc"],
        "final_epoch": epochs[-1]["epoch"],
    }


def parse_counts(text):
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def parse_parameter_count(text):
    return int(text.replace(",", ""))


def discover_runs(log_dir, dataset=None, datasets=None):
    if datasets is None:
        paths = sorted(log_dir.glob("*.log"))
        return [(dataset, path) for path in paths]

    selected = tuple(item.strip() for item in datasets.split(",") if item.strip())
    if not selected:
        raise ValueError("--datasets must include at least one dataset.")
    runs = []
    for dataset_name in selected:
        paths = sorted(log_dir.glob(f"{dataset_name}__*.log"))
        if not paths:
            raise FileNotFoundError(
                f"No recovery logs found for {dataset_name} in {log_dir}."
            )
        runs.extend((dataset_name, path) for path in paths)
    return runs


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    runs = discover_runs(log_dir, dataset=args.dataset, datasets=args.datasets)
    if not runs:
        raise FileNotFoundError(f"No .log files found in {log_dir}.")

    rows = []
    for dataset, log_path in runs:
        reference = load_reference(args.reference_csv, dataset)
        metrics = parse_log(log_path)
        rows.append(
            {
                "dataset": dataset,
                **metrics,
                **reference,
                "recovery_gain": round(
                    metrics["best_acc"] - reference["pruning_only_acc"], 4
                ),
                "vs_reset_lora": round(
                    metrics["best_acc"] - reference["reset_lora_best_acc"], 4
                ),
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
            f"{row['allocation_label']:24s} "
            f"vpt={row['total_vpt_prompt_tokens']:3d} "
            f"kv={row['total_kv_prompt_tokens']:3d} "
            f"best={row['best_acc']:.2f} final={row['final_acc']:.2f} "
            f"gain={row['recovery_gain']:+.2f} vs_lora={row['vs_reset_lora']:+.2f}"
        )


if __name__ == "__main__":
    main()
