"""CLI entrypoint for progressive pruning pipelines."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning.source import build_pruning_source
from progressive_pruning.objectives import build_objective
from progressive_pruning.pipeline import (
    parse_target_ratios,
    run_progressive_pruning,
    run_prune_recover_progressive,
)


def parse_calibration_batches(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "full", "all"}:
        return None
    return int(normalized)


def parse_args():
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    # Keep this entrypoint config-driven so CE and fixed-prototype contrastive
    # scoring share one CLI.
    add("--config", type=str, help="Path to a YAML config file.")
    add("--source-type", dest="source_type", choices=["checkpoint", "timm"])
    add("--checkpoint-path", dest="checkpoint_path")
    add("--backbone-name", dest="backbone_name")
    add("--num-classes", dest="num_classes", type=int)
    add("--img-size", dest="img_size", type=int)
    add("--pretrained", action=argparse.BooleanOptionalAction, default=True)

    add("--objective", type=str, default="ce")
    add("--prototype-cache-path", dest="prototype_cache_path", type=str, default=None)
    add("--prototype-temperature", dest="prototype_temperature", type=float, default=0.1)
    add("--prototype-dataset", dest="prototype_dataset", type=str, default=None)
    add("--prototype-split", dest="prototype_split", choices=["train", "test"], default="train")
    add("--prototype-batch-size", dest="prototype_batch_size", type=int, default=None)
    add("--prototype-eval-split", dest="prototype_eval_split", choices=["train", "test"], default="test")
    add("--dataset", type=str)
    add("--calibration-dataset", dest="calibration_dataset", type=str)
    add("--calibration-batch-size", dest="calibration_batch_size", type=int, default=64)
    add("--calibration-batches", dest="calibration_batches", default=None)
    add("--calibration-split", dest="calibration_split", choices=["train", "test"], default="train")
    add("--calibration-seed", dest="calibration_seed", type=int, default=42)
    add("--batch-size", dest="batch_size", type=int, default=64)
    add("--num-workers", dest="num_workers", type=int, default=4)
    add("--data-root", dest="data_root", type=str, default="./data")
    add("--target-ratios", dest="target_ratios", type=str, default=None)

    add("--pruning-modules", dest="pruning_modules", type=str, default="mlp")
    add("--global-pruning", dest="global_pruning", action=argparse.BooleanOptionalAction, default=True)
    add("--round-to", dest="round_to", type=int, default=None)
    add("--gate-taylor-location", dest="gate_taylor_location", type=str, default="fc2_in")
    add("--gate-taylor-reduction", dest="gate_taylor_reduction", type=str, default="sum_square")
    add("--gate-taylor-aggregation", dest="gate_taylor_aggregation", type=str, default="elementwise")
    add("--inspect-groups", dest="inspect_groups", action="store_true")

    add("--output-dir", dest="output_dir", type=str)
    add("--results-path", dest="results_path", type=str, default=None)
    add("--save-artifacts", dest="save_artifacts", action=argparse.BooleanOptionalAction, default=True)
    add("--eval-each-step", dest="eval_each_step", action=argparse.BooleanOptionalAction, default=True)
    add("--eval-dataset", dest="eval_dataset", type=str, default=None)
    add("--eval-batch-size", dest="eval_batch_size", type=int, default=None)
    add("--eval-split", dest="eval_split", choices=["train", "test"], default="test")
    add("--max-batches", dest="max_batches", type=int, default=None)
    add("--seed", type=int, default=42)
    add("--verbose", action=argparse.BooleanOptionalAction, default=True)
    add("--prepare-objective-only", dest="prepare_objective_only", action="store_true")

    add(
        "--pipeline-mode",
        dest="pipeline_mode",
        choices=["rescore_only", "prune_recover"],
        default="rescore_only",
    )
    add(
        "--intermediate-recovery-epochs",
        dest="intermediate_recovery_epochs",
        type=int,
        default=1,
    )
    add(
        "--intermediate-recovery-batch-size",
        dest="intermediate_recovery_batch_size",
        type=int,
        default=None,
    )
    add(
        "--intermediate-recovery-batches",
        dest="intermediate_recovery_batches",
        type=int,
        default=None,
    )
    add(
        "--intermediate-recovery-optimizer",
        dest="intermediate_recovery_optimizer",
        type=str,
        default="AdamW",
    )
    add(
        "--intermediate-recovery-lr",
        dest="intermediate_recovery_lr",
        type=float,
        default=5e-4,
    )
    add(
        "--intermediate-recovery-classifier-lr",
        dest="intermediate_recovery_classifier_lr",
        type=float,
        default=None,
    )
    add(
        "--intermediate-recovery-weight-decay",
        dest="intermediate_recovery_weight_decay",
        type=float,
        default=0.05,
    )
    add(
        "--intermediate-recovery-momentum",
        dest="intermediate_recovery_momentum",
        type=float,
        default=0.9,
    )
    add(
        "--intermediate-recovery-scheduler",
        dest="intermediate_recovery_scheduler",
        type=str,
        default="CosineAnnealingLR",
    )
    add(
        "--intermediate-recovery-warmup-epochs",
        dest="intermediate_recovery_warmup_epochs",
        type=int,
        default=0,
    )
    add(
        "--intermediate-recovery-lora-rank",
        dest="intermediate_recovery_lora_rank",
        type=int,
        default=4,
    )
    add(
        "--intermediate-recovery-lora-alpha",
        dest="intermediate_recovery_lora_alpha",
        type=float,
        default=None,
    )
    add(
        "--intermediate-recovery-lora-modules",
        dest="intermediate_recovery_lora_modules",
        type=str,
        default="qkv,proj,mlp",
    )
    add(
        "--intermediate-recovery-qkv-lora-components",
        dest="intermediate_recovery_qkv_lora_components",
        type=str,
        default="q,k,v",
    )
    add(
        "--intermediate-recovery-seed",
        dest="intermediate_recovery_seed",
        type=int,
        default=None,
    )

    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as file:
            parser.set_defaults(**yaml.safe_load(file))
    args = parser.parse_args()
    args.calibration_batches = parse_calibration_batches(args.calibration_batches)
    if args.target_ratios is not None:
        args.target_ratios = parse_target_ratios(args.target_ratios)
    return args


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    config = vars(args)
    if config.get("target_ratios") is None:
        raise ValueError("target_ratios is required.")
    if not config.get("output_dir"):
        raise ValueError("output_dir is required.")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Source loading remains delegated to pruning/ so checkpoint/timm behavior
    # is shared with existing pruning entrypoints.
    source = build_pruning_source(config, device=device)
    objective = build_objective(args.objective, config=config)
    objective.setup(source, device)
    if args.prepare_objective_only:
        print(json.dumps(objective.metadata(), indent=2))
        print("[ProgressivePruning] objective preparation completed.")
        return

    if args.pipeline_mode == "prune_recover":
        rows = run_prune_recover_progressive(source, objective, config, device)
    else:
        rows = run_progressive_pruning(source, objective, config, device)
    print(f"[ProgressivePruning] completed {len(rows)} pruning steps.")


if __name__ == "__main__":
    main()
