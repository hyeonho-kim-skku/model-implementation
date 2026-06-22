"""CLI entrypoint for progressive pruning pipelines."""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning.source import build_pruning_source
from progressive_pruning.objectives import build_objective
from progressive_pruning.pipeline import parse_target_ratios, run_progressive_pruning


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

    # Keep this entrypoint config-driven so CE and Prototype/SupCon share one
    # CLI.
    add("--config", type=str, help="Path to a YAML config file.")
    add("--source-type", dest="source_type", choices=["checkpoint", "timm"])
    add("--checkpoint-path", dest="checkpoint_path")
    add("--backbone-name", dest="backbone_name")
    add("--num-classes", dest="num_classes", type=int)
    add("--img-size", dest="img_size", type=int)
    add("--pretrained", action=argparse.BooleanOptionalAction, default=True)

    add("--objective", type=str, default="ce")
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
    objective = build_objective(args.objective)
    objective.setup(source, device)

    rows = run_progressive_pruning(source, objective, config, device)
    print(f"[ProgressivePruning] completed {len(rows)} pruning steps.")


if __name__ == "__main__":
    main()
