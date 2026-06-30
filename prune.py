import argparse
import os

import torch
import yaml

from pruning.source import build_pruning_source
from pruning.structured import prune_model


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_calibration_batches(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = str(value).strip().lower()
    if value in {"", "none", "null", "full", "all"}:
        return None
    return int(value)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the yaml config file")
    parser.add_argument("--source-type", dest="source_type", type=str, choices=["checkpoint", "timm"], help="Model source for pruning")
    parser.add_argument("--checkpoint-path", dest="checkpoint_path", type=str, help="Path to a checkpoint source")
    parser.add_argument("--backbone-name", dest="backbone_name", type=str, help="timm backbone name for source_type=timm")
    parser.add_argument("--num-classes", dest="num_classes", type=int, help="Number of classes for source_type=timm")
    parser.add_argument("--img-size", dest="img_size", type=int, help="Input image size for source_type=timm")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Load pretrained timm weights for source_type=timm")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="./pruned", help="Directory to save pruned artifacts")
    parser.add_argument("--output-path", dest="output_path", type=str, default=None, help="Optional full path for the pruned artifact")
    parser.add_argument("--importance", dest="importance", type=str, choices=["magnitude", "taylor", "activation_taylor", "gate_taylor", "head_gate_taylor"], default="magnitude", help="Importance criterion for structured pruning")
    parser.add_argument("--activation-taylor-reduction", dest="activation_taylor_reduction", type=str, choices=["sum_abs", "abs_sum"], default="sum_abs", help="Reduction for activation_taylor scores")
    parser.add_argument("--gate-taylor-reduction", dest="gate_taylor_reduction", type=str, choices=["signed_damage", "sum_abs", "sum_square"], default="sum_abs", help="Reduction for gate_taylor scores")
    parser.add_argument("--gate-taylor-location", dest="gate_taylor_location", type=str, default="fc1_out", help="Gate insertion point for gate_taylor")
    parser.add_argument("--gate-taylor-aggregation", dest="gate_taylor_aggregation", type=str, choices=["elementwise", "samplewise", "channelwise", "tokenwise"], default="elementwise", help="Aggregation unit for gate_taylor scores")
    parser.add_argument("--head-gate-taylor-reduction", dest="head_gate_taylor_reduction", type=str, choices=["signed_damage", "sum_abs", "sum_square"], default="sum_abs", help="Reduction for head_gate_taylor scores")
    parser.add_argument("--head-gate-taylor-aggregation", dest="head_gate_taylor_aggregation", type=str, choices=["elementwise", "samplewise", "channelwise", "tokenwise"], default="samplewise", help="Aggregation unit for head_gate_taylor scores")
    parser.add_argument("--pruning-ratio", dest="pruning_ratio", type=float, default=0.2, help="Structured pruning ratio")
    parser.add_argument("--pruning-modules", dest="pruning_modules", type=str, default=None, help="Comma-separated pruning targets: head,mlp")
    parser.add_argument("--target-block-indices", dest="target_block_indices", type=str, default=None, help="Optional comma-separated transformer block indices to prune")
    parser.add_argument("--iterative-steps", dest="iterative_steps", type=int, default=1, help="Number of iterative pruning steps")
    parser.add_argument("--global-pruning", dest="global_pruning", action=argparse.BooleanOptionalAction, default=False, help="Use global pruning across target modules")
    parser.add_argument("--round-to", dest="round_to", type=int, default=None, help="Round pruned dimensions to a multiple")
    parser.add_argument("--calibration-dataset", dest="calibration_dataset", type=str, default=None, help="Dataset used to compute Taylor gradients")
    parser.add_argument("--calibration-batch-size", dest="calibration_batch_size", type=int, default=64, help="Batch size for Taylor calibration")
    parser.add_argument("--calibration-batches", dest="calibration_batches", type=parse_calibration_batches, default=1, help="Number of Taylor calibration batches, or 'full'")
    parser.add_argument("--calibration-split", dest="calibration_split", type=str, choices=["train", "test"], default="train", help="Dataset split for Taylor calibration")
    parser.add_argument("--calibration-seed", dest="calibration_seed", type=int, default=None, help="Optional DataLoader shuffle seed for Taylor calibration")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--data-root", dest="data_root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--inspect-groups", dest="inspect_groups", action="store_true", help="Print target shape changes after pruning")
    return parser


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    source = build_pruning_source(vars(args), device=device)
    prune_model(
        model=source.model,
        model_config=source.model_config,
        source_info=source.source_info,
        output_dir=args.output_dir,
        output_path=args.output_path,
        importance=args.importance,
        pruning_ratio=args.pruning_ratio,
        pruning_modules=args.pruning_modules,
        target_block_indices=args.target_block_indices,
        iterative_steps=args.iterative_steps,
        global_pruning=args.global_pruning,
        round_to=args.round_to,
        calibration_dataset=args.calibration_dataset,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        calibration_split=args.calibration_split,
        calibration_seed=args.calibration_seed,
        activation_taylor_reduction=args.activation_taylor_reduction,
        gate_taylor_reduction=args.gate_taylor_reduction,
        gate_taylor_location=args.gate_taylor_location,
        gate_taylor_aggregation=args.gate_taylor_aggregation,
        head_gate_taylor_reduction=args.head_gate_taylor_reduction,
        head_gate_taylor_aggregation=args.head_gate_taylor_aggregation,
        num_workers=args.num_workers,
        data_root=args.data_root,
        inspect_groups=args.inspect_groups,
        device=device,
    )


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as file:
            config_dict = yaml.safe_load(file)
        parser.set_defaults(**config_dict)
    args = parser.parse_args()
    args.calibration_batches = parse_calibration_batches(args.calibration_batches)

    main(args)
