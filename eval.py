import argparse

import torch
import yaml

from engine import run_checkpoint_eval


device = "cuda" if torch.cuda.is_available() else "cpu"


def build_parser():
    # This entrypoint evaluates saved training checkpoints. It is separate from
    # eval_pruned.py, which evaluates pruning artifacts.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the yaml config file")
    parser.add_argument("--checkpoint-path", dest="checkpoint_path", type=str, help="Path to a LoRA merged or pruned VPT training checkpoint")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Optional evaluation batch size override")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate: train or test")
    parser.add_argument("--max-batches", dest="max_batches", type=int, default=None, help="Optional quick eval batch limit")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as file:
            config_dict = yaml.safe_load(file)
        parser.set_defaults(**config_dict)
    args = parser.parse_args()

    run_checkpoint_eval(args, device)
