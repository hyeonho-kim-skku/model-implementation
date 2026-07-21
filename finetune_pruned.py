"""CLI for full ImageNet fine-tuning of a serialized pruned artifact."""

import argparse

import torch
import yaml

from pruning.finetune import run_finetune


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    for key in ("epochs", "output_dir", "max_train_batches", "max_val_batches"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; launch with CUDA_VISIBLE_DEVICES=7.")
    summary = run_finetune(config, device="cuda", resume_path=args.resume)
    print(f"[FineTune] best_top1={summary['best_top1']:.3f}%")
    print(f"[FineTune] summary={config['output_dir']}/summary.json")


if __name__ == "__main__":
    main()
