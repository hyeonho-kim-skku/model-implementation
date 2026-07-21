"""CLI for full ImageNet fine-tuning of a serialized pruned artifact."""

import argparse

import yaml

from pruning.finetune import (
    destroy_distributed_runtime,
    initialize_distributed_runtime,
    run_finetune,
)


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
    runtime = initialize_distributed_runtime()
    try:
        summary = run_finetune(config, runtime, resume_path=args.resume)
        if runtime.is_main_process:
            print(f"[FineTune] best_top1={summary['best_top1']:.3f}%")
            print(f"[FineTune] summary={config['output_dir']}/summary.json")
    finally:
        destroy_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
