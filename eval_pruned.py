import argparse
import json
import os

import torch
import yaml

from pruning.eval import evaluate_pruned_model


device = "cuda" if torch.cuda.is_available() else "cpu"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the yaml config file")
    parser.add_argument("--artifact-path", dest="artifact_path", type=str, help="Path to a pruned model artifact")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--data-root", dest="data_root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate: train or test")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--max-batches", dest="max_batches", type=int, default=None, help="Optional quick eval batch limit")
    parser.add_argument("--output-json", dest="output_json", type=str, default=None, help="Optional path to save evaluation metrics as JSON")
    return parser


def save_metrics_json(args, artifact, metrics):
    result = {
        "artifact_path": args.artifact_path,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "metrics": metrics,
        "model_config": artifact.get("model_config", {}),
        "source": artifact.get("source", {}),
        "pruning_config": artifact.get("pruning_config", {}),
        "pruning_stats": artifact.get("pruning_stats", {}),
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as file:
        json.dump(result, file, indent=2)
    print(f"[PrunedEval] metrics saved to: {args.output_json}")


def main(args):
    artifact, metrics = evaluate_pruned_model(
        artifact_path=args.artifact_path,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        split=args.split,
        device=device,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        data_root=args.data_root,
    )
    print(f"[PrunedEval] artifact: {args.artifact_path}")
    print(f"[PrunedEval] dataset: {args.dataset} ({args.split})")
    print(f"[PrunedEval] loss: {metrics['loss']:.4f}")
    print(f"[PrunedEval] acc: {metrics['acc']:.2f}%")
    print(f"[PrunedEval] source: {artifact.get('source', {})}")
    print(f"[PrunedEval] pruning config: {artifact.get('pruning_config', {})}")
    print(f"[PrunedEval] pruning stats: {artifact.get('pruning_stats', {})}")
    if args.output_json:
        save_metrics_json(args, artifact, metrics)


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as file:
            config_dict = yaml.safe_load(file)
        parser.set_defaults(**config_dict)
    args = parser.parse_args()

    main(args)
