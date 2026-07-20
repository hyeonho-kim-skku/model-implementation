"""Evaluate a dense pretrained timm classifier on a validation split."""

import argparse
import json
import os

import torch
import yaml

from engine import evaluate_timm_baseline


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a YAML config file")
    parser.add_argument("--backbone-name", dest="backbone_name", type=str, required=False)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--data-root", dest="data_root", type=str, default="./data")
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=1000)
    parser.add_argument("--img-size", dest="img_size", type=int, default=None)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--classifier-init",
        dest="classifier_init",
        choices=["random", "pretrained"],
        default="pretrained",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--max-batches", dest="max_batches", type=int, default=None)
    parser.add_argument("--output-json", dest="output_json", type=str, default=None)
    return parser


def validate_args(args):
    if not args.backbone_name:
        raise ValueError("--backbone-name is required.")
    if not args.dataset:
        raise ValueError("--dataset is required.")


def main(args):
    validate_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = evaluate_timm_baseline(
        backbone_name=args.backbone_name,
        dataset_name=args.dataset,
        data_root=args.data_root,
        device=device,
        num_classes=args.num_classes,
        img_size=args.img_size,
        pretrained=args.pretrained,
        classifier_init=args.classifier_init,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )

    print(f"[TIMMEval] device: {device}")
    print(f"[TIMMEval] model: {args.backbone_name}")
    print(f"[TIMMEval] dataset: {args.dataset} (val)")
    print(f"[TIMMEval] data config: {result['data_config']}")
    print(f"[TIMMEval] loss: {result['metrics']['loss']:.4f}")
    print(f"[TIMMEval] top-1: {result['metrics']['top1']:.2f}%")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as file:
            json.dump(result, file, indent=2)
        print(f"[TIMMEval] metrics saved to: {args.output_json}")


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as file:
            parser.set_defaults(**yaml.safe_load(file))
    main(parser.parse_args())
