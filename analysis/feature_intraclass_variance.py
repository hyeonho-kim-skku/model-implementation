"""Measure macro intra-class variance from fine-tuned model features.

This script is the first step for feature-space pruning analysis:

1. Load one fine-tuned baseline checkpoint for a target dataset.
2. Build the matching dataset loader.
3. Extract final backbone features with ``model.forward_features``.
4. Compute the supplement-defined macro intra-class variance.
5. Save a small JSON artifact with the metric and feature metadata.

Later extensions should reuse the same feature extraction path before adding:

1. Final-feature masking experiments.
2. CE-Taylor feature-dimension importance.
3. IntraVar-Taylor feature-dimension importance.
4. Fisher Discriminant Ratio.

The intended first target datasets are the fine-grained datasets used in the
current pruning runs: ``cub200``, ``fgvc_aircraft``, and ``stanford_cars``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import get_loader
from pruning.checkpoint import build_dense_model_from_checkpoint
from utils import move_to_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", dest="checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--data-root", dest="data_root", type=str, default="./data")
    parser.add_argument("--max-batches", dest="max_batches", type=int, default=None)
    parser.add_argument(
        "--features-cache",
        dest="features_cache",
        type=str,
        default=None,
        help="Optional .pt cache to load. If missing and no --features-output is set, this path is used for saving.",
    )
    parser.add_argument(
        "--features-output",
        dest="features_output",
        type=str,
        default=None,
        help="Optional .pt path to save extracted CPU features and labels.",
    )
    parser.add_argument("--output-json", dest="output_json", type=str, default=None)
    return parser


def class_counts(labels: torch.Tensor) -> dict[str, int]:
    """Return class histogram in a JSON-friendly shape."""

    # labels: [N]
    classes, counts = torch.unique(labels.cpu(), return_counts=True)
    return {str(int(cls.item())): int(count.item()) for cls, count in zip(classes, counts)}


@torch.no_grad()
def extract_features(model, dataloader, device: torch.device, max_batches: int | None = None):
    """Run the model once over a loader and cache final backbone features on CPU."""

    model.eval()
    features = []
    labels = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        # images: [B, C, H, W], target: [B]
        images, target = move_to_device(batch, device)
        # batch_features: usually [B, D] for TIMMClassifier.
        batch_features = model.forward_features(images)
        # Most TIMM classifiers already return [B, D]. Flatten only as a guard
        # for backbones that expose spatial feature maps.
        if batch_features.ndim > 2:
            # batch_features: [B, ...] -> [B, D]
            batch_features = torch.flatten(batch_features, start_dim=1)
        features.append(batch_features.detach().cpu())
        labels.append(target.detach().cpu())

    if not features:
        raise ValueError("No feature batches were processed.")
    # features: [N, D], labels: [N]
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def macro_intra_class_variance(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Return supplement-style class-balanced intra-class variance."""

    # features: [N, D], labels: [N]
    features = features.float()
    labels = labels.long()
    classes = torch.unique(labels)
    if classes.numel() == 0:
        raise ValueError("Cannot compute intra-class variance without labels.")

    per_class_variances = []
    for cls in classes:
        # class_features: [N_c, D]
        class_features = features[labels == cls]
        # class_mean: [D]
        class_mean = class_features.mean(dim=0)
        # For each class, compute mean squared Euclidean distance to its class
        # centroid, then average those class variances equally.
        # squared_distances: [N_c]
        squared_distances = ((class_features - class_mean) ** 2).sum(dim=1)
        per_class_variances.append(squared_distances.mean())

    return float(torch.stack(per_class_variances).mean().item())


def load_feature_cache(path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load a previously extracted feature cache."""

    payload = torch.load(path, map_location="cpu")
    if "features" not in payload or "labels" not in payload:
        raise ValueError(f"Feature cache does not contain features and labels: {path}")
    metadata = {key: value for key, value in payload.items() if key not in {"features", "labels"}}
    return payload["features"], payload["labels"], metadata


def save_feature_cache(path: str, features: torch.Tensor, labels: torch.Tensor, metadata: dict) -> None:
    """Save features, labels, and provenance so metric runs can be reproduced."""

    # features: [N, D], labels: [N]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"features": features.cpu(), "labels": labels.cpu(), **metadata}, path)


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def build_loader(args: argparse.Namespace):
    """Build the project dataset loader in deterministic evaluation mode."""

    return get_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        mode="test",
        train=(args.split == "train"),
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )


def load_or_extract_features(args: argparse.Namespace, device: torch.device):
    """Prefer cached features; otherwise reconstruct the model and extract them."""

    cache_path = args.features_cache
    if cache_path and Path(cache_path).exists():
        features, labels, metadata = load_feature_cache(cache_path)
        return features, labels, metadata, "cache"

    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required when no existing --features-cache is provided.")

    # Reuse the dense reconstruction path used by pruning/eval so LoRA
    # checkpoints are converted into a plain TIMMClassifier consistently.
    _, model = build_dense_model_from_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    dataloader = build_loader(args)
    features, labels = extract_features(model, dataloader, device, max_batches=args.max_batches)
    metadata = {
        "checkpoint_path": args.checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "data_root": args.data_root,
        "max_batches": args.max_batches,
    }

    output_path = args.features_output or cache_path
    if output_path:
        save_feature_cache(output_path, features, labels, metadata)
        metadata["features_path"] = output_path

    return features, labels, metadata, "extracted"


def summarize(args: argparse.Namespace, features: torch.Tensor, labels: torch.Tensor, metadata: dict, source: str):
    """Compute raw and normalized feature compactness metrics."""

    # features: [N, D], labels: [N]
    normalized_features = F.normalize(features.float(), dim=1)
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "source": source,
        "checkpoint_path": metadata.get("checkpoint_path") or args.checkpoint_path,
        "features_path": metadata.get("features_path") or args.features_cache,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "class_count": int(torch.unique(labels).numel()),
        "class_counts": class_counts(labels),
        "intra_class_variance": macro_intra_class_variance(features, labels),
        "normalized_intra_class_variance": macro_intra_class_variance(normalized_features, labels),
        "max_batches": metadata.get("max_batches", args.max_batches),
    }
    return result


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels, metadata, source = load_or_extract_features(args, device)
    result = summarize(args, features, labels, metadata, source)

    print(f"[FeatureMetrics] dataset: {result['dataset']} ({result['split']})")
    print(f"[FeatureMetrics] source: {result['source']}")
    print(f"[FeatureMetrics] samples/features: {result['num_samples']} / {result['feature_dim']}")
    print(f"[FeatureMetrics] classes: {result['class_count']}")
    print(f"[FeatureMetrics] intra-class variance: {result['intra_class_variance']:.6f}")
    print(
        "[FeatureMetrics] normalized intra-class variance: "
        f"{result['normalized_intra_class_variance']:.6f}"
    )

    if args.output_json:
        save_json(args.output_json, result)
        print(f"[FeatureMetrics] metrics saved to: {args.output_json}")


if __name__ == "__main__":
    main(build_parser().parse_args())
