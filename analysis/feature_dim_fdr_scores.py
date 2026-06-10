"""Compute per-feature-dimension Fisher discriminant scores.

This script creates a score cache that can be consumed by
``feature_mask_linear_probe.py`` or other mask-evaluation scripts. It is a
filter-style feature selection baseline:

1. Load cached train features ``Z`` and labels ``y``.
2. Compute per-dimension within-class variance.
3. Compute per-dimension between-class variance.
4. Save ``FDR_j = between_j / within_j`` as a ``scores`` tensor with shape
   ``[D]``.

The score cache intentionally matches the Taylor score cache format so the
existing masking and linear-probe pipeline can reuse it without special cases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.feature_intraclass_variance import load_feature_cache  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-cache", dest="features_cache", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--fdr-global-mean",
        dest="fdr_global_mean",
        type=str,
        default="sample",
        choices=["sample", "class"],
        help="Global mean definition for FDR. Use sample to match the paper; class is a macro diagnostic.",
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--output-scores", dest="output_scores", type=str, required=True)
    parser.add_argument("--output-json", dest="output_json", type=str, default=None)
    return parser


def compute_per_dim_fdr_scores(
    features: torch.Tensor,
    labels: torch.Tensor,
    global_mean: str = "sample",
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Return per-dimension FDR, within, and between scores."""

    # features: [N, D], labels: [N]
    if global_mean not in {"sample", "class"}:
        raise ValueError(f"Unsupported FDR global mean: {global_mean}")

    features = features.float()
    labels = labels.long()
    classes = torch.unique(labels)
    if classes.numel() == 0:
        raise ValueError("Cannot compute per-dim FDR without labels.")

    class_means = []
    within_terms = []
    for cls in classes:
        # class_features: [N_c, D]
        class_features = features[labels == cls]
        # class_mean: [D]
        class_mean = class_features.mean(dim=0)
        # class_variance: [D]
        class_variance = (class_features - class_mean).square().mean(dim=0)
        class_means.append(class_mean)
        within_terms.append(class_variance)

    # class_means: [C, D], within_terms: [C, D]
    class_means = torch.stack(class_means, dim=0)
    within_terms = torch.stack(within_terms, dim=0)
    within_scores = within_terms.mean(dim=0)  # [D]

    if global_mean == "sample":
        # global_center: [D]. Paper-style sample mean.
        global_center = features.mean(dim=0)
    else:
        # global_center: [D]. Macro/class-balanced diagnostic mean.
        global_center = class_means.mean(dim=0)

    # between_scores: [D]
    between_scores = (class_means - global_center).square().mean(dim=0)
    fdr_scores = between_scores / (within_scores + eps)  # [D]

    return {
        "scores": fdr_scores.detach().cpu(),
        "within_scores": within_scores.detach().cpu(),
        "between_scores": between_scores.detach().cpu(),
    }


def tensor_summary(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    tensor = tensor.float()
    return {
        f"{prefix}_min": float(tensor.min().item()),
        f"{prefix}_max": float(tensor.max().item()),
        f"{prefix}_mean": float(tensor.mean().item()),
        f"{prefix}_std": float(tensor.std(unbiased=False).item()),
    }


def save_scores(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def main(args: argparse.Namespace) -> None:
    features, labels, feature_metadata = load_feature_cache(args.features_cache)
    score_tensors = compute_per_dim_fdr_scores(
        features=features,
        labels=labels,
        global_mean=args.fdr_global_mean,
        eps=args.eps,
    )

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "features_cache": args.features_cache,
        "score_type": "fdr",
        "fdr_global_mean": args.fdr_global_mean,
        "eps": float(args.eps),
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "class_count": int(torch.unique(labels).numel()),
        **tensor_summary(score_tensors["scores"], "score"),
        **tensor_summary(score_tensors["within_scores"], "within"),
        **tensor_summary(score_tensors["between_scores"], "between"),
    }
    if feature_metadata:
        metadata["feature_metadata"] = feature_metadata

    save_scores(
        args.output_scores,
        {
            **score_tensors,
            **metadata,
        },
    )

    print(f"[FeatureFDR] dataset: {metadata['dataset']} ({metadata['split']})")
    print(f"[FeatureFDR] samples/features: {metadata['num_samples']} / {metadata['feature_dim']}")
    print(f"[FeatureFDR] classes: {metadata['class_count']}")
    print(
        "[FeatureFDR] score min/mean/max: "
        f"{metadata['score_min']:.6f} / {metadata['score_mean']:.6f} / {metadata['score_max']:.6f}"
    )
    print(f"[FeatureFDR] scores saved to: {args.output_scores}")

    if args.output_json:
        save_json(args.output_json, metadata)
        print(f"[FeatureFDR] summary saved to: {args.output_json}")


if __name__ == "__main__":
    main(build_parser().parse_args())
