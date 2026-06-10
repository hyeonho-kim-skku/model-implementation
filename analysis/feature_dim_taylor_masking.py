"""Estimate final-feature-dimension Taylor importance with explicit masks.

This script is the next step after ``feature_intraclass_variance.py``:

1. Load cached final backbone features ``Z`` and labels ``y``.
2. Load the fine-tuned checkpoint only to recover the trained classifier head.
3. Introduce an explicit feature mask ``m`` with shape ``[B, D]`` per batch.
4. Compute task loss from ``classifier(Z * m)``.
5. Use the Taylor signal of the mask, ``m * dL/dm``, as per-dimension
   feature importance.
6. Later, mask low-importance dimensions and recompute macro intra-class
   variance using the shared metric implementation.

The important modeling choice is that the mask is applied to the final feature
representation, matching the feature-space masking idea rather than physically
removing dimensions. The per-sample mask shape keeps Taylor signals visible as
``[B, D]`` before the selected reduction order combines them into ``[D]``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.feature_intraclass_variance import load_feature_cache, macro_intra_class_variance
from pruning.checkpoint import build_dense_model_from_checkpoint


AXES = ("sample", "element")
MAGNITUDES = ("square", "abs")
MASK_POLICIES = ("low", "high", "random")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", dest="checkpoint_path", type=str, required=True)
    parser.add_argument("--features-cache", dest="features_cache", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--axis", choices=AXES, default="sample")
    parser.add_argument("--magnitude", choices=MAGNITUDES, default="square")
    parser.add_argument("--ratios", type=str, default="0,0.1,0.2,0.3,0.5")
    parser.add_argument(
        "--mask-policies",
        dest="mask_policies",
        type=str,
        default="low",
        help="Comma-separated policies: low, high, random.",
    )
    parser.add_argument(
        "--random-seeds",
        dest="random_seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated seeds used only when mask policy includes random.",
    )
    parser.add_argument("--output-scores", dest="output_scores", type=str, default=None)
    parser.add_argument("--output-jsonl", dest="output_jsonl", type=str, default=None)
    return parser


def parse_ratios(value: str) -> list[float]:
    ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    for ratio in ratios:
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError(f"Masking ratios must satisfy 0 <= ratio < 1, got {ratio}.")
    return ratios


def parse_choices(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    choices = [item.strip() for item in value.split(",") if item.strip()]
    for choice in choices:
        if choice not in allowed:
            raise ValueError(f"Unsupported {name}: {choice}. Expected one of {allowed}.")
    if not choices:
        raise ValueError(f"At least one {name} must be provided.")
    return choices


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_classifier(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    _, model = build_dense_model_from_checkpoint(checkpoint_path, map_location=device)
    classifier = model.classifier.to(device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return classifier


def make_feature_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    # features: [N, D], labels: [N]
    dataset = TensorDataset(features.float(), labels.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def reduce_signal(
    signal: torch.Tensor,
    axis: str,
    magnitude: str,
    element_scores: torch.Tensor | None,
    sample_signal: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Accumulate one batch of mask Taylor signal according to the selected order."""

    # signal: [B, D]
    if axis == "sample":
        if sample_signal is None:
            sample_signal = torch.zeros(signal.shape[1], device=signal.device)
        # sample_signal: [D]. Sample axis is signed-summed before magnitude.
        sample_signal += signal.sum(dim=0)
        return element_scores, sample_signal

    if element_scores is None:
        element_scores = torch.zeros(signal.shape[1], device=signal.device)
    # element_scores: [D]. Magnitude is applied before summing the sample axis.
    if magnitude == "square":
        element_scores += signal.square().sum(dim=0)
    elif magnitude == "abs":
        element_scores += signal.abs().sum(dim=0)
    else:
        raise ValueError(f"Unsupported magnitude: {magnitude}")
    return element_scores, sample_signal


def finalize_scores(
    axis: str,
    magnitude: str,
    element_scores: torch.Tensor | None,
    sample_signal: torch.Tensor | None,
) -> torch.Tensor:
    if axis == "element":
        if element_scores is None:
            raise ValueError("No element scores were accumulated.")
        return element_scores.detach().cpu()

    if sample_signal is None:
        raise ValueError("No sample signal was accumulated.")
    if magnitude == "square":
        return sample_signal.square().detach().cpu()
    if magnitude == "abs":
        return sample_signal.abs().detach().cpu()
    raise ValueError(f"Unsupported magnitude: {magnitude}")


def compute_mask_taylor_scores(
    classifier: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    axis: str,
    magnitude: str,
    device: torch.device,
) -> torch.Tensor:
    """Compute one Taylor importance score per final feature dimension."""

    feature_loader = make_feature_loader(features, labels, batch_size)
    element_scores = None
    sample_signal = None

    for batch_features, batch_labels in feature_loader:
        # batch_features: [B, D], batch_labels: [B]
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        classifier.zero_grad(set_to_none=True)
        # mask: [B, D]. A separate mask element makes each sample-dim Taylor
        # signal visible before the chosen reduction order combines it.
        mask = torch.ones_like(batch_features, requires_grad=True)
        masked_features = batch_features * mask  # [B, D]
        logits = classifier(masked_features)  # [B, C]
        loss = F.cross_entropy(logits, batch_labels, reduction="sum")
        loss.backward()

        # signal: [B, D], t_ij = m_ij * dL / dm_ij.
        signal = mask.detach() * mask.grad.detach()
        element_scores, sample_signal = reduce_signal(
            signal=signal,
            axis=axis,
            magnitude=magnitude,
            element_scores=element_scores,
            sample_signal=sample_signal,
        )

    return finalize_scores(axis, magnitude, element_scores, sample_signal)


def build_feature_mask(
    scores: torch.Tensor,
    ratio: float,
    policy: str,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # scores: [D]
    feature_dim = scores.numel()
    masked_count = int(feature_dim * ratio)

    if policy == "low":
        sorted_indices = torch.argsort(scores, descending=False)  # [D]
        masked_indices = sorted_indices[:masked_count]  # [masked_count]
    elif policy == "high":
        sorted_indices = torch.argsort(scores, descending=True)  # [D]
        masked_indices = sorted_indices[:masked_count]  # [masked_count]
    elif policy == "random":
        rng = random.Random(seed)
        indices = list(range(feature_dim))
        rng.shuffle(indices)
        masked_indices = torch.tensor(indices[:masked_count], dtype=torch.long)  # [masked_count]
    else:
        raise ValueError(f"Unsupported mask policy: {policy}")

    kept_mask = torch.ones(feature_dim, dtype=torch.bool)  # [D]
    if masked_count > 0:
        kept_mask[masked_indices] = False
    kept_indices = torch.nonzero(kept_mask, as_tuple=False).flatten()  # [D - masked_count]

    mask = torch.ones(feature_dim, dtype=torch.float32)  # [D]
    if masked_count > 0:
        mask[masked_indices] = 0.0
    return mask, masked_indices, kept_indices


@torch.no_grad()
def evaluate_masked_classifier(
    classifier: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate the cached-feature classifier after applying one feature mask."""

    # features: [N, D], labels: [N], mask: [D]
    feature_loader = make_feature_loader(features, labels, batch_size)
    mask = mask.to(device).view(1, -1)  # [1, D]

    total_loss = 0.0
    correct = 0
    total = 0
    for batch_features, batch_labels in feature_loader:
        # batch_features: [B, D], batch_labels: [B]
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        masked_features = batch_features * mask  # [B, D]
        logits = classifier(masked_features)  # [B, C]
        loss = F.cross_entropy(logits, batch_labels, reduction="sum")

        total_loss += float(loss.item())
        correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
        total += int(batch_labels.numel())

    if total == 0:
        raise ValueError("No examples were evaluated.")
    return {
        "ce_loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


def masked_evaluation_row(
    features: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    ratio: float,
    policy: str,
    seed: int | None,
    metadata: dict,
    classifier: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> dict:
    # features: [N, D], labels: [N], scores: [D]
    mask, masked_indices, kept_indices = build_feature_mask(scores, ratio, policy, seed)
    masked_features = features.float() * mask.view(1, -1)  # [N, D]
    normalized_masked_features = F.normalize(masked_features, dim=1)  # [N, D]
    classifier_metrics = evaluate_masked_classifier(
        classifier=classifier,
        features=features,
        labels=labels,
        mask=mask,
        batch_size=batch_size,
        device=device,
    )
    return {
        **metadata,
        "mask_policy": policy,
        "seed": seed,
        "ratio": ratio,
        "masked_dims": int(masked_indices.numel()),
        "kept_dims": int(kept_indices.numel()),
        "intra_class_variance": macro_intra_class_variance(masked_features, labels),
        "normalized_intra_class_variance": macro_intra_class_variance(normalized_masked_features, labels),
        **classifier_metrics,
    }


def save_scores(path: str, scores: torch.Tensor, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"scores": scores.cpu(), **payload}, path)


def append_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels, feature_metadata = load_feature_cache(args.features_cache)
    classifier = load_classifier(args.checkpoint_path, device)
    scores = compute_mask_taylor_scores(
        classifier=classifier,
        features=features,
        labels=labels,
        batch_size=args.batch_size,
        axis=args.axis,
        magnitude=args.magnitude,
        device=device,
    )

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint_path": args.checkpoint_path,
        "features_cache": args.features_cache,
        "axis": args.axis,
        "magnitude": args.magnitude,
        "batch_size": args.batch_size,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "score_min": float(scores.min().item()),
        "score_max": float(scores.max().item()),
        "score_mean": float(scores.float().mean().item()),
    }
    if feature_metadata:
        metadata["feature_metadata"] = feature_metadata

    if args.output_scores:
        save_scores(args.output_scores, scores, metadata)

    ratios = parse_ratios(args.ratios)
    policies = parse_choices(args.mask_policies, MASK_POLICIES, "mask policy")
    random_seeds = parse_ints(args.random_seeds)

    rows = []
    for policy in policies:
        seeds = random_seeds if policy == "random" else [None]
        for seed in seeds:
            for ratio in ratios:
                rows.append(
                    masked_evaluation_row(
                        features=features,
                        labels=labels,
                        scores=scores,
                        ratio=ratio,
                        policy=policy,
                        seed=seed,
                        metadata=metadata,
                        classifier=classifier,
                        batch_size=args.batch_size,
                        device=device,
                    )
                )

    for row in rows:
        seed_text = "" if row["seed"] is None else f" seed={row['seed']}"
        print(
            "[FeatureTaylor] "
            f"policy={row['mask_policy']}{seed_text} "
            f"ratio={row['ratio']:.3f} masked={row['masked_dims']} kept={row['kept_dims']} "
            f"intra={row['intra_class_variance']:.6f} "
            f"norm_intra={row['normalized_intra_class_variance']:.6f} "
            f"loss={row['ce_loss']:.6f} acc={row['accuracy']:.2f}"
        )

    if args.output_jsonl:
        append_jsonl(args.output_jsonl, rows)
        print(f"[FeatureTaylor] rows saved to: {args.output_jsonl}")
    if args.output_scores:
        print(f"[FeatureTaylor] scores saved to: {args.output_scores}")


if __name__ == "__main__":
    main(build_parser().parse_args())
