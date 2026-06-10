"""Retrain a linear classifier on masked cached features.

This script tests whether feature-dimension masks that improve compactness can
also support classification after the classifier head adapts to the masked
feature distribution.

Workflow:

1. Load cached train/test final features and labels.
2. Load a per-dimension score cache, such as Taylor scores from train features.
3. Build one feature mask per ratio/policy.
4. Train a fresh linear classifier on masked train features.
5. Evaluate the retrained head on masked test features.
6. Save one JSONL row per mask configuration.

The backbone is not trained here. This is intentionally a lightweight
feature-space linear probe / head-retraining experiment.
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

from analysis.feature_dim_taylor_masking import (  # noqa: E402
    MASK_POLICIES,
    build_feature_mask,
    parse_choices,
    parse_ints,
    parse_ratios,
)
from analysis.feature_intraclass_variance import (  # noqa: E402
    load_feature_cache,
    macro_fisher_discriminant_ratio,
    macro_intra_class_variance,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features-cache", dest="train_features_cache", type=str, required=True)
    parser.add_argument("--test-features-cache", dest="test_features_cache", type=str, required=True)
    parser.add_argument("--scores-cache", dest="scores_cache", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ratios", type=str, default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument(
        "--mask-policies",
        dest="mask_policies",
        type=str,
        default="low,high,random",
        help="Comma-separated policies: low, high, random.",
    )
    parser.add_argument(
        "--random-seeds",
        dest="random_seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds used only when mask policy includes random.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device string. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--fdr-global-mean",
        dest="fdr_global_mean",
        type=str,
        default="sample",
        choices=["sample", "class"],
        help="Global mean definition for FDR. Use sample to match the paper; class is a macro diagnostic.",
    )
    parser.add_argument("--output-jsonl", dest="output_jsonl", type=str, required=True)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_scores_cache(path: str) -> tuple[torch.Tensor, dict]:
    """Load per-dimension scores and keep score provenance for JSONL rows."""

    payload = torch.load(path, map_location="cpu")
    if "scores" not in payload:
        raise ValueError(f"Scores cache does not contain a 'scores' tensor: {path}")
    metadata = {key: value for key, value in payload.items() if key != "scores"}
    return payload["scores"].float(), metadata


def infer_num_classes(*labels_list: torch.Tensor) -> int:
    """Infer class count from one or more zero-indexed label tensors."""

    max_label = max(int(labels.long().max().item()) for labels in labels_list)
    return max_label + 1


def make_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    # features: [N, D], labels: [N]
    dataset = TensorDataset(features.float(), labels.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def apply_feature_mask(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # features: [N, D], mask: [D]
    return features.float() * mask.float().view(1, -1)  # [N, D]


def train_linear_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[torch.nn.Linear, dict]:
    """Train one fresh linear classifier on masked cached features."""

    # train_features: [N_train, D], train_labels: [N_train]
    classifier = torch.nn.Linear(train_features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_loader(train_features, train_labels, batch_size, shuffle=True)

    last_loss = 0.0
    for _epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        total = 0
        for batch_features, batch_labels in train_loader:
            # batch_features: [B, D], batch_labels: [B]
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = classifier(batch_features)  # [B, C]
            loss = F.cross_entropy(logits, batch_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(batch_labels.numel())
            total_loss += float(loss.item()) * batch_size_actual
            total += batch_size_actual

        if total == 0:
            raise ValueError("No training examples were processed.")
        last_loss = total_loss / total

    return classifier, {"train_ce_loss": last_loss}


@torch.no_grad()
def evaluate_linear_head(
    classifier: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate a trained linear head on cached features."""

    # features: [N, D], labels: [N]
    classifier.eval()
    loader = make_loader(features, labels, batch_size, shuffle=False)

    total_loss = 0.0
    correct = 0
    total = 0
    for batch_features, batch_labels in loader:
        # batch_features: [B, D], batch_labels: [B]
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        logits = classifier(batch_features)  # [B, C]
        loss = F.cross_entropy(logits, batch_labels, reduction="sum")
        predictions = logits.argmax(dim=1)  # [B]

        total_loss += float(loss.item())
        correct += int((predictions == batch_labels).sum().item())
        total += int(batch_labels.numel())

    if total == 0:
        raise ValueError("No evaluation examples were processed.")
    return {
        "test_ce_loss": total_loss / total,
        "test_accuracy": 100.0 * correct / total,
    }


def feature_compactness_rows(
    features: torch.Tensor,
    labels: torch.Tensor,
    fdr_global_mean: str,
) -> dict:
    """Compute raw and L2-normalized test feature compactness metrics."""

    # features: [N, D], labels: [N]
    normalized_features = F.normalize(features.float(), dim=1)  # [N, D]
    fdr_metrics = macro_fisher_discriminant_ratio(features, labels, global_mean=fdr_global_mean)
    normalized_fdr_metrics = macro_fisher_discriminant_ratio(
        normalized_features,
        labels,
        global_mean=fdr_global_mean,
    )
    return {
        "test_intra_class_variance": macro_intra_class_variance(features, labels),
        "test_normalized_intra_class_variance": macro_intra_class_variance(normalized_features, labels),
        "test_within_class_variance": fdr_metrics["within_class_variance"],
        "test_between_class_variance": fdr_metrics["between_class_variance"],
        "test_fisher_discriminant_ratio": fdr_metrics["fisher_discriminant_ratio"],
        "test_normalized_within_class_variance": normalized_fdr_metrics["within_class_variance"],
        "test_normalized_between_class_variance": normalized_fdr_metrics["between_class_variance"],
        "test_normalized_fisher_discriminant_ratio": normalized_fdr_metrics["fisher_discriminant_ratio"],
    }


def run_one_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    scores: torch.Tensor,
    ratio: float,
    policy: str,
    seed: int | None,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    metadata: dict,
) -> dict:
    """Build one mask, retrain one head, and return one JSONL-ready row."""

    # train_features/test_features: [N, D], scores/mask: [D]
    mask, masked_indices, kept_indices = build_feature_mask(scores, ratio, policy, seed)
    masked_train_features = apply_feature_mask(train_features, mask)  # [N_train, D]
    masked_test_features = apply_feature_mask(test_features, mask)  # [N_test, D]

    classifier, train_metrics = train_linear_head(
        train_features=masked_train_features,
        train_labels=train_labels,
        num_classes=num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    test_metrics = evaluate_linear_head(
        classifier=classifier,
        features=masked_test_features,
        labels=test_labels,
        batch_size=args.batch_size,
        device=device,
    )
    compactness_metrics = feature_compactness_rows(
        masked_test_features,
        test_labels,
        fdr_global_mean=args.fdr_global_mean,
    )

    return {
        **metadata,
        "mask_policy": policy,
        "seed": seed,
        "ratio": ratio,
        "masked_dims": int(masked_indices.numel()),
        "kept_dims": int(kept_indices.numel()),
        **train_metrics,
        **test_metrics,
        **compactness_metrics,
    }


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_features, train_labels, train_metadata = load_feature_cache(args.train_features_cache)
    test_features, test_labels, test_metadata = load_feature_cache(args.test_features_cache)
    scores, score_metadata = load_scores_cache(args.scores_cache)

    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError(
            "Train and test feature dimensions must match: "
            f"{train_features.shape[1]} != {test_features.shape[1]}"
        )
    if train_features.shape[1] != scores.numel():
        raise ValueError(
            "Feature dimension and score length must match: "
            f"{train_features.shape[1]} != {scores.numel()}"
        )

    ratios = parse_ratios(args.ratios)
    policies = parse_choices(args.mask_policies, MASK_POLICIES, "mask policy")
    random_seeds = parse_ints(args.random_seeds)
    num_classes = infer_num_classes(train_labels, test_labels)

    metadata = {
        "dataset": args.dataset,
        "train_features_cache": args.train_features_cache,
        "test_features_cache": args.test_features_cache,
        "scores_cache": args.scores_cache,
        "num_train_samples": int(train_features.shape[0]),
        "num_test_samples": int(test_features.shape[0]),
        "feature_dim": int(train_features.shape[1]),
        "num_classes": int(num_classes),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "fdr_global_mean": args.fdr_global_mean,
        "run_seed": int(args.seed),
    }
    if train_metadata:
        metadata["train_feature_metadata"] = train_metadata
    if test_metadata:
        metadata["test_feature_metadata"] = test_metadata
    if score_metadata:
        metadata["score_metadata"] = score_metadata

    rows = []
    for policy in policies:
        seeds = random_seeds if policy == "random" else [None]
        for seed in seeds:
            for ratio in ratios:
                # Re-seed before each probe so policy comparisons are not driven
                # by different linear-head initializations or dataloader order.
                # For random masks, ``seed`` changes only the selected feature
                # dimensions; the train seed stays fixed for a cleaner baseline.
                probe_seed = args.seed
                set_seed(probe_seed)
                row = run_one_probe(
                    train_features=train_features,
                    train_labels=train_labels,
                    test_features=test_features,
                    test_labels=test_labels,
                    scores=scores,
                    ratio=ratio,
                    policy=policy,
                    seed=seed,
                    num_classes=num_classes,
                    args=args,
                    device=device,
                    metadata={**metadata, "probe_seed": int(probe_seed)},
                )
                rows.append(row)

                seed_text = "" if seed is None else f" seed={seed}"
                print(
                    "[FeatureProbe] "
                    f"policy={policy}{seed_text} ratio={ratio:.3f} "
                    f"masked={row['masked_dims']} kept={row['kept_dims']} "
                    f"train_loss={row['train_ce_loss']:.6f} "
                    f"test_loss={row['test_ce_loss']:.6f} "
                    f"test_acc={row['test_accuracy']:.2f} "
                    f"test_norm_intra={row['test_normalized_intra_class_variance']:.6f} "
                    f"test_norm_fdr={row['test_normalized_fisher_discriminant_ratio']:.6f}"
                )

    write_jsonl(args.output_jsonl, rows)
    print(f"[FeatureProbe] rows saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main(build_parser().parse_args())
