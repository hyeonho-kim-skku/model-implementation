"""Utilities for fixed final-feature masks used during Taylor calibration."""

from __future__ import annotations

import torch


VALID_FEATURE_DIM_MASK_POLICIES = {"low", "high"}


def load_feature_dim_scores(path, map_location="cpu"):
    """Load one score per final feature dimension from a torch payload."""

    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, torch.Tensor):
        scores = payload
        metadata = {}
    elif isinstance(payload, dict):
        if "scores" not in payload:
            raise ValueError(f"Feature-dim score file has no 'scores' tensor: {path!r}.")
        scores = payload["scores"]
        metadata = {key: value for key, value in payload.items() if key != "scores"}
    else:
        raise ValueError(f"Unsupported feature-dim score payload in {path!r}.")

    if not isinstance(scores, torch.Tensor):
        raise ValueError(f"Feature-dim scores must be a tensor: {path!r}.")
    if scores.ndim != 1:
        raise ValueError(
            f"Feature-dim scores must have shape [D], got {tuple(scores.shape)} from {path!r}."
        )
    return scores.float().cpu(), metadata


def build_feature_dim_mask(scores, ratio, policy):
    """Build a binary keep mask from final-feature-dimension scores.

    scores: [D]
    mask: [D], where 1 keeps a feature dim and 0 removes it from the loss.
    """

    if policy not in VALID_FEATURE_DIM_MASK_POLICIES:
        raise ValueError(
            "feature_dim_mask_policy must be one of "
            f"{sorted(VALID_FEATURE_DIM_MASK_POLICIES)}, got {policy!r}."
        )
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"feature_dim_mask_ratio must satisfy 0 <= ratio < 1, got {ratio}.")

    feature_dim = int(scores.numel())
    masked_count = int(feature_dim * ratio)
    if policy == "low":
        sorted_indices = torch.argsort(scores, descending=False)  # [D]
    else:
        sorted_indices = torch.argsort(scores, descending=True)  # [D]

    masked_indices = sorted_indices[:masked_count].long()  # [masked_count]
    mask = torch.ones(feature_dim, dtype=torch.float32)  # [D]
    if masked_count > 0:
        mask[masked_indices] = 0.0
    kept_indices = torch.nonzero(mask.bool(), as_tuple=False).flatten()  # [D - masked_count]
    return mask, masked_indices, kept_indices


def load_feature_dim_mask(path, ratio, policy, expected_dim=None):
    """Load scores and return a fixed final-feature keep mask plus metadata."""

    scores, score_metadata = load_feature_dim_scores(path)
    if expected_dim is not None and int(scores.numel()) != int(expected_dim):
        raise ValueError(
            "Feature-dim score length does not match model feature dim: "
            f"{scores.numel()} != {expected_dim}."
        )

    mask, masked_indices, kept_indices = build_feature_dim_mask(scores, ratio, policy)
    metadata = {
        "feature_dim_score_path": str(path),
        "feature_dim_mask_ratio": float(ratio),
        "feature_dim_mask_policy": policy,
        "feature_dim": int(scores.numel()),
        "feature_dim_masked_dims": int(masked_indices.numel()),
        "feature_dim_kept_dims": int(kept_indices.numel()),
        "feature_dim_score_min": float(scores.min().item()),
        "feature_dim_score_max": float(scores.max().item()),
        "feature_dim_score_mean": float(scores.mean().item()),
    }
    if score_metadata:
        metadata["feature_dim_score_metadata"] = score_metadata
    return mask, metadata
