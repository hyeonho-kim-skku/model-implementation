"""Save and restore cached attention-dim gate Taylor scores."""

from __future__ import annotations

import os
from pathlib import Path

import torch


ATTENTION_DIM_GATE_TAYLOR_CACHE_FORMAT = "attention_dim_gate_taylor_score_cache_v1"


def save_attention_dim_score_cache(path, scores, metadata):
    path = Path(path)
    os.makedirs(path.parent or ".", exist_ok=True)
    payload = {
        "format": ATTENTION_DIM_GATE_TAYLOR_CACHE_FORMAT,
        "metadata": dict(metadata),
        "scores": {
            int(block_idx): score.detach().cpu().clone()
            for block_idx, score in scores.items()
        },
    }
    torch.save(payload, path)


def load_attention_dim_score_cache(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    if payload.get("format") != ATTENTION_DIM_GATE_TAYLOR_CACHE_FORMAT:
        raise ValueError(f"Unsupported attention-dim score cache format in {path!r}.")
    scores = {
        int(block_idx): score
        for block_idx, score in payload.get("scores", {}).items()
    }
    if not scores:
        raise ValueError(f"Attention-dim score cache has no scores: {path!r}.")
    return scores, payload.get("metadata", {})


def validate_attention_dim_score_cache(
    model,
    scores,
    metadata,
    *,
    dataset,
    checkpoint_path,
    attention_dim_target,
    attention_dim_reduction,
    attention_dim_aggregation,
    attention_dim_gate_location,
    calibration_split,
    calibration_batches,
    calibration_seed,
    target_block_indices=None,
):
    calibration_batches = calibration_batches if calibration_batches is not None else "full"
    expected = {
        "dataset": dataset,
        "checkpoint_path": checkpoint_path,
        "importance": "attention_dim_gate_taylor",
        "attention_dim_target": attention_dim_target,
        "attention_dim_reduction": attention_dim_reduction,
        "attention_dim_aggregation": attention_dim_aggregation,
        "attention_dim_gate_location": attention_dim_gate_location,
        "attention_dim_score_mode": "attention_dim_gate_grad",
        "calibration_split": calibration_split,
        "calibration_batches": calibration_batches,
        "calibration_seed": calibration_seed,
        "loss_reduction": "sum",
        "num_blocks": len(model.encoder.blocks),
    }
    mismatches = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Attention-dim score cache metadata mismatch: {mismatches}")

    if target_block_indices is None:
        block_indices = range(len(model.encoder.blocks))
    else:
        block_indices = [int(block_idx) for block_idx in target_block_indices]
    bad_shapes = {}
    missing = []
    for block_idx in block_indices:
        score = scores.get(int(block_idx))
        if score is None:
            missing.append(int(block_idx))
            continue
        attn = model.encoder.blocks[int(block_idx)].attn
        expected_shape = (int(attn.num_heads), int(attn.original_head_dim))
        if tuple(score.shape) != expected_shape:
            bad_shapes[int(block_idx)] = {
                "expected": expected_shape,
                "found": tuple(score.shape),
            }
    if missing:
        raise ValueError(f"Attention-dim score cache missing blocks: {missing}")
    if bad_shapes:
        raise ValueError(f"Attention-dim score cache shape mismatch: {bad_shapes}")
