"""Save and restore cached attention-head gate Taylor scores."""

from __future__ import annotations

import os
from pathlib import Path

import torch


HEAD_GATE_TAYLOR_CACHE_FORMAT = "head_gate_taylor_score_cache_v1"


def capture_head_taylor_scores(model, module_keyed_scores):
    """Snapshot qkv-keyed attention-head scores by block index."""

    _validate_model_blocks(model, "Head gate Taylor score cache")

    snapshot = {}
    for block_idx, block in enumerate(model.encoder.blocks):
        score = module_keyed_scores.get(block.attn.qkv)
        if score is not None:
            snapshot[int(block_idx)] = score.detach().cpu().clone()
    return snapshot


def restore_head_taylor_scores(model, block_idx_scores):
    """Map cached block-index scores onto the current model's qkv modules."""

    _validate_model_blocks(model, "Head gate Taylor score restore")

    restored = {}
    for block_idx, score in block_idx_scores.items():
        block_idx = int(block_idx)
        if block_idx < 0 or block_idx >= len(model.encoder.blocks):
            raise ValueError(f"Cached block index {block_idx} is out of range.")
        qkv = model.encoder.blocks[block_idx].attn.qkv
        restored[qkv] = score.to(qkv.weight.device).clone()
    return restored


def save_head_taylor_score_cache(path, scores, metadata):
    """Persist block-index head gate Taylor scores and metadata."""

    path = Path(path)
    os.makedirs(path.parent or ".", exist_ok=True)
    cpu_scores = {
        int(block_idx): score.detach().cpu().clone()
        for block_idx, score in scores.items()
    }
    payload = {
        "format": HEAD_GATE_TAYLOR_CACHE_FORMAT,
        "metadata": dict(metadata),
        "scores": cpu_scores,
    }
    torch.save(payload, path)


def load_head_taylor_score_cache(path, map_location="cpu"):
    """Load a head gate Taylor score cache."""

    payload = torch.load(path, map_location=map_location)
    if payload.get("format") != HEAD_GATE_TAYLOR_CACHE_FORMAT:
        raise ValueError(f"Unsupported head gate Taylor score cache format in {path!r}.")
    metadata = payload.get("metadata", {})
    scores = {int(block_idx): score for block_idx, score in payload.get("scores", {}).items()}
    if not scores:
        raise ValueError(f"Head gate Taylor score cache has no scores: {path!r}.")
    return scores, metadata


def validate_head_taylor_score_cache(
    model,
    scores,
    metadata,
    *,
    dataset,
    checkpoint_path,
    head_gate_taylor_location,
    head_gate_taylor_reduction,
    head_gate_taylor_aggregation,
    calibration_split,
    calibration_batches,
    calibration_seed,
    target_block_indices=None,
):
    """Validate that a cache matches the requested head-pruning experiment."""

    _validate_model_blocks(model, "Head gate Taylor score cache")
    calibration_batches = calibration_batches if calibration_batches is not None else "full"
    expected = {
        "dataset": dataset,
        "checkpoint_path": checkpoint_path,
        "importance": "head_gate_taylor",
        "head_gate_taylor_location": head_gate_taylor_location,
        "head_gate_taylor_reduction": head_gate_taylor_reduction,
        "head_gate_taylor_aggregation": head_gate_taylor_aggregation,
        "head_gate_taylor_score_mode": "head_gate_grad",
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
        raise ValueError(f"Head gate Taylor score cache metadata mismatch: {mismatches}")

    if target_block_indices is None:
        block_indices = range(len(model.encoder.blocks))
    else:
        block_indices = [int(block_idx) for block_idx in target_block_indices]

    missing_blocks = []
    bad_shapes = {}
    for block_idx in block_indices:
        if block_idx < 0 or block_idx >= len(model.encoder.blocks):
            raise ValueError(f"Target block index {block_idx} is out of range.")
        block = model.encoder.blocks[block_idx]
        score = scores.get(block_idx)
        if score is None:
            missing_blocks.append(block_idx)
            continue
        expected_heads = int(block.attn.num_heads)
        if tuple(score.shape) != (expected_heads,):
            bad_shapes[block_idx] = {
                "expected": (expected_heads,),
                "found": tuple(score.shape),
            }

    if missing_blocks:
        raise ValueError(f"Head gate Taylor score cache is missing blocks: {missing_blocks}")
    if bad_shapes:
        raise ValueError(f"Head gate Taylor score cache shape mismatch: {bad_shapes}")


def _validate_model_blocks(model, context):
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "blocks"):
        raise ValueError(f"{context} needs model.encoder.blocks.")
