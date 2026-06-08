"""Save and restore cached MLP gate Taylor channel scores."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def capture_mlp_taylor_scores(model, module_keyed_scores):
    """Snapshot fc1-keyed MLP scores by block index."""

    if not hasattr(model.encoder, "blocks"):
        raise ValueError("Gate Taylor score cache needs model.encoder.blocks.")

    snapshot = {}
    for block_idx, block in enumerate(model.encoder.blocks):
        score = module_keyed_scores.get(block.mlp.fc1)
        if score is not None:
            snapshot[int(block_idx)] = score.detach().cpu().clone()
    return snapshot


def restore_mlp_taylor_scores(model, block_idx_scores):
    """Map cached block-index scores onto the current model's fc1 modules."""

    if not hasattr(model.encoder, "blocks"):
        raise ValueError("Gate Taylor score restore needs model.encoder.blocks.")

    restored = {}
    for block_idx, score in block_idx_scores.items():
        block_idx = int(block_idx)
        if block_idx < 0 or block_idx >= len(model.encoder.blocks):
            raise ValueError(f"Cached block index {block_idx} is out of range.")
        fc1 = model.encoder.blocks[block_idx].mlp.fc1
        restored[fc1] = score.to(fc1.weight.device).clone()
    return restored


def save_gate_taylor_score_cache(path, scores, metadata):
    """Persist block-index gate Taylor scores and metadata."""

    path = Path(path)
    os.makedirs(path.parent or ".", exist_ok=True)
    cpu_scores = {
        int(block_idx): score.detach().cpu().clone()
        for block_idx, score in scores.items()
    }
    payload = {
        "format": "gate_taylor_score_cache_v1",
        "metadata": dict(metadata),
        "scores": cpu_scores,
    }
    torch.save(payload, path)


def load_gate_taylor_score_cache(path, map_location="cpu"):
    """Load a gate Taylor score cache created by save_gate_taylor_score_cache()."""

    payload = torch.load(path, map_location=map_location)
    if payload.get("format") != "gate_taylor_score_cache_v1":
        raise ValueError(f"Unsupported gate Taylor score cache format in {path!r}.")
    metadata = payload.get("metadata", {})
    scores = {int(block_idx): score for block_idx, score in payload.get("scores", {}).items()}
    if not scores:
        raise ValueError(f"Gate Taylor score cache has no scores: {path!r}.")
    return scores, metadata


def validate_gate_taylor_score_cache(
    model,
    scores,
    metadata,
    *,
    dataset,
    checkpoint_path,
    gate_taylor_location,
    gate_taylor_reduction,
    gate_taylor_aggregation,
    calibration_split,
    calibration_batches,
    calibration_seed,
):
    """Validate that a cache matches the requested global pruning experiment."""

    calibration_batches = calibration_batches if calibration_batches is not None else "full"
    expected = {
        "dataset": dataset,
        "checkpoint_path": checkpoint_path,
        "importance": "gate_taylor",
        "gate_taylor_location": gate_taylor_location,
        "gate_taylor_reduction": gate_taylor_reduction,
        "gate_taylor_aggregation": gate_taylor_aggregation,
        "gate_taylor_score_mode": "elementwise_gate_grad",
        "calibration_split": calibration_split,
        "calibration_batches": calibration_batches,
        "calibration_seed": calibration_seed,
        "loss_reduction": "sum",
        "num_blocks": len(model.encoder.blocks),
    }

    mismatches = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key, "elementwise" if key == "gate_taylor_aggregation" else None) != value
    }
    if mismatches:
        raise ValueError(f"Gate Taylor score cache metadata mismatch: {mismatches}")

    missing_blocks = []
    bad_shapes = {}
    for block_idx, block in enumerate(model.encoder.blocks):
        score = scores.get(block_idx)
        if score is None:
            missing_blocks.append(block_idx)
            continue
        expected_hidden = block.mlp.fc1.out_features
        if tuple(score.shape) != (expected_hidden,):
            bad_shapes[block_idx] = {
                "expected": (expected_hidden,),
                "found": tuple(score.shape),
            }

    if missing_blocks:
        raise ValueError(f"Gate Taylor score cache is missing blocks: {missing_blocks}")
    if bad_shapes:
        raise ValueError(f"Gate Taylor score cache shape mismatch: {bad_shapes}")
