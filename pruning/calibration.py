"""Shared supervised calibration for Taylor-based pruning criteria."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from datasets import get_loader
from utils import move_to_device


VALID_TAYLOR_CALIBRATION_OBJECTIVES = {"ce", "feature_dim_masked_ce"}


def _normalize_calibration_batches(calibration_batches):
    """Accept an integer batch limit or None/full for the whole loader."""

    if calibration_batches is None:
        return None
    if isinstance(calibration_batches, str):
        value = calibration_batches.strip().lower()
        if value in {"", "none", "null", "full", "all"}:
            return None
        calibration_batches = int(value)
    if calibration_batches < 0:
        raise ValueError("calibration_batches must be non-negative or full.")
    return calibration_batches


def compute_taylor_gradients(
    model,
    calibration_dataset,
    calibration_batch_size,
    calibration_batches,
    calibration_split,
    num_workers,
    data_root,
    device,
    calibration_seed=None,
    activation_taylor_collector=None,
    gate_taylor_collector=None,
    calibration_objective="ce",
    feature_dim_mask=None,
    feature_dim_mask_metadata=None,
    calibration_loss_fn=None,
    head_gate_taylor_collector=None,
):
    """Run supervised batches so Taylor pruning criteria can read gradients."""

    if calibration_dataset is None:
        raise ValueError("Taylor pruning needs calibration_dataset.")
    if calibration_loss_fn is not None and not callable(calibration_loss_fn):
        raise TypeError("calibration_loss_fn must be callable or None.")
    if (
        calibration_loss_fn is None
        and calibration_objective not in VALID_TAYLOR_CALIBRATION_OBJECTIVES
    ):
        raise ValueError(
            "calibration_objective must be one of "
            f"{sorted(VALID_TAYLOR_CALIBRATION_OBJECTIVES)}, got {calibration_objective!r}."
        )
    if (
        calibration_loss_fn is None
        and calibration_objective == "feature_dim_masked_ce"
        and feature_dim_mask is None
    ):
        raise ValueError("feature_dim_mask is required for feature_dim_masked_ce.")

    calibration_batches = _normalize_calibration_batches(calibration_batches)
    generator = None
    if calibration_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(calibration_seed))

    dataloader = get_loader(
        calibration_dataset,
        calibration_batch_size,
        mode="test",
        train=(calibration_split == "train"),
        shuffle=(calibration_split == "train"),
        drop_last=False,
        num_workers=num_workers,
        data_root=data_root,
        generator=generator,
    )

    model.eval()
    model.zero_grad(set_to_none=True)
    if feature_dim_mask is not None:
        feature_dim_mask = feature_dim_mask.to(
            device=device,
            dtype=torch.float32,
        ).view(1, -1)

    processed_batches = 0
    total_examples = 0
    for batch_idx, batch in enumerate(dataloader):
        if calibration_batches is not None and batch_idx >= calibration_batches:
            break
        images, labels = move_to_device(batch, device)
        if calibration_loss_fn is not None:
            loss = calibration_loss_fn(model, images, labels)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError(
                    "calibration_loss_fn must return a scalar torch.Tensor."
                )
            if not loss.requires_grad:
                raise ValueError(
                    "calibration_loss_fn returned a loss that does not require gradients."
                )
            if not torch.isfinite(loss.detach()):
                raise ValueError("calibration_loss_fn returned a non-finite loss.")
        else:
            if calibration_objective == "ce":
                logits = model(images)
            elif calibration_objective == "feature_dim_masked_ce":
                features = model.forward_features(images)
                if features.shape[-1] != feature_dim_mask.shape[-1]:
                    raise ValueError(
                        "Feature mask dimension does not match model features: "
                        f"{feature_dim_mask.shape[-1]} != {features.shape[-1]}."
                    )
                masked_features = features * feature_dim_mask
                logits = model.classifier(masked_features)
            else:
                raise ValueError(
                    f"Unsupported calibration objective: {calibration_objective!r}."
                )
            loss = F.cross_entropy(logits, labels, reduction="sum")

        loss.backward()
        if activation_taylor_collector is not None:
            activation_taylor_collector.accumulate_batch()
        if gate_taylor_collector is not None:
            gate_taylor_collector.accumulate_batch()
        if head_gate_taylor_collector is not None:
            head_gate_taylor_collector.accumulate_batch()

        processed_batches += 1
        total_examples += labels.size(0)

    if processed_batches == 0:
        raise ValueError("Taylor pruning did not process any calibration batches.")

    calibration_config = {
        "dataset": calibration_dataset,
        "batch_size": calibration_batch_size,
        "requested_batches": (
            calibration_batches if calibration_batches is not None else "full"
        ),
        "split": calibration_split,
        "transform_mode": "test",
        "objective": calibration_objective,
        "loss_reduction": "sum",
        "seed": calibration_seed,
        "processed_batches": processed_batches,
        "processed_examples": total_examples,
    }
    if feature_dim_mask_metadata is not None:
        calibration_config["feature_dim_mask"] = dict(feature_dim_mask_metadata)
    if activation_taylor_collector is not None:
        calibration_config["activation_taylor_reduction"] = (
            activation_taylor_collector.reduction
        )
    if gate_taylor_collector is not None:
        calibration_config["gate_taylor_reduction"] = gate_taylor_collector.reduction
        calibration_config["gate_taylor_location"] = gate_taylor_collector.gate_location
        calibration_config["gate_taylor_aggregation"] = gate_taylor_collector.aggregation
        calibration_config["gate_taylor_score_mode"] = gate_taylor_collector.score_mode
    if head_gate_taylor_collector is not None:
        calibration_config["head_gate_taylor_reduction"] = (
            head_gate_taylor_collector.reduction
        )
        calibration_config["head_gate_taylor_location"] = (
            head_gate_taylor_collector.gate_location
        )
        calibration_config["head_gate_taylor_aggregation"] = (
            head_gate_taylor_collector.aggregation
        )
        calibration_config["head_gate_taylor_score_mode"] = (
            head_gate_taylor_collector.score_mode
        )
    return calibration_config
