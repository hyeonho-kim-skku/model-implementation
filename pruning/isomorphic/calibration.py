"""Calibration used by the upstream GroupTaylor Isomorphic method."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from datasets import CONFIG as DATASET_CONFIG
from datasets import get_loader
from pruning.calibration import _normalize_calibration_batches, _resolve_calibration_transform
from utils import move_to_device


def accumulate_group_taylor_gradients(
    model,
    *,
    dataset,
    batch_size,
    batches,
    split,
    seed,
    transform,
    num_workers,
    data_root,
    device,
):
    """Accumulate ordinary GroupTaylor gradients exactly as the reference does.

    The reference repository calls ``cross_entropy`` with its default mean
    reduction once per calibration batch.  This is deliberately separate from
    the project's sum-square/samplewise calibration implementation.
    """

    if dataset is None:
        raise ValueError("isomorphic_taylor requires calibration_dataset.")
    batches = _normalize_calibration_batches(batches)
    transform_mode, transform_metadata = _resolve_calibration_transform(transform)
    dataset_config = DATASET_CONFIG.get(dataset)
    if dataset_config is not None:
        transform_metadata["normalize"] = {
            "mean": tuple(dataset_config["mean"]),
            "std": tuple(dataset_config["std"]),
        }

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    loader = get_loader(
        dataset,
        batch_size,
        mode=transform_mode,
        train=(split == "train"),
        shuffle=(split == "train"),
        drop_last=False,
        num_workers=num_workers,
        data_root=data_root,
        generator=generator,
    )

    model.eval()
    model.zero_grad(set_to_none=True)
    processed_batches = 0
    processed_examples = 0
    for batch_index, batch in enumerate(loader):
        if batches is not None and batch_index >= batches:
            break
        images, labels = move_to_device(batch, device)
        logits = model(images)
        # Keep the reference method's mean-over-batch CE; importance is then
        # evaluated by torch_pruning.importance.GroupTaylorImportance.
        F.cross_entropy(logits, labels).backward()
        processed_batches += 1
        processed_examples += labels.size(0)

    if processed_batches == 0:
        raise ValueError("Isomorphic Taylor calibration did not process any batches.")
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "requested_batches": batches if batches is not None else "full",
        "split": split,
        "transform": transform_metadata,
        "objective": "ce",
        "loss_reduction": "mean",
        "seed": seed,
        "processed_batches": processed_batches,
        "processed_examples": processed_examples,
        "criterion": "torch_pruning.GroupTaylorImportance",
    }
