"""Fixed class-prototype utilities for representation-aware pruning scores."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from utils import move_to_device


PROTOTYPE_CACHE_VERSION = 1
_UNIT_NORM_ATOL = 1e-4


def _validate_feature_batch(features, labels):
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        shape = getattr(features, "shape", None)
        raise ValueError(f"Expected [batch, feature_dim] features, got {shape}.")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        shape = getattr(labels, "shape", None)
        raise ValueError(f"Expected [batch] labels, got {shape}.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Feature and label batch sizes differ: "
            f"{features.shape[0]} != {labels.shape[0]}."
        )
    if not torch.isfinite(features).all():
        raise ValueError("Prototype features contain non-finite values.")


@torch.no_grad()
def build_class_prototypes(
    model,
    dataloader,
    num_classes,
    device,
    max_batches=None,
):
    """Build unit class means from unit-normalized final backbone features."""

    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")

    model = model.to(device)
    model.eval()
    class_sums = None
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    processed_batches = 0

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = move_to_device(batch, device)
        labels = labels.long()
        features = model.forward_features(images)
        _validate_feature_batch(features, labels)

        if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
            raise ValueError(
                f"Labels must be in [0, {num_classes}), got "
                f"[{int(labels.min())}, {int(labels.max())}]."
            )

        features = F.normalize(features.float(), p=2, dim=-1)
        if not torch.isfinite(features).all():
            raise ValueError("Normalized prototype features contain non-finite values.")

        features_cpu = features.cpu()
        labels_cpu = labels.cpu()
        if class_sums is None:
            class_sums = torch.zeros(
                num_classes,
                features_cpu.shape[1],
                dtype=torch.float32,
            )
        elif class_sums.shape[1] != features_cpu.shape[1]:
            raise ValueError(
                "Feature dimension changed while building prototypes: "
                f"{class_sums.shape[1]} != {features_cpu.shape[1]}."
            )

        class_sums.index_add_(0, labels_cpu, features_cpu)
        class_counts.index_add_(
            0,
            labels_cpu,
            torch.ones_like(labels_cpu, dtype=torch.long),
        )
        processed_batches += 1

    if processed_batches == 0 or class_sums is None:
        raise ValueError("Prototype construction did not process any batches.")

    missing_classes = torch.nonzero(class_counts == 0, as_tuple=False).flatten().tolist()
    if missing_classes:
        raise ValueError(f"Prototype construction did not observe classes: {missing_classes}.")

    class_means = class_sums / class_counts.to(torch.float32).unsqueeze(1)
    prototypes = F.normalize(class_means, p=2, dim=-1)
    validate_prototypes(prototypes, class_counts, num_classes=num_classes)
    return prototypes, class_counts


def validate_prototypes(prototypes, class_counts, num_classes=None):
    """Validate prototype shapes, counts, finiteness, and unit norms."""

    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        shape = getattr(prototypes, "shape", None)
        raise ValueError(f"Expected [classes, feature_dim] prototypes, got {shape}.")
    if not isinstance(class_counts, torch.Tensor) or class_counts.ndim != 1:
        shape = getattr(class_counts, "shape", None)
        raise ValueError(f"Expected [classes] class_counts, got {shape}.")
    if prototypes.shape[0] != class_counts.shape[0]:
        raise ValueError(
            "Prototype and class-count sizes differ: "
            f"{prototypes.shape[0]} != {class_counts.shape[0]}."
        )
    if num_classes is not None and prototypes.shape[0] != int(num_classes):
        raise ValueError(
            f"Expected {int(num_classes)} prototypes, got {prototypes.shape[0]}."
        )
    if prototypes.shape[1] == 0:
        raise ValueError("Prototype feature dimension must be positive.")
    if (class_counts <= 0).any():
        raise ValueError("Every prototype class must have at least one sample.")
    if not torch.isfinite(prototypes).all():
        raise ValueError("Prototype tensor contains non-finite values.")

    norms = prototypes.float().norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=_UNIT_NORM_ATOL, rtol=0.0):
        raise ValueError("Prototype vectors must have unit L2 norm.")


@torch.no_grad()
def evaluate_nearest_prototype(
    model,
    dataloader,
    prototypes,
    device,
    temperature=0.1,
    max_batches=None,
):
    """Evaluate normalized CLS features against fixed unit prototypes."""

    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    model = model.to(device)
    model.eval()
    prototypes = prototypes.to(device=device, dtype=torch.float32)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = move_to_device(batch, device)
        labels = labels.long()
        features = model.forward_features(images)
        _validate_feature_batch(features, labels)
        features = F.normalize(features.float(), p=2, dim=-1)
        if features.shape[1] != prototypes.shape[1]:
            raise ValueError(
                "Feature and prototype dimensions differ: "
                f"{features.shape[1]} != {prototypes.shape[1]}."
            )

        logits = features @ prototypes.T / temperature
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.numel()

    if total_examples == 0:
        raise ValueError("Nearest-prototype evaluation did not process any examples.")
    return {
        "loss": total_loss / total_examples,
        "acc": 100.0 * total_correct / total_examples,
        "num_examples": total_examples,
    }


def save_prototype_cache(path, prototypes, class_counts, metadata):
    """Save validated prototypes and their reproducibility metadata."""

    validate_prototypes(prototypes, class_counts)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PROTOTYPE_CACHE_VERSION,
        "prototypes": prototypes.detach().cpu(),
        "class_counts": class_counts.detach().cpu(),
        "metadata": dict(metadata),
    }
    torch.save(payload, path)


def load_prototype_cache(path, expected_metadata=None):
    """Load a prototype cache and reject incompatible provenance."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prototype cache does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required_keys = {"version", "prototypes", "class_counts", "metadata"}
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(
            f"Prototype cache is missing keys {sorted(missing_keys)}: {path}"
        )
    if payload["version"] != PROTOTYPE_CACHE_VERSION:
        raise ValueError(
            "Prototype cache version mismatch: "
            f"{payload['version']} != {PROTOTYPE_CACHE_VERSION}."
        )

    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Prototype cache metadata must be a dictionary.")
    for key, expected_value in (expected_metadata or {}).items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Prototype cache metadata mismatch for {key!r}: "
                f"{actual_value!r} != {expected_value!r}."
            )

    prototypes = payload["prototypes"]
    class_counts = payload["class_counts"]
    validate_prototypes(
        prototypes,
        class_counts,
        num_classes=metadata.get("num_classes"),
    )
    if metadata.get("feature_dim") not in {None, prototypes.shape[1]}:
        raise ValueError(
            "Prototype cache feature_dim metadata does not match the tensor: "
            f"{metadata.get('feature_dim')} != {prototypes.shape[1]}."
        )
    return prototypes, class_counts, metadata
