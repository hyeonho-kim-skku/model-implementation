"""Scoring objectives for progressive pruning.

The CE baseline is implemented first. Fixed-prototype contrastive scoring is
intentionally stubbed until its prototype construction and cache are added.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from datasets import get_loader
from progressive_pruning.representation import (
    build_class_prototypes,
    evaluate_nearest_prototype,
    load_prototype_cache,
    save_prototype_cache,
)


@dataclass(frozen=True)
class ObjectiveConfig:
    """Lightweight objective descriptor used by the progressive pipeline."""

    name: str


class CrossEntropyObjective:
    """Classifier-aware CE scoring objective for the baseline pipeline."""

    name = "ce"

    # Delegates to the existing pruning.compute_taylor_gradients CE path.
    calibration_objective = "ce"

    def setup(self, source, device):
        return None

    def metadata(self):
        return {
            "objective": self.name,
            "description": "cross_entropy_on_trained_classifier",
        }


class FixedPrototypeContrastiveObjective:
    """Score normalized CLS features against fixed dense-source prototypes."""

    name = "prototype_contrastive"

    def __init__(self, config):
        self.config = dict(config or {})
        self.temperature = float(self.config.get("prototype_temperature", 0.1))
        if self.temperature <= 0.0:
            raise ValueError(
                f"prototype_temperature must be positive, got {self.temperature}."
            )
        self.prototypes = None
        self.prototype_metadata = None
        self.cache_loaded = None

    def _dataset(self):
        dataset = (
            self.config.get("prototype_dataset")
            or self.config.get("calibration_dataset")
            or self.config.get("dataset")
        )
        if not dataset:
            raise ValueError("prototype_dataset or dataset is required.")
        return dataset

    def _num_classes(self, source):
        value = source.model_config.get("num_classes")
        if value is None and hasattr(source.model, "classifier"):
            value = source.model.classifier.out_features
        if value is None:
            raise ValueError("Cannot determine num_classes for prototype construction.")
        return int(value)

    def _feature_dim(self, source):
        value = getattr(source.model.encoder, "num_features", None)
        if value is None and hasattr(source.model, "classifier"):
            value = source.model.classifier.in_features
        if value is None:
            raise ValueError("Cannot determine final feature dimension for prototypes.")
        return int(value)

    def _expected_metadata(self, source):
        return {
            "dataset": self._dataset(),
            "split": self.config.get("prototype_split", "train"),
            "transform_mode": "test",
            "source_type": source.source_info.get("type"),
            "source_path": source.source_info.get("path"),
            "backbone_name": source.model_config.get("backbone_name"),
            "num_classes": self._num_classes(source),
            "feature_dim": self._feature_dim(source),
            "normalization": "sample_l2_then_class_mean_l2",
        }

    def _make_loader(self, dataset, split, batch_size, shuffle=False):
        if split not in {"train", "test"}:
            raise ValueError(f"Prototype split must be 'train' or 'test', got {split!r}.")
        return get_loader(
            dataset,
            batch_size,
            mode="test",
            train=(split == "train"),
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.config.get("num_workers", 4),
            data_root=self.config.get("data_root", "./data"),
        )

    def setup(self, source, device):
        cache_path = self.config.get("prototype_cache_path")
        if not cache_path:
            raise ValueError("prototype_cache_path is required for prototype scoring.")

        expected_metadata = self._expected_metadata(source)
        if os.path.isfile(cache_path):
            prototypes, class_counts, metadata = load_prototype_cache(
                cache_path,
                expected_metadata=expected_metadata,
            )
            cache_loaded = True
        else:
            batch_size = int(
                self.config.get("prototype_batch_size")
                or self.config.get("calibration_batch_size")
                or self.config.get("batch_size", 64)
            )
            prototype_loader = self._make_loader(
                expected_metadata["dataset"],
                expected_metadata["split"],
                batch_size,
                shuffle=False,
            )
            prototypes, class_counts = build_class_prototypes(
                source.model,
                prototype_loader,
                num_classes=expected_metadata["num_classes"],
                device=device,
            )
            metadata = dict(expected_metadata)

            eval_split = self.config.get("prototype_eval_split", "test")
            if eval_split:
                eval_loader = self._make_loader(
                    expected_metadata["dataset"],
                    eval_split,
                    batch_size,
                    shuffle=False,
                )
                metadata["nearest_prototype_evaluation"] = {
                    "split": eval_split,
                    "temperature": self.temperature,
                    "metrics": evaluate_nearest_prototype(
                        source.model,
                        eval_loader,
                        prototypes,
                        device=device,
                        temperature=self.temperature,
                    ),
                }

            save_prototype_cache(cache_path, prototypes, class_counts, metadata)
            cache_loaded = False

        metadata = dict(metadata)
        metadata.pop("class_counts", None)
        metadata["class_count_summary"] = {
            "total_examples": int(class_counts.sum().item()),
            "min_per_class": int(class_counts.min().item()),
            "max_per_class": int(class_counts.max().item()),
        }
        self.prototypes = prototypes.to(device=device, dtype=torch.float32).detach()
        self.prototypes.requires_grad_(False)
        self.prototype_metadata = metadata
        self.cache_loaded = cache_loaded

    def loss(self, model, images, labels):
        if self.prototypes is None:
            raise RuntimeError("Prototype objective must be set up before scoring.")
        features = model.forward_features(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            shape = getattr(features, "shape", None)
            raise ValueError(f"Expected [batch, feature_dim] features, got {shape}.")
        if features.shape[1] != self.prototypes.shape[1]:
            raise ValueError(
                "Feature and prototype dimensions differ: "
                f"{features.shape[1]} != {self.prototypes.shape[1]}."
            )

        features = F.normalize(features.float(), p=2, dim=-1)
        logits = features @ self.prototypes.T / self.temperature
        return F.cross_entropy(logits, labels.long(), reduction="sum")

    def metadata(self):
        if self.prototype_metadata is None:
            raise RuntimeError("Prototype objective metadata is unavailable before setup.")
        return {
            "objective": self.name,
            "description": "fixed_dense_class_prototype_cosine_ce",
            "temperature": self.temperature,
            "cache_path": self.config.get("prototype_cache_path"),
            "cache_loaded": self.cache_loaded,
            "prototype": dict(self.prototype_metadata),
        }


def build_objective(name, config=None):
    """Build a scoring objective by name."""

    normalized = (name or "ce").strip().lower()
    if normalized in {"ce", "baseline", "cross_entropy"}:
        return CrossEntropyObjective()
    if normalized in {"prototype_contrastive", "prototype", "representation"}:
        return FixedPrototypeContrastiveObjective(config)
    raise ValueError(f"Unsupported progressive pruning objective: {name!r}.")
