"""Scoring objectives for progressive pruning.

The CE baseline is implemented first. Prototype/SupCon is intentionally stubbed
until its prototype construction and cache are added.
"""

from __future__ import annotations

from dataclasses import dataclass


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


class PrototypeSupConObjective:
    """Planned representation-aware scoring objective."""

    name = "prototype_supcon"

    # TODO(progressive_pruning): implement prototype construction/cache and a
    # forward_features-based supervised contrastive loss.
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PrototypeSupConObjective is planned but not implemented yet."
        )


def build_objective(name):
    """Build a scoring objective by name."""

    normalized = (name or "ce").strip().lower()
    if normalized in {"ce", "baseline", "cross_entropy"}:
        return CrossEntropyObjective()
    if normalized in {"prototype_supcon", "supcon", "representation"}:
        return PrototypeSupConObjective()
    raise ValueError(f"Unsupported progressive pruning objective: {name!r}.")
