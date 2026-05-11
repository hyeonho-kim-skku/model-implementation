"""Build dense model sources for structured pruning.

The pruning algorithm should not care whether the dense model came from a
trained checkpoint or directly from timm pretrained weights. This module keeps
that source-specific setup in one place and returns a common PruningSource that
the pruning code can consume.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.timm_classifier import TIMMClassifier
from pruning.checkpoint import build_dense_model_from_checkpoint


@dataclass(frozen=True)
class PruningSource:
    # model: the dense TIMMClassifier instance that will be pruned.
    # model_config: minimal config needed to trace/evaluate the model later.
    # source_info: metadata explaining where the model came from.
    model: object
    model_config: dict
    source_info: dict


def _require(config, key):
    value = config.get(key)
    if value is None:
        raise ValueError(f"{key} is required for source_type={config.get('source_type')!r}.")
    return value


def _build_checkpoint_source(config, device):
    # Checkpoint sources are expected to be training checkpoints that contain a
    # merged dense state. For LoRA checkpoints, this removes LoRA wrappers before
    # pruning so Torch-Pruning sees plain nn.Linear modules.
    checkpoint_path = _require(config, "checkpoint_path")
    checkpoint, model = build_dense_model_from_checkpoint(checkpoint_path, map_location=device)
    return PruningSource(
        model=model,
        model_config=checkpoint["model_config"],
        source_info={
            "type": "checkpoint",
            "path": checkpoint_path,
            "checkpoint_meta": {
                "acc": checkpoint.get("acc"),
                "epoch": checkpoint.get("epoch"),
                "model_config": checkpoint.get("model_config"),
            },
        },
    )


def _build_timm_source(config):
    # Direct timm sources start from pretrained backbone weights and a freshly
    # initialized classifier. The classifier is included so the object has the
    # same TIMMClassifier shape as checkpoint-derived sources, but its accuracy
    # is not meaningful until a downstream probing/recovery step trains it.
    backbone_name = _require(config, "backbone_name")
    num_classes = _require(config, "num_classes")
    img_size = config.get("img_size")
    pretrained = config.get("pretrained", True)

    model = TIMMClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        img_size=img_size,
        freeze_encoder=False,
    )
    return PruningSource(
        model=model,
        model_config=model.export_config(),
        source_info={
            "type": "timm",
            "path": None,
            "checkpoint_meta": None,
            "backbone_name": backbone_name,
            "pretrained": pretrained,
        },
    )


def build_pruning_source(config, device="cpu"):
    # source_type is intentionally required. Keeping it explicit prevents old
    # configs from silently taking a different path than the experiment intended.
    source_type = config.get("source_type")
    if source_type is None:
        raise ValueError("source_type is required. Use 'checkpoint' or 'timm'.")

    source_type = source_type.lower()
    if source_type == "checkpoint":
        return _build_checkpoint_source(config, device=device)
    if source_type == "timm":
        return _build_timm_source(config)

    raise ValueError(f"Unsupported source_type: {source_type!r}.")
