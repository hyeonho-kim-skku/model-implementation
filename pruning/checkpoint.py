"""Checkpoint loading helpers for structured pruning.

This module reconstructs a plain TIMMClassifier from a timm_lora checkpoint by
loading the checkpoint's merged dense weights. The reconstructed model no longer
contains LoRA wrapper modules, which makes it suitable for pruning utilities
that operate on standard nn.Linear layers.
"""

import torch

from models.timm_classifier import TIMMClassifier


def load_checkpoint(checkpoint_path, map_location="cpu"):
    # Trainer가 저장한 .pth checkpoint를 읽는다.
    # map_location은 GPU에서 저장된 파일을 CPU나 특정 GPU로 안전하게 불러오기 위해 사용한다.
    return torch.load(checkpoint_path, map_location=map_location)


def build_dense_model_from_checkpoint(checkpoint_path, map_location="cpu"):
    # LoRA checkpoint 안에는 LoRA wrapper가 남아 있는 state_dict("model")와
    # LoRA를 원래 Linear weight에 합친 dense state("merged_model")가 함께 저장된다.
    # pruning은 일반 nn.Linear 구조에서 하는 편이 안전하므로 merged_model을 사용한다.
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)

    if "model_config" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_config.")
    if "merged_model" not in checkpoint:
        raise ValueError("Checkpoint does not contain merged_model.")

    model_config = checkpoint["model_config"]
    merged_model = checkpoint["merged_model"]

    if model_config.get("model") != "timm_lora":
        raise ValueError("Only timm_lora checkpoints are currently supported for dense reconstruction.")

    # LoRA wrapper가 없는 일반 TIMMClassifier를 같은 backbone/classifier shape으로 만든다.
    # pretrained=False인 이유는 곧 merged_model weight를 strict=True로 덮어쓸 것이기 때문이다.
    dense_model = TIMMClassifier(
        backbone_name=model_config["backbone_name"],
        num_classes=model_config["num_classes"],
        pretrained=False,
        img_size=model_config.get("img_size"),
    )
    # merged_model은 encoder/classifier로 나뉘어 저장되어 있으므로,
    # TIMMClassifier의 state_dict key 형식에 맞춰 encoder.* / classifier.* prefix를 붙인다.
    dense_model.load_state_dict(
        {
            **{f"encoder.{name}": tensor for name, tensor in merged_model["encoder"].items()},
            **{f"classifier.{name}": tensor for name, tensor in merged_model["classifier"].items()},
        },
        strict=True,
    )
    # 원본 checkpoint metadata와 pruning 가능한 dense model을 함께 반환한다.
    return checkpoint, dense_model
