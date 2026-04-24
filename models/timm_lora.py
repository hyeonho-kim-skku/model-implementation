from collections.abc import Iterable

import timm
import torch
import torch.nn as nn

from .lora import FusedQKVLoRA


def _normalize_components(target_components):
    if target_components is None:
        return ("q", "v")
    if isinstance(target_components, str):
        return tuple(component.strip() for component in target_components.split(",") if component.strip())
    if isinstance(target_components, Iterable):
        return tuple(target_components)
    raise TypeError("target_components must be None, a comma-separated string, or an iterable.")


def inject_lora_into_vit(
    encoder,
    rank,
    alpha=None,
    target_components=("q", "v"),
):
    if not hasattr(encoder, "blocks"):
        raise ValueError("This timm model does not expose transformer blocks for LoRA injection.")

    injected_module_names = []
    normalized_components = _normalize_components(target_components)

    for block_idx, block in enumerate(encoder.blocks):
        if not hasattr(block, "attn") or not hasattr(block.attn, "qkv"):
            raise ValueError(
                f"Block {block_idx} does not expose attn.qkv, so fused-qkv LoRA injection is not supported."
            )

        block.attn.qkv = FusedQKVLoRA(
            qkv=block.attn.qkv,
            rank=rank,
            alpha=alpha,
            target_components=normalized_components,
        )
        injected_module_names.append(f"blocks.{block_idx}.attn.qkv")

    return injected_module_names


class TIMMLoRA(nn.Module):
    def __init__(
        self,
        backbone_name,
        num_classes,
        rank=4,
        pretrained=True,
        img_size=None,
        lora_alpha=None,
        lora_components=("q", "v"),
    ):
        super().__init__()
        if backbone_name is None:
            raise ValueError("backbone_name must be provided for model='timm_lora'.")
        if num_classes is None:
            raise ValueError("num_classes must be provided for model='timm_lora'.")
        if rank <= 0:
            raise ValueError("rank must be greater than 0 for LoRA.")

        create_model_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
        }
        if img_size is not None:
            create_model_kwargs["img_size"] = img_size

        self.backbone_name = backbone_name
        self.encoder = timm.create_model(backbone_name, **create_model_kwargs)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.injected_module_names = inject_lora_into_vit(
            self.encoder,
            rank=rank,
            alpha=lora_alpha,
            target_components=lora_components,
        )

        feature_dim = getattr(self.encoder, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"{backbone_name} does not expose encoder.num_features.")

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x):
        features = self.encoder.forward_features(x)

        if hasattr(self.encoder, "forward_head"):
            pooled = self.encoder.forward_head(features, pre_logits=True)
            if pooled.ndim == 2:
                return pooled

        if isinstance(features, torch.Tensor) and features.ndim == 3:
            return features[:, 0]
        return features

    def forward(self, x):
        features = self.forward_features(x)
        return self.classifier(features)
