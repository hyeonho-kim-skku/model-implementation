from collections.abc import Iterable
import copy

import timm
import torch
import torch.nn as nn

from .lora import FusedQKVLoRA, LoRAWrappedLinear


DEFAULT_LORA_MODULES = ("qkv",)
VALID_LORA_MODULES = {"qkv", "proj", "mlp"}


def _normalize_csv_or_iterable(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Iterable):
        return tuple(value)
    raise TypeError("Expected None, a comma-separated string, or an iterable.")


def _normalize_qkv_lora_components(qkv_lora_components):
    return _normalize_csv_or_iterable(qkv_lora_components, ("q", "v"))


def _normalize_lora_modules(lora_modules):
    normalized_modules = tuple(item.lower() for item in _normalize_csv_or_iterable(lora_modules, DEFAULT_LORA_MODULES))
    invalid_modules = set(normalized_modules) - VALID_LORA_MODULES
    if invalid_modules:
        raise ValueError(f"Unsupported LoRA modules: {sorted(invalid_modules)}")
    return normalized_modules


def _set_child_module(parent_module, child_name, new_module):
    setattr(parent_module, child_name, new_module)


def count_parameters(module):
    total_params = 0
    trainable_params = 0
    for parameter in module.parameters():
        num_params = parameter.numel()
        total_params += num_params
        if parameter.requires_grad:
            trainable_params += num_params
    return trainable_params, total_params


def inject_lora_into_vit(
    encoder,
    rank,
    alpha=None,
    qkv_lora_components=("q", "v"),
    lora_modules=("qkv",),
):
    if not hasattr(encoder, "blocks"):
        raise ValueError("This timm model does not expose transformer blocks for LoRA injection.")

    injected_module_names = []
    normalized_components = _normalize_qkv_lora_components(qkv_lora_components)
    normalized_modules = _normalize_lora_modules(lora_modules)

    for block_idx, block in enumerate(encoder.blocks):
        if "qkv" in normalized_modules:
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

        if "proj" in normalized_modules:
            if not hasattr(block, "attn") or not hasattr(block.attn, "proj"):
                raise ValueError(
                    f"Block {block_idx} does not expose attn.proj, so attention projection LoRA is not supported."
                )

            _set_child_module(
                block.attn,
                "proj",
                LoRAWrappedLinear(block.attn.proj, rank=rank, alpha=alpha),
            )
            injected_module_names.append(f"blocks.{block_idx}.attn.proj")

        if "mlp" in normalized_modules:
            if not hasattr(block, "mlp") or not hasattr(block.mlp, "fc1") or not hasattr(block.mlp, "fc2"):
                raise ValueError(
                    f"Block {block_idx} does not expose mlp.fc1/fc2, so MLP LoRA injection is not supported."
                )

            _set_child_module(
                block.mlp,
                "fc1",
                LoRAWrappedLinear(block.mlp.fc1, rank=rank, alpha=alpha),
            )
            injected_module_names.append(f"blocks.{block_idx}.mlp.fc1")

            _set_child_module(
                block.mlp,
                "fc2",
                LoRAWrappedLinear(block.mlp.fc2, rank=rank, alpha=alpha),
            )
            injected_module_names.append(f"blocks.{block_idx}.mlp.fc2")

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
        qkv_lora_components=("q", "v"),
        lora_modules=("qkv",),
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
        self.img_size = img_size
        self.lora_rank = rank
        self.lora_alpha = lora_alpha
        self.qkv_lora_components = _normalize_qkv_lora_components(qkv_lora_components)
        self.lora_modules = _normalize_lora_modules(lora_modules)
        self.pretrained = pretrained
        self.encoder = timm.create_model(backbone_name, **create_model_kwargs)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.injected_module_names = inject_lora_into_vit(
            self.encoder,
            rank=rank,
            alpha=lora_alpha,
            qkv_lora_components=self.qkv_lora_components,
            lora_modules=self.lora_modules,
        )

        feature_dim = getattr(self.encoder, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"{backbone_name} does not expose encoder.num_features.")

        self.classifier = nn.Linear(feature_dim, num_classes)
        trainable_params, total_params = count_parameters(self)
        self.trainable_params = trainable_params
        self.total_params = total_params

        print(f"[TIMMLoRA] backbone: {self.backbone_name}")
        print(f"[TIMMLoRA] injected modules ({len(self.injected_module_names)}):")
        for module_name in self.injected_module_names:
            print(f"  - {module_name}")
        print(
            f"[TIMMLoRA] trainable params: {self.trainable_params:,} / "
            f"{self.total_params:,} ({100.0 * self.trainable_params / self.total_params:.2f}%)"
        )

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

    def export_config(self):
        return {
            "model": "timm_lora",
            "backbone_name": self.backbone_name,
            "img_size": self.img_size,
            "num_classes": self.classifier.out_features,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_modules": list(self.lora_modules),
            "qkv_lora_components": list(self.qkv_lora_components),
            "pretrained": self.pretrained,
        }

    def _build_merged_encoder(self):
        merged_encoder = copy.deepcopy(self.encoder)
        for block in merged_encoder.blocks:
            if isinstance(block.attn.qkv, FusedQKVLoRA):
                block.attn.qkv = block.attn.qkv.to_merged_linear()
            if isinstance(block.attn.proj, LoRAWrappedLinear):
                block.attn.proj = block.attn.proj.to_merged_linear()
            if isinstance(block.mlp.fc1, LoRAWrappedLinear):
                block.mlp.fc1 = block.mlp.fc1.to_merged_linear()
            if isinstance(block.mlp.fc2, LoRAWrappedLinear):
                block.mlp.fc2 = block.mlp.fc2.to_merged_linear()
        return merged_encoder

    def export_merged_state(self):
        merged_encoder = self._build_merged_encoder()
        return {
            "encoder": merged_encoder.state_dict(),
            "classifier": self.classifier.state_dict(),
        }
