import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank residual branch for a single linear projection."""

    def __init__(self, in_features, out_features, rank, alpha=None, bias=False):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be greater than 0.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        if self.lora_b.bias is not None:
            nn.init.zeros_(self.lora_b.bias)

    def forward(self, x):
        return self.lora_b(self.lora_a(x)) * self.scaling

    def merged_weight(self):
        return torch.matmul(self.lora_b.weight, self.lora_a.weight) * self.scaling


class LoRAWrappedLinear(nn.Module):
    """Wrap a frozen linear layer with a trainable LoRA residual branch."""

    def __init__(self, linear, rank, alpha=None):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("LoRAWrappedLinear expects an nn.Linear module.")

        self.linear = linear
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

        self.lora = LoRALinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=False,
        )

        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    def to_merged_linear(self):
        merged_linear = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
        )
        merged_linear.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype)
        merged_linear.weight.data.copy_(self.linear.weight.data + self.lora.merged_weight().to(self.linear.weight.dtype))
        if self.linear.bias is not None:
            merged_linear.bias.data.copy_(self.linear.bias.data)
        return merged_linear


class FusedQKVLoRA(nn.Module):
    """LoRA wrapper for fused qkv projections used by timm ViT-style attention."""

    _COMPONENT_INDEX = {
        "q": 0,
        "k": 1,
        "v": 2,
    }

    def __init__(self, qkv, rank, alpha=None, target_components=("q", "v")):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError("FusedQKVLoRA expects an nn.Linear qkv projection.")
        if qkv.out_features % 3 != 0:
            raise ValueError("FusedQKVLoRA expects qkv.out_features to be divisible by 3.")

        self.qkv = qkv
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.component_dim = qkv.out_features // 3

        for parameter in self.qkv.parameters():
            parameter.requires_grad = False

        normalized_components = []
        for component in target_components:
            key = component.lower()
            if key not in self._COMPONENT_INDEX:
                raise ValueError(f"Unsupported qkv component: {component}")
            normalized_components.append(key)

        self.target_components = tuple(normalized_components)
        self.adapters = nn.ModuleDict(
            {
                component: LoRALinear(
                    in_features=self.in_features,
                    out_features=self.component_dim,
                    rank=rank,
                    alpha=alpha,
                    bias=False,
                )
                for component in self.target_components
            }
        )

    def forward(self, x):
        qkv = self.qkv(x)
        for component, adapter in self.adapters.items():
            start = self._COMPONENT_INDEX[component] * self.component_dim
            end = start + self.component_dim
            qkv[..., start:end] = qkv[..., start:end] + adapter(x)
        return qkv

    def to_merged_linear(self):
        merged_qkv = nn.Linear(
            self.qkv.in_features,
            self.qkv.out_features,
            bias=self.qkv.bias is not None,
        )
        merged_qkv.to(device=self.qkv.weight.device, dtype=self.qkv.weight.dtype)
        merged_qkv.weight.data.copy_(self.qkv.weight.data)
        if self.qkv.bias is not None:
            merged_qkv.bias.data.copy_(self.qkv.bias.data)

        for component, adapter in self.adapters.items():
            start = self._COMPONENT_INDEX[component] * self.component_dim
            end = start + self.component_dim
            merged_qkv.weight.data[start:end] += adapter.merged_weight().to(self.qkv.weight.dtype)

        return merged_qkv
