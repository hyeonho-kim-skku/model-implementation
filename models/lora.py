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

        # LoRA는 full weight update를 직접 학습하지 않고,
        # in_features -> rank -> out_features인 작은 두 Linear의 곱으로 update를 만든다.
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        # B를 0으로 시작하면 학습 시작 시점의 LoRA 출력은 0이다.
        # 즉, 처음에는 frozen base model의 출력과 완전히 같은 상태에서 시작한다.
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        if self.lora_b.bias is not None:
            nn.init.zeros_(self.lora_b.bias)

    def forward(self, x):
        # x: [..., in_features]
        # output: [..., out_features], 기존 Linear 출력에 더할 low-rank residual.
        return self.lora_b(self.lora_a(x)) * self.scaling

    def merged_weight(self):
        # 두 low-rank weight를 곱해 원래 Linear weight와 같은 shape의 update matrix로 만든다.
        return torch.matmul(self.lora_b.weight, self.lora_a.weight) * self.scaling


class LoRAWrappedLinear(nn.Module):
    """Wrap a frozen linear layer with a trainable LoRA residual branch."""

    def __init__(self, linear, rank, alpha=None):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("LoRAWrappedLinear expects an nn.Linear module.")

        # 기존 pretrained Linear는 그대로 보관하고 freeze한다.
        self.linear = linear
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

        # frozen Linear와 같은 입력/출력 shape을 가지는 LoRA residual branch를 추가한다.
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
        # 원래 frozen Linear 출력에 trainable LoRA update를 더한다.
        return self.linear(x) + self.lora(x)

    def to_merged_linear(self):
        # export/pruning용: frozen weight와 LoRA update를 합쳐 일반 nn.Linear 하나로 되돌린다.
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

        # timm ViT attention은 q, k, v를 따로 Linear로 만들지 않고,
        # 하나의 Linear가 [q, k, v]를 이어 붙인 출력을 만드는 경우가 많다.
        self.qkv = qkv
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.component_dim = qkv.out_features // 3

        # 원래 qkv projection은 freeze하고, 아래에서 만든 LoRA branch만 학습한다.
        for parameter in self.qkv.parameters():
            parameter.requires_grad = False

        normalized_components = []
        for component in target_components:
            key = component.lower()
            if key not in self._COMPONENT_INDEX:
                raise ValueError(f"Unsupported qkv component: {component}")
            normalized_components.append(key)

        # q/k/v 중 target_components에 들어온 component에만 별도 LoRA adapter를 만든다.
        # 예를 들어 ("q", "v")면 q slice와 v slice에만 low-rank residual을 더한다.
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
        # 원래 frozen qkv projection을 먼저 계산한다. 마지막 차원은 [q | k | v] 순서다.
        qkv = self.qkv(x)
        for component, adapter in self.adapters.items():
            start = self._COMPONENT_INDEX[component] * self.component_dim
            end = start + self.component_dim
            # 선택된 q/k/v slice에만 LoRA residual을 더한다.
            qkv[..., start:end] = qkv[..., start:end] + adapter(x)
        return qkv

    def to_merged_linear(self):
        # pruning/export 때 쓰기 좋도록 frozen qkv weight에 LoRA weight를 더해
        # 다시 하나의 nn.Linear로 합친다.
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


class RaggedFusedQKVLoRA(nn.Module):
    """LoRA wrapper for RaggedFusedQKVAttention qkv projections."""

    _COMPONENT_INDEX = {
        "q": 0,
        "k": 1,
        "v": 2,
    }

    def __init__(self, qkv, qk_width, v_width, rank, alpha=None, target_components=("q", "v")):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError("RaggedFusedQKVLoRA expects an nn.Linear qkv projection.")
        expected_out = 2 * int(qk_width) + int(v_width)
        if qkv.out_features != expected_out:
            raise ValueError(
                "RaggedFusedQKVLoRA qkv width mismatch: "
                f"{qkv.out_features} != 2 * {qk_width} + {v_width}."
            )
        self.qkv = qkv
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.qk_width = int(qk_width)
        self.v_width = int(v_width)

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
                    out_features=self._component_width(component),
                    rank=rank,
                    alpha=alpha,
                    bias=False,
                )
                for component in self.target_components
            }
        )

    def _component_width(self, component):
        if component in {"q", "k"}:
            return self.qk_width
        if component == "v":
            return self.v_width
        raise ValueError(f"Unsupported qkv component: {component}")

    def _component_range(self, component):
        if component == "q":
            return 0, self.qk_width
        if component == "k":
            return self.qk_width, 2 * self.qk_width
        if component == "v":
            return 2 * self.qk_width, 2 * self.qk_width + self.v_width
        raise ValueError(f"Unsupported qkv component: {component}")

    def forward(self, x):
        qkv = self.qkv(x)
        for component, adapter in self.adapters.items():
            start, end = self._component_range(component)
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
            start, end = self._component_range(component)
            merged_qkv.weight.data[start:end] += adapter.merged_weight().to(self.qkv.weight.dtype)
        return merged_qkv
