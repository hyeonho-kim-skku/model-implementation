"""Ragged fused-QKV attention for head-dimension pruning experiments.

The module keeps a single fused qkv Linear, but allows each attention head to
retain a different number of Q/K and V dimensions. It is intended for pruned
timm ViT attention blocks and supports the current ViT-Base LoRA experiments
where q_norm, k_norm, and scale_norm are Identity modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RaggedFusedQKVAttention(nn.Module):
    """Fused-QKV attention with per-head QK/V dimensions.

    qkv output layout is:
      [all Q heads | all K heads | all V heads]

    Q and K always share the same per-head dimensions. V dimensions may differ
    from Q/K dimensions. The attention output concatenates all V heads and feeds
    that tensor to ``proj``.
    """

    is_ragged_fused_qkv_attention = True

    def __init__(
        self,
        qkv,
        proj,
        *,
        num_heads,
        qk_dim_indices,
        v_dim_indices,
        attn_drop=None,
        proj_drop=None,
        original_head_dim=None,
        fused_attn=False,
    ):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError("RaggedFusedQKVAttention expects qkv to be nn.Linear.")
        if not isinstance(proj, nn.Linear):
            raise TypeError("RaggedFusedQKVAttention expects proj to be nn.Linear.")

        self.qkv = qkv
        self.proj = proj
        self.num_heads = int(num_heads)
        self.qk_dim_indices = _normalize_dim_indices(qk_dim_indices, self.num_heads)
        self.v_dim_indices = _normalize_dim_indices(v_dim_indices, self.num_heads)
        self.original_head_dim = int(
            original_head_dim
            if original_head_dim is not None
            else max(max(items, default=-1) for items in self.qk_dim_indices) + 1
        )
        self.attn_drop = attn_drop if attn_drop is not None else nn.Dropout(0.0)
        self.proj_drop = proj_drop if proj_drop is not None else nn.Dropout(0.0)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.norm = nn.Identity()
        # Ragged QK/V dims can make PyTorch's fused SDPA backends reject a
        # shape even when the math is valid. Use the explicit attention path for
        # correctness and predictable pruning experiments.
        self.fused_attn = False
        self._qk_gate_hook = None
        self._qkv_gate_hook = None

        self._refresh_metadata()
        self._validate_shapes()

    @classmethod
    def from_timm_attention(cls, attn):
        """Convert a dense timm ViT Attention module into the ragged form."""

        if not isinstance(getattr(attn, "q_norm", nn.Identity()), nn.Identity):
            raise ValueError("Ragged attention v1 only supports Identity q_norm.")
        if not isinstance(getattr(attn, "k_norm", nn.Identity()), nn.Identity):
            raise ValueError("Ragged attention v1 only supports Identity k_norm.")
        if not isinstance(getattr(attn, "norm", nn.Identity()), nn.Identity):
            raise ValueError("Ragged attention v1 only supports Identity scale norm.")

        head_dim = int(attn.head_dim)
        num_heads = int(attn.num_heads)
        dim_indices = [list(range(head_dim)) for _ in range(num_heads)]
        return cls(
            qkv=attn.qkv,
            proj=attn.proj,
            num_heads=num_heads,
            qk_dim_indices=dim_indices,
            v_dim_indices=dim_indices,
            attn_drop=attn.attn_drop,
            proj_drop=attn.proj_drop,
            original_head_dim=head_dim,
            fused_attn=getattr(attn, "fused_attn", False),
        )

    def _refresh_metadata(self):
        self.qk_head_dims = [len(items) for items in self.qk_dim_indices]
        self.v_head_dims = [len(items) for items in self.v_dim_indices]
        self.qk_width = int(sum(self.qk_head_dims))
        self.v_width = int(sum(self.v_head_dims))
        self.attn_dim = self.v_width
        self.head_dim = self.original_head_dim
        qk_scale = self.original_head_dim ** -0.5
        self.scale = [qk_scale for _ in self.qk_head_dims]
        self.q_offsets = _offsets(self.qk_head_dims, start=0)
        self.k_offsets = _offsets(self.qk_head_dims, start=self.qk_width)
        self.v_offsets = _offsets(self.v_head_dims, start=2 * self.qk_width)

    def _validate_shapes(self):
        if len(self.qk_dim_indices) != self.num_heads:
            raise ValueError("qk_dim_indices length must match num_heads.")
        if len(self.v_dim_indices) != self.num_heads:
            raise ValueError("v_dim_indices length must match num_heads.")
        if any(dim <= 0 for dim in self.qk_head_dims):
            raise ValueError("Every head must retain at least one Q/K dim.")
        if any(dim <= 0 for dim in self.v_head_dims):
            raise ValueError("Every head must retain at least one V dim.")
        expected_qkv_out = 2 * self.qk_width + self.v_width
        if self.qkv.out_features != expected_qkv_out:
            raise ValueError(
                "qkv.out_features does not match ragged metadata: "
                f"{self.qkv.out_features} != 2 * {self.qk_width} + {self.v_width}."
            )
        if self.proj.in_features != self.v_width:
            raise ValueError(
                "proj.in_features does not match V width: "
                f"{self.proj.in_features} != {self.v_width}."
            )

    def qkv_component_ranges(self):
        """Return contiguous ranges for q, k, and v components."""

        return {
            "q": (0, self.qk_width),
            "k": (self.qk_width, 2 * self.qk_width),
            "v": (2 * self.qk_width, 2 * self.qk_width + self.v_width),
        }

    def head_component_ranges(self, component):
        if component == "q":
            return self.q_offsets
        if component == "k":
            return self.k_offsets
        if component == "v":
            return self.v_offsets
        raise ValueError(f"Unsupported component: {component!r}.")

    def export_ragged_metadata(self):
        return {
            "num_heads": self.num_heads,
            "original_head_dim": self.original_head_dim,
            "qk_head_dims": list(self.qk_head_dims),
            "v_head_dims": list(self.v_head_dims),
            "qk_dim_indices": [list(items) for items in self.qk_dim_indices],
            "v_dim_indices": [list(items) for items in self.v_dim_indices],
            "qkv_shape": (self.qkv.in_features, self.qkv.out_features),
            "proj_shape": (self.proj.in_features, self.proj.out_features),
        }

    def forward(self, x, attn_mask=None, is_causal=False):
        if attn_mask is not None:
            raise ValueError("RaggedFusedQKVAttention v1 does not support attn_mask.")
        if is_causal:
            raise ValueError("RaggedFusedQKVAttention v1 does not support causal attention.")

        batch_size, tokens, _channels = x.shape
        qkv = self.qkv(x)
        if self._can_use_regular_forward():
            x = self._forward_regular(qkv, batch_size, tokens)
        else:
            x = self._forward_ragged(qkv, batch_size, tokens)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _can_use_regular_forward(self):
        if self._qk_gate_hook is not None or self._qkv_gate_hook is not None:
            return False
        return len(set(self.qk_head_dims)) == 1 and len(set(self.v_head_dims)) == 1

    def _forward_regular(self, qkv, batch_size, tokens):
        qk_dim = self.qk_head_dims[0]
        v_dim = self.v_head_dims[0]
        q_end = self.qk_width
        k_end = 2 * self.qk_width
        q = qkv[..., :q_end].reshape(batch_size, tokens, self.num_heads, qk_dim)
        k = qkv[..., q_end:k_end].reshape(batch_size, tokens, self.num_heads, qk_dim)
        v = qkv[..., k_end:].reshape(batch_size, tokens, self.num_heads, v_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * float(self.scale[0])
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        return x.transpose(1, 2).reshape(batch_size, tokens, self.v_width)

    def _forward_ragged(self, qkv, batch_size, tokens):
        outputs = []
        for head_idx in range(self.num_heads):
            q_start, q_end = self.q_offsets[head_idx]
            k_start, k_end = self.k_offsets[head_idx]
            v_start, v_end = self.v_offsets[head_idx]

            q = qkv[..., q_start:q_end].unsqueeze(1)
            k = qkv[..., k_start:k_end].unsqueeze(1)
            v = qkv[..., v_start:v_end].unsqueeze(1)

            if self._qk_gate_hook is not None:
                q, k = self._qk_gate_hook(self, head_idx, q, k)
            if self._qkv_gate_hook is not None:
                q, k, v = self._qkv_gate_hook(self, head_idx, q, k, v)

            if self.fused_attn:
                head_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                q = q * float(self.scale[head_idx])
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                head_out = attn @ v
            outputs.append(head_out.squeeze(1))

        return torch.cat(outputs, dim=-1).reshape(batch_size, tokens, self.v_width)


def convert_vit_attention_to_ragged(encoder):
    """Replace timm Attention blocks with RaggedFusedQKVAttention modules."""

    if not hasattr(encoder, "blocks"):
        raise ValueError("Expected an encoder with transformer blocks.")
    converted = []
    for block_idx, block in enumerate(encoder.blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        if getattr(attn, "is_ragged_fused_qkv_attention", False):
            continue
        block.attn = RaggedFusedQKVAttention.from_timm_attention(attn)
        converted.append(block_idx)
    return converted


def _normalize_dim_indices(dim_indices, num_heads):
    if len(dim_indices) != int(num_heads):
        raise ValueError("dim_indices length must match num_heads.")
    normalized = []
    for items in dim_indices:
        values = [int(item) for item in items]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate dim indices are not allowed: {values}.")
        if any(value < 0 for value in values):
            raise ValueError(f"Dim indices must be non-negative: {values}.")
        normalized.append(values)
    return normalized


def _offsets(lengths, start):
    offsets = []
    cursor = int(start)
    for length in lengths:
        offsets.append((cursor, cursor + int(length)))
        cursor += int(length)
    return offsets
