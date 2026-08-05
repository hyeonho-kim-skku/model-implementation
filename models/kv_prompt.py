"""Learnable key-value prompts for structurally pruned timm attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVPromptedAttention(nn.Module):
    """Wrap timm ViT attention with a shared learnable K/V prompt.

    The prompt lives directly in the projected per-head attention space.  It is
    prepended to K and V, while Q and the output sequence length are unchanged.
    A single parameter is shared by K and V, matching the default E2VPT setup.
    """

    def __init__(self, attention, num_prompt_tokens):
        super().__init__()
        num_prompt_tokens = int(num_prompt_tokens)
        if num_prompt_tokens <= 0:
            raise ValueError("num_prompt_tokens must be greater than 0.")

        required = (
            "qkv",
            "q_norm",
            "k_norm",
            "attn_drop",
            "norm",
            "proj",
            "proj_drop",
            "num_heads",
            "head_dim",
            "attn_dim",
            "scale",
            "fused_attn",
        )
        missing = [name for name in required if not hasattr(attention, name)]
        if missing:
            raise TypeError(
                "KV prompts require timm ViT-style attention; "
                f"missing attributes: {missing}."
            )
        if not isinstance(attention.qkv, nn.Linear) or not isinstance(
            attention.proj, nn.Linear
        ):
            raise TypeError("KV prompts require Linear qkv and proj modules.")
        if not isinstance(attention.q_norm, nn.Identity) or not isinstance(
            attention.k_norm, nn.Identity
        ):
            raise ValueError("KV prompts currently require Identity q_norm and k_norm.")

        self.num_heads = int(attention.num_heads)
        self.head_dim = int(attention.head_dim)
        self.attn_dim = int(attention.attn_dim)
        if self.attn_dim != self.num_heads * self.head_dim:
            raise ValueError("attn_dim must equal num_heads * head_dim.")
        if attention.qkv.out_features != 3 * self.attn_dim:
            raise ValueError("qkv.out_features must equal 3 * attn_dim.")
        if attention.proj.in_features != self.attn_dim:
            raise ValueError("proj.in_features must equal attn_dim.")

        # Preserve the original module names so recovery checkpoint state_dict
        # keys remain blocks.<i>.attn.qkv/proj rather than gaining a wrapper
        # nesting level.
        self.qkv = attention.qkv
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.attn_drop = attention.attn_drop
        self.norm = attention.norm
        self.proj = attention.proj
        self.proj_drop = attention.proj_drop
        self.scale = float(attention.scale)
        self.fused_attn = bool(attention.fused_attn)

        self.num_prompt_tokens = num_prompt_tokens
        self.kv_prompt = nn.Parameter(
            torch.empty(self.num_heads, self.num_prompt_tokens, self.head_dim)
        )
        nn.init.kaiming_uniform_(
            self.kv_prompt,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

    @property
    def prompt_parameter_count(self):
        return self.kv_prompt.numel()

    def forward(self, x, attn_mask=None, is_causal=False):
        if attn_mask is not None or is_causal:
            raise ValueError("KVPromptedAttention supports non-masked ViT attention only.")

        batch_size, num_query_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            num_query_tokens,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        prompt = self.kv_prompt.unsqueeze(0).expand(batch_size, -1, -1, -1)
        k = torch.cat((prompt, k), dim=2)
        v = torch.cat((prompt, v), dim=2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attention = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attention = self.attn_drop(attention)
            x = attention @ v

        x = x.transpose(1, 2).reshape(
            batch_size, num_query_tokens, self.attn_dim
        )
        x = self.norm(x)
        x = self.proj(x)
        return self.proj_drop(x)


def inject_kv_prompts(blocks, token_counts):
    """Replace selected block attention modules and return their indices."""

    if len(blocks) != len(token_counts):
        raise ValueError("KV prompt token counts must match transformer blocks.")

    prompted_layers = []
    for layer_index, (block, count) in enumerate(zip(blocks, token_counts)):
        count = int(count)
        if count < 0:
            raise ValueError("KV prompt token counts must be non-negative.")
        if count == 0:
            continue
        block.attn = KVPromptedAttention(block.attn, count)
        prompted_layers.append(layer_index)
    return tuple(prompted_layers)
