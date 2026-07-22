"""ViT-specific adapter for upstream Isomorphic Pruning.

Torch-Pruning mutates linear dimensions but timm Attention also stores static
head metadata.  These helpers make the mutation executable and record its
structural effect without changing the project's joint-pruning helpers.
"""

from __future__ import annotations

import torch.nn as nn


def collect_vit_attention_qkv(model):
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "blocks"):
        raise ValueError("Isomorphic pruning currently requires a timm ViT encoder.")
    result = {}
    for block_index, block in enumerate(model.encoder.blocks):
        attn = block.attn
        if not isinstance(attn.qkv, nn.Linear) or not isinstance(attn.proj, nn.Linear):
            raise TypeError("Isomorphic ViT adapter requires fused qkv and proj Linear layers.")
        result[block_index] = attn.qkv
    return result


def collect_vit_structure(model):
    """Return dimensions affected by the full Isomorphic method."""

    blocks = {}
    for index, block in enumerate(model.encoder.blocks):
        attn = block.attn
        blocks[f"blocks.{index}"] = {
            "embed_dim": int(block.norm1.normalized_shape[0]),
            "mlp_hidden_dim": int(block.mlp.fc1.out_features),
            "num_heads": int(attn.num_heads),
            "head_dim": int(attn.head_dim),
            "attn_dim": int(getattr(attn, "attn_dim", attn.num_heads * attn.head_dim)),
        }
    return {
        "classifier_in_features": int(model.classifier.in_features),
        "blocks": blocks,
    }


def refresh_vit_attention_metadata(model, num_heads_by_qkv):
    """Synchronize timm Attention fields after TP prunes heads/head dimensions."""

    for block in model.encoder.blocks:
        attn = block.attn
        qkv = attn.qkv
        num_heads = int(num_heads_by_qkv[qkv])
        if num_heads < 1:
            raise ValueError("Isomorphic pruning removed all heads from a block.")
        if qkv.out_features % (3 * num_heads) != 0:
            raise ValueError("Pruned qkv width is incompatible with its head count.")
        head_dim = qkv.out_features // (3 * num_heads)
        if attn.proj.in_features != num_heads * head_dim:
            raise ValueError("Pruned attention proj width is incompatible with qkv.")
        attn.num_heads = num_heads
        attn.head_dim = head_dim
        attn.attn_dim = num_heads * head_dim
        attn.scale = head_dim ** -0.5


def build_structure_summary(before, after):
    """Summarize full-width, MLP, head and head-dimension changes."""

    summary = {"classifier": {}, "blocks": {}}
    summary["classifier"] = {
        "in_features_before": before["classifier_in_features"],
        "in_features_after": after["classifier_in_features"],
    }
    for name, old in before["blocks"].items():
        new = after["blocks"][name]
        summary["blocks"][name] = {
            **{
                key: {"before": old[key], "after": new[key], "pruned": old[key] - new[key]}
                for key in ("embed_dim", "mlp_hidden_dim", "num_heads", "head_dim", "attn_dim")
            }
        }
    return summary
