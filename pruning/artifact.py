"""Structural metadata, summaries, and serialized pruning artifacts."""

from __future__ import annotations

from pruning.structured_core import normalize_target_block_indices


def linear_shape(layer):
    return (layer.in_features, layer.out_features)


def attention_metadata(block):
    attn = block.attn
    return {
        "num_heads": int(attn.num_heads),
        "head_dim": int(attn.head_dim),
        "attn_dim": int(getattr(attn, "attn_dim", attn.num_heads * attn.head_dim)),
        "qkv": linear_shape(attn.qkv),
        "proj": linear_shape(attn.proj),
    }


def collect_attention_metadata(model, target_block_indices=None):
    metadata = {}
    selected_block_indices = normalize_target_block_indices(
        target_block_indices,
        num_blocks=len(model.encoder.blocks),
    )
    for block_idx, block in enumerate(model.encoder.blocks):
        if selected_block_indices is not None and block_idx not in selected_block_indices:
            continue
        metadata[f"blocks.{block_idx}.attn"] = attention_metadata(block)
    return metadata


def collect_target_shapes(model, pruning_modules, target_block_indices=None):
    shapes = {}
    if not pruning_modules:
        return shapes

    selected_block_indices = normalize_target_block_indices(
        target_block_indices,
        num_blocks=len(model.encoder.blocks),
    )
    for block_idx, block in enumerate(model.encoder.blocks):
        if selected_block_indices is not None and block_idx not in selected_block_indices:
            continue
        if "mlp" in pruning_modules:
            shapes[f"blocks.{block_idx}.mlp.fc1"] = linear_shape(block.mlp.fc1)
            shapes[f"blocks.{block_idx}.mlp.fc2"] = linear_shape(block.mlp.fc2)
        if "head" in pruning_modules:
            shapes[f"blocks.{block_idx}.attn.qkv"] = linear_shape(block.attn.qkv)
            shapes[f"blocks.{block_idx}.attn.proj"] = linear_shape(block.attn.proj)
            shapes[f"blocks.{block_idx}.attn.num_heads"] = block.attn.num_heads
            shapes[f"blocks.{block_idx}.attn.head_dim"] = block.attn.head_dim
            shapes[f"blocks.{block_idx}.attn.attn_dim"] = getattr(
                block.attn,
                "attn_dim",
                block.attn.num_heads * block.attn.head_dim,
            )
    return shapes


def print_shape_changes(before_shapes, after_shapes, max_lines=24):
    def format_shape(value):
        return f"Linear{value}" if isinstance(value, tuple) else str(value)

    changed = [
        (name, before_shapes[name], after_shapes.get(name))
        for name in before_shapes
        if after_shapes.get(name) != before_shapes[name]
    ]
    print(f"[Pruning][Inspect] changed target layers: {len(changed)}")
    for name, before, after in changed[:max_lines]:
        print(f"  {name}: {format_shape(before)} -> {format_shape(after)}")
    if len(changed) > max_lines:
        print(f"  ... {len(changed) - max_lines} more changed layers")


def ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def build_target_pruning_summary(before_shapes, after_shapes):
    """Summarize target layer sparsity from before/after structural metadata."""

    summary = {"overall": {}, "by_layer": {}}
    mlp_before = 0
    mlp_after = 0
    heads_before = 0
    heads_after = 0

    for name, before in before_shapes.items():
        after = after_shapes.get(name)
        if after is None:
            continue
        if name.endswith(".mlp.fc1"):
            layer_name = name.rsplit(".fc1", 1)[0]
            hidden_before = before[1]
            hidden_after = after[1]
            pruned_hidden = hidden_before - hidden_after
            summary["by_layer"][layer_name] = {
                "type": "mlp",
                "hidden_before": hidden_before,
                "hidden_after": hidden_after,
                "pruned_hidden": pruned_hidden,
                "pruned_ratio": ratio(pruned_hidden, hidden_before),
            }
            mlp_before += hidden_before
            mlp_after += hidden_after
        elif name.endswith(".attn.num_heads"):
            layer_name = name.rsplit(".num_heads", 1)[0]
            pruned_heads = before - after
            summary["by_layer"][layer_name] = {
                "type": "head",
                "heads_before": before,
                "heads_after": after,
                "pruned_heads": pruned_heads,
                "pruned_ratio": ratio(pruned_heads, before),
            }
            heads_before += before
            heads_after += after

    if mlp_before:
        pruned_hidden = mlp_before - mlp_after
        summary["overall"]["mlp"] = {
            "hidden_before": mlp_before,
            "hidden_after": mlp_after,
            "pruned_hidden": pruned_hidden,
            "pruned_ratio": ratio(pruned_hidden, mlp_before),
        }
    if heads_before:
        pruned_heads = heads_before - heads_after
        summary["overall"]["head"] = {
            "heads_before": heads_before,
            "heads_after": heads_after,
            "pruned_heads": pruned_heads,
            "pruned_ratio": ratio(pruned_heads, heads_before),
        }
    return summary


def print_pruning_summary(summary, max_lines=24):
    for target_type, values in summary.get("overall", {}).items():
        if target_type == "mlp":
            print(
                "[Pruning][Summary] mlp hidden: "
                f"{values['hidden_before']} -> {values['hidden_after']} "
                f"({values['pruned_hidden']} pruned, ratio={values['pruned_ratio']:.4f})"
            )
        elif target_type == "head":
            print(
                "[Pruning][Summary] attention heads: "
                f"{values['heads_before']} -> {values['heads_after']} "
                f"({values['pruned_heads']} pruned, ratio={values['pruned_ratio']:.4f})"
            )

    changed_layers = [
        (name, values)
        for name, values in summary.get("by_layer", {}).items()
        if values.get("pruned_hidden", values.get("pruned_heads", 0)) != 0
    ]
    if max_lines <= 0:
        return
    for name, values in changed_layers[:max_lines]:
        if values["type"] == "mlp":
            print(
                f"  {name}: hidden {values['hidden_before']} -> "
                f"{values['hidden_after']} ({values['pruned_hidden']} pruned)"
            )
        elif values["type"] == "head":
            print(
                f"  {name}: heads {values['heads_before']} -> "
                f"{values['heads_after']} ({values['pruned_heads']} pruned)"
            )
    if len(changed_layers) > max_lines:
        print(f"  ... {len(changed_layers) - max_lines} more changed target layers")


def refresh_attention_metadata(model, attention_metadata_before, pruning_modules):
    """Update timm attention metadata after attention heads have been pruned."""

    if "head" not in pruning_modules:
        return
    if attention_metadata_before is None:
        raise ValueError("attention_metadata_before is required for attention head pruning.")
    for block_idx, block in enumerate(model.encoder.blocks):
        attn = block.attn
        block_name = f"blocks.{block_idx}.attn"
        if block_name not in attention_metadata_before:
            continue
        original_metadata = attention_metadata_before[block_name]
        original_head_dim = original_metadata["head_dim"]
        if attn.qkv.out_features % (3 * original_head_dim) != 0:
            raise ValueError(
                "Pruned qkv width is incompatible with the original attention head dimension."
            )
        new_num_heads = attn.qkv.out_features // (3 * original_head_dim)
        if new_num_heads < 1:
            raise ValueError("Attention head pruning removed all heads from a block.")
        if attn.proj.in_features != new_num_heads * original_head_dim:
            raise ValueError(
                "Pruned attention projection width is incompatible with qkv head metadata."
            )
        attn.num_heads = new_num_heads
        attn.head_dim = original_head_dim
        attn.attn_dim = new_num_heads * original_head_dim
        attn.scale = attn.head_dim ** -0.5


def build_pruning_artifact(
    model,
    model_config,
    source_info,
    importance,
    calibration_config,
    pruning_modules,
    target_block_indices,
    pruning_ratio,
    mlp_pruning_ratio,
    head_pruning_ratio,
    iterative_steps,
    global_pruning,
    round_to,
    importance_group_reduction,
    importance_normalizer,
    activation_taylor_reduction,
    gate_taylor_reduction,
    gate_taylor_location,
    gate_taylor_aggregation,
    head_gate_taylor_reduction,
    head_gate_taylor_location,
    head_gate_taylor_aggregation,
    head_pruning_root,
    base_macs,
    base_params,
    pruned_macs,
    pruned_params,
    num_pruned_groups,
    target_pruning_summary,
    attention_metadata_before=None,
    attention_metadata_after=None,
    selected_attention_heads=None,
    direct_head_pruning_metadata=None,
    num_pruned_mlp_groups=None,
    num_pruned_heads=None,
):
    """Package the pruned model and pruning statistics into a serializable artifact."""

    return {
        "model": model,
        "source": source_info,
        "model_config": model_config,
        "pruning_config": {
            "importance": importance,
            "pruning_modules": list(pruning_modules),
            "target_block_indices": (
                None if target_block_indices is None else list(target_block_indices)
            ),
            "pruning_ratio": pruning_ratio,
            "mlp_pruning_ratio": mlp_pruning_ratio,
            "head_pruning_ratio": head_pruning_ratio,
            "joint_ranking_scopes": (
                {"mlp": "global", "head": "global"}
                if importance == "joint_gate_taylor"
                else None
            ),
            "joint_calibration_passes": (
                1 if importance == "joint_gate_taylor" else None
            ),
            "iterative_steps": iterative_steps,
            "global_pruning": global_pruning,
            "round_to": round_to,
            "importance_group_reduction": importance_group_reduction,
            "importance_normalizer": importance_normalizer,
            "activation_taylor_reduction": activation_taylor_reduction,
            "gate_taylor_reduction": gate_taylor_reduction,
            "gate_taylor_location": gate_taylor_location,
            "gate_taylor_aggregation": gate_taylor_aggregation,
            "head_gate_taylor_reduction": head_gate_taylor_reduction,
            "head_gate_taylor_location": head_gate_taylor_location,
            "head_gate_taylor_aggregation": head_gate_taylor_aggregation,
            "head_pruning_root": head_pruning_root,
            "calibration": calibration_config,
        },
        "pruning_stats": {
            "base_macs": base_macs,
            "pruned_macs": pruned_macs,
            "base_params": base_params,
            "pruned_params": pruned_params,
            "num_pruned_groups": num_pruned_groups,
            "num_pruned_mlp_groups": num_pruned_mlp_groups,
            "num_pruned_heads": num_pruned_heads,
            "target_pruning_summary": target_pruning_summary,
            "attention_metadata_before": attention_metadata_before,
            "attention_metadata_after": attention_metadata_after,
            "selected_attention_heads": selected_attention_heads,
            "direct_head_pruning_metadata": direct_head_pruning_metadata,
        },
    }
