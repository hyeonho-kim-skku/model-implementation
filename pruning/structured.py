from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch_pruning as tp

from pruning.checkpoint import build_dense_model_from_checkpoint


DEFAULT_PRUNING_MODULES = ("mlp",)
VALID_PRUNING_MODULES = {"qkv", "mlp"}


@dataclass(frozen=True)
class PruningTargets:
    mlp_layers: set[nn.Module]
    attention_proj_layers: set[nn.Module]
    num_heads: dict[nn.Module, int]


def _normalize_pruning_modules(pruning_modules: str | Iterable[str] | None) -> tuple[str, ...]:
    if pruning_modules is None:
        return DEFAULT_PRUNING_MODULES
    if isinstance(pruning_modules, str):
        normalized_modules = tuple(
            item.strip().lower() for item in pruning_modules.split(",") if item.strip()
        )
    else:
        normalized_modules = tuple(item.lower() for item in pruning_modules)

    invalid_modules = set(normalized_modules) - VALID_PRUNING_MODULES
    if invalid_modules:
        raise ValueError(f"Unsupported pruning modules: {sorted(invalid_modules)}")
    return normalized_modules


def _iter_vit_blocks(model):
    if not hasattr(model.encoder, "blocks"):
        raise ValueError("This model does not expose transformer blocks for structured pruning.")
    return model.encoder.blocks


def _collect_pruning_targets(model, pruning_modules: tuple[str, ...]) -> PruningTargets:
    mlp_layers = set()
    attention_proj_layers = set()
    num_heads = {}
    prune_attention = "qkv" in pruning_modules

    for block in _iter_vit_blocks(model):
        if prune_attention:
            # Torch-Pruning's MHA path roots width pruning at proj.in_features,
            # then propagates matching q/k/v output pruning through the graph.
            attention_proj_layers.add(block.attn.proj)
            num_heads[block.attn.qkv] = block.attn.num_heads
        if "mlp" in pruning_modules:
            # fc1.out_features is the MLP hidden width. fc2.out_features is the
            # residual stream width and must stay fixed for post-training pruning.
            mlp_layers.add(block.mlp.fc1)

    return PruningTargets(
        mlp_layers=mlp_layers,
        attention_proj_layers=attention_proj_layers,
        num_heads=num_heads,
    )


def _count_ops_and_params(model, example_inputs):
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return int(macs), int(params)


def _build_pruner(
    model,
    example_inputs,
    pruning_ratio,
    pruning_modules,
    iterative_steps,
    global_pruning,
    round_to,
):
    importance = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [model.classifier]
    root_module_types = [nn.Linear]
    targets = _collect_pruning_targets(model, pruning_modules)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=pruning_ratio,
        iterative_steps=iterative_steps,
        global_pruning=global_pruning,
        ignored_layers=ignored_layers,
        round_to=round_to,
        root_module_types=root_module_types,
        num_heads=targets.num_heads,
        prune_head_dims=True,
        prune_num_heads=False,
    )
    return pruner, targets


def _is_target_group(dep, targets: PruningTargets, dependency_graph) -> bool:
    layer = dep.layer
    handler = dep.handler
    if layer in targets.mlp_layers:
        return dependency_graph.is_out_channel_pruning_fn(handler)
    if layer in targets.attention_proj_layers:
        return dependency_graph.is_in_channel_pruning_fn(handler)
    return False


def _execute_targeted_pruning(pruner, targets: PruningTargets):
    pruned_groups = []
    for group in pruner.step(interactive=True):
        dep, idxs = group[0]
        if not _is_target_group(dep, targets, pruner.DG):
            continue
        group.prune()
        pruned_groups.append((dep.layer, dep.handler, idxs))
    return pruned_groups


def _refresh_attention_metadata(model):
    for block in _iter_vit_blocks(model):
        attn = block.attn
        if attn.qkv.out_features % (3 * attn.num_heads) != 0:
            raise ValueError(
                "Pruned qkv width is incompatible with the current number of attention heads."
            )
        attn.head_dim = attn.qkv.out_features // (3 * attn.num_heads)
        attn.attn_dim = attn.head_dim * attn.num_heads
        attn.scale = attn.head_dim ** -0.5


def _build_pruning_artifact(
    checkpoint,
    model,
    pruning_modules,
    pruning_ratio,
    iterative_steps,
    global_pruning,
    round_to,
    base_macs,
    base_params,
    pruned_macs,
    pruned_params,
    pruned_groups,
):
    return {
        "model": model,
        "source_checkpoint": checkpoint,
        "model_config": checkpoint["model_config"],
        "pruning_config": {
            "pruning_modules": list(pruning_modules),
            "pruning_ratio": pruning_ratio,
            "iterative_steps": iterative_steps,
            "global_pruning": global_pruning,
            "round_to": round_to,
        },
        "pruning_stats": {
            "base_macs": base_macs,
            "pruned_macs": pruned_macs,
            "base_params": base_params,
            "pruned_params": pruned_params,
            "num_pruned_groups": len(pruned_groups),
        },
    }


def prune_checkpoint(
    checkpoint_path,
    output_dir,
    output_path=None,
    pruning_ratio=0.2,
    pruning_modules="mlp",
    iterative_steps=1,
    global_pruning=False,
    round_to=None,
    device="cpu",
):
    checkpoint, model = build_dense_model_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()

    model_config = checkpoint["model_config"]
    example_inputs = torch.randn(
        1,
        3,
        model_config["img_size"],
        model_config["img_size"],
        device=device,
    )

    normalized_modules = _normalize_pruning_modules(pruning_modules)
    base_macs, base_params = _count_ops_and_params(model, example_inputs)
    pruner, targets = _build_pruner(
        model=model,
        example_inputs=example_inputs,
        pruning_ratio=pruning_ratio,
        pruning_modules=normalized_modules,
        iterative_steps=iterative_steps,
        global_pruning=global_pruning,
        round_to=round_to,
    )
    pruned_groups = _execute_targeted_pruning(pruner, targets)
    _refresh_attention_metadata(model)
    pruned_macs, pruned_params = _count_ops_and_params(model, example_inputs)

    artifact = _build_pruning_artifact(
        checkpoint=checkpoint,
        model=model.cpu(),
        pruning_modules=normalized_modules,
        pruning_ratio=pruning_ratio,
        iterative_steps=iterative_steps,
        global_pruning=global_pruning,
        round_to=round_to,
        base_macs=base_macs,
        base_params=base_params,
        pruned_macs=pruned_macs,
        pruned_params=pruned_params,
        pruned_groups=pruned_groups,
    )

    if output_path is None:
        output_path = os.path.join(output_dir, "pruned_timm_classifier.pth")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(artifact, output_path)

    print(f"[Pruning] checkpoint: {checkpoint_path}")
    print(f"[Pruning] modules: {list(normalized_modules)}")
    print(f"[Pruning] ratio: {pruning_ratio}")
    print(f"[Pruning] groups pruned: {len(pruned_groups)}")
    print(f"[Pruning] MACs: {base_macs:,} -> {pruned_macs:,}")
    print(f"[Pruning] Params: {base_params:,} -> {pruned_params:,}")
    print(f"[Pruning] saved to: {output_path}")

    return artifact
