from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch_pruning as tp

from pruning.checkpoint import build_dense_model_from_checkpoint


DEFAULT_PRUNING_MODULES = ("proj", "mlp")
VALID_PRUNING_MODULES = {"qkv", "proj", "mlp"}


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


def _collect_prunable_layers(model, pruning_modules: tuple[str, ...]) -> list[nn.Module]:
    prunable_layers = []

    if not hasattr(model.encoder, "blocks"):
        raise ValueError("This model does not expose transformer blocks for structured pruning.")

    for block in model.encoder.blocks:
        if "qkv" in pruning_modules:
            prunable_layers.append(block.attn.qkv)
        if "proj" in pruning_modules:
            prunable_layers.append(block.attn.proj)
        if "mlp" in pruning_modules:
            prunable_layers.append(block.mlp.fc1)
            prunable_layers.append(block.mlp.fc2)

    return prunable_layers


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
    target_layers = set(_collect_prunable_layers(model, pruning_modules))

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
    )
    return pruner, target_layers


def _execute_targeted_pruning(pruner, target_layers):
    pruned_groups = []
    for group in pruner.step(interactive=True):
        dep, idxs = group[0]
        root_layer = dep.layer
        if root_layer not in target_layers:
            continue
        group.prune()
        pruned_groups.append((root_layer, idxs))
    return pruned_groups


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
    pruning_ratio=0.2,
    pruning_modules="proj,mlp",
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
    pruner, target_layers = _build_pruner(
        model=model,
        example_inputs=example_inputs,
        pruning_ratio=pruning_ratio,
        pruning_modules=normalized_modules,
        iterative_steps=iterative_steps,
        global_pruning=global_pruning,
        round_to=round_to,
    )
    pruned_groups = _execute_targeted_pruning(pruner, target_layers)
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

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pruned_timm_classifier.pth")
    torch.save(artifact, output_path)

    print(f"[Pruning] checkpoint: {checkpoint_path}")
    print(f"[Pruning] modules: {list(normalized_modules)}")
    print(f"[Pruning] ratio: {pruning_ratio}")
    print(f"[Pruning] groups pruned: {len(pruned_groups)}")
    print(f"[Pruning] MACs: {base_macs:,} -> {pruned_macs:,}")
    print(f"[Pruning] Params: {base_params:,} -> {pruned_params:,}")
    print(f"[Pruning] saved to: {output_path}")

    return artifact
