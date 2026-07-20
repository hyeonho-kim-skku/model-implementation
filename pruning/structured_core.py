"""Torch-Pruning target, importance, and pruner construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torch_pruning as tp

from pruning.importance import MLPActivationTaylorImportance


VALID_PRUNING_MODULES = {"head", "mlp"}


@dataclass(frozen=True)
class PruningTargets:
    """Concrete modules selected as pruning roots."""

    mlp_layers: tuple[nn.Module, ...]
    attention_proj_layers: tuple[nn.Module, ...]
    num_heads: dict[nn.Module, int]


@dataclass(frozen=True)
class PruningScopeConfig:
    """Torch-Pruning ratio scopes for the selected pruning targets."""

    pruning_ratio: float
    pruning_ratio_dict: dict[nn.Module | tuple[nn.Module, ...], float] | None
    head_pruning_ratio: float
    head_pruning_ratio_dict: dict[nn.Module, float] | None


def normalize_pruning_modules(pruning_modules: str | None) -> tuple[str, ...]:
    if pruning_modules is None:
        return ()
    normalized_modules = tuple(
        item.strip().lower() for item in pruning_modules.split(",") if item.strip()
    )
    invalid_modules = set(normalized_modules) - VALID_PRUNING_MODULES
    if invalid_modules:
        raise ValueError(f"Unsupported pruning modules: {sorted(invalid_modules)}")
    return normalized_modules


def normalize_target_block_indices(
    target_block_indices,
    num_blocks: int,
) -> tuple[int, ...] | None:
    if target_block_indices is None:
        return None
    if isinstance(target_block_indices, str):
        if not target_block_indices.strip():
            return None
        indices = tuple(
            int(item.strip())
            for item in target_block_indices.split(",")
            if item.strip()
        )
    else:
        indices = tuple(int(item) for item in target_block_indices)

    invalid_indices = [idx for idx in indices if idx < 0 or idx >= num_blocks]
    if invalid_indices:
        raise ValueError(
            f"target_block_indices contains out-of-range indices {invalid_indices}; "
            f"valid range is 0..{num_blocks - 1}."
        )
    return tuple(dict.fromkeys(indices))


def collect_pruning_targets(
    model,
    pruning_modules: tuple[str, ...],
    target_block_indices=None,
) -> PruningTargets:
    if not hasattr(model.encoder, "blocks"):
        raise ValueError("This model does not expose transformer blocks for structured pruning.")

    mlp_layers = []
    attention_proj_layers = []
    num_heads = {}
    selected_block_indices = normalize_target_block_indices(
        target_block_indices,
        num_blocks=len(model.encoder.blocks),
    )

    for block_idx, block in enumerate(model.encoder.blocks):
        if selected_block_indices is not None and block_idx not in selected_block_indices:
            continue
        if "head" in pruning_modules:
            attention_proj_layers.append(block.attn.proj)
            num_heads[block.attn.qkv] = block.attn.num_heads
        if "mlp" in pruning_modules:
            mlp_layers.append(block.mlp.fc1)

    return PruningTargets(
        mlp_layers=tuple(mlp_layers),
        attention_proj_layers=tuple(attention_proj_layers),
        num_heads=num_heads,
    )


def count_ops_and_params(model, example_inputs):
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return int(macs), int(params)


def build_importance(
    importance,
    activation_taylor_scores=None,
    gate_taylor_scores=None,
):
    importance = (importance or "magnitude").strip().lower()
    common_kwargs = {"group_reduction": "mean", "normalizer": "mean"}
    gate_taylor_kwargs = {"group_reduction": "mean", "normalizer": None}

    if importance == "magnitude":
        return tp.importance.MagnitudeImportance(p=2, **common_kwargs), importance
    if importance == "taylor":
        return tp.importance.TaylorImportance(**common_kwargs), importance
    if importance == "activation_taylor":
        if activation_taylor_scores is None:
            raise ValueError("activation_taylor_scores is required for activation_taylor.")
        return MLPActivationTaylorImportance(
            activation_taylor_scores,
            **common_kwargs,
        ), importance
    if importance == "gate_taylor":
        if gate_taylor_scores is None:
            raise ValueError("gate_taylor_scores is required for gate_taylor.")
        return MLPActivationTaylorImportance(
            gate_taylor_scores,
            **gate_taylor_kwargs,
        ), importance
    raise ValueError(
        "importance must be 'magnitude', 'taylor', 'activation_taylor', or 'gate_taylor'."
    )


def build_pruning_scope_config(
    targets: PruningTargets,
    pruning_modules: tuple[str, ...],
    pruning_ratio: float,
    global_pruning: bool,
) -> PruningScopeConfig:
    pruning_ratio_dict = {}
    head_pruning_ratio_dict = {}
    head_pruning_ratio = 0.0

    if "mlp" in pruning_modules:
        if global_pruning:
            pruning_ratio_dict[targets.mlp_layers] = pruning_ratio
        else:
            for layer in targets.mlp_layers:
                pruning_ratio_dict[layer] = pruning_ratio

    if "head" in pruning_modules:
        qkv_layers = tuple(targets.num_heads.keys())
        if qkv_layers:
            if global_pruning:
                pruning_ratio_dict[qkv_layers] = pruning_ratio
            else:
                for qkv_layer in qkv_layers:
                    pruning_ratio_dict[qkv_layer] = pruning_ratio
        if global_pruning:
            head_pruning_ratio = pruning_ratio
        else:
            for qkv_layer in targets.num_heads:
                head_pruning_ratio_dict[qkv_layer] = pruning_ratio

    return PruningScopeConfig(
        pruning_ratio=0.0,
        pruning_ratio_dict=pruning_ratio_dict or None,
        head_pruning_ratio=head_pruning_ratio,
        head_pruning_ratio_dict=head_pruning_ratio_dict or None,
    )


def build_pruner(
    model,
    example_inputs,
    importance,
    pruning_ratio,
    pruning_modules,
    target_block_indices,
    iterative_steps,
    global_pruning,
    round_to,
    activation_taylor_scores=None,
    gate_taylor_scores=None,
):
    importance, importance_type = build_importance(
        importance,
        activation_taylor_scores=activation_taylor_scores,
        gate_taylor_scores=gate_taylor_scores,
    )
    targets = collect_pruning_targets(
        model,
        pruning_modules,
        target_block_indices,
    )
    scope_config = build_pruning_scope_config(
        targets=targets,
        pruning_modules=pruning_modules,
        pruning_ratio=pruning_ratio,
        global_pruning=global_pruning,
    )

    pruner = tp.pruner.BasePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=scope_config.pruning_ratio,
        pruning_ratio_dict=scope_config.pruning_ratio_dict,
        iterative_steps=iterative_steps,
        global_pruning=global_pruning,
        ignored_layers=[model.classifier],
        round_to=round_to,
        root_module_types=[nn.Linear],
        num_heads=targets.num_heads,
        prune_head_dims=False,
        prune_num_heads=("head" in pruning_modules),
        head_pruning_ratio=scope_config.head_pruning_ratio,
        head_pruning_ratio_dict=scope_config.head_pruning_ratio_dict,
    )
    return pruner, importance_type


def validate_pruning_ratio(name, value):
    value = float(value)
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}.")
    return value
