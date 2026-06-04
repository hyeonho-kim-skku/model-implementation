"""Structured pruning pipeline for dense timm ViT-style classifiers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

from datasets import get_loader
from pruning.importance import (
    MLPActivationTaylorCollector,
    MLPActivationTaylorImportance,
    MLPGateTaylorCollector,
    VALID_ACTIVATION_TAYLOR_REDUCTIONS,
    VALID_GATE_TAYLOR_REDUCTIONS,
    VALID_GATE_TAYLOR_LOCATIONS,
)
from utils import move_to_device


VALID_PRUNING_MODULES = {"head", "mlp"}


def _normalize_calibration_batches(calibration_batches):
    """Accept an integer batch limit or None/full for the whole loader."""

    if calibration_batches is None:
        return None
    if isinstance(calibration_batches, str):
        value = calibration_batches.strip().lower()
        if value in {"", "none", "null", "full", "all"}:
            return None
        calibration_batches = int(value)
    if calibration_batches < 0:
        raise ValueError("calibration_batches must be non-negative or full.")
    return calibration_batches


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


def _normalize_pruning_modules(pruning_modules: str | None) -> tuple[str, ...]:
    """Normalize a comma-separated string into supported pruning targets.

    Examples:
        None -> ()
        "mlp" -> ("mlp",)
        "head,mlp" -> ("head", "mlp")
    """

    if pruning_modules is None:
        return ()
    normalized_modules = tuple(
        item.strip().lower() for item in pruning_modules.split(",") if item.strip()
    )

    invalid_modules = set(normalized_modules) - VALID_PRUNING_MODULES
    if invalid_modules:
        raise ValueError(f"Unsupported pruning modules: {sorted(invalid_modules)}")
    return normalized_modules


def _normalize_target_block_indices(target_block_indices, num_blocks: int) -> tuple[int, ...] | None:
    """Normalize optional transformer block indices for layer-wise pruning."""

    if target_block_indices is None:
        return None
    if isinstance(target_block_indices, str):
        if not target_block_indices.strip():
            return None
        indices = tuple(int(item.strip()) for item in target_block_indices.split(",") if item.strip())
    else:
        indices = tuple(int(item) for item in target_block_indices)

    invalid_indices = [idx for idx in indices if idx < 0 or idx >= num_blocks]
    if invalid_indices:
        raise ValueError(
            f"target_block_indices contains out-of-range indices {invalid_indices}; "
            f"valid range is 0..{num_blocks - 1}."
        )
    return tuple(dict.fromkeys(indices))


def _collect_pruning_targets(
    model,
    pruning_modules: tuple[str, ...],
    target_block_indices=None,
) -> PruningTargets:
    """Collect concrete nn.Module objects that should be allowed to trigger pruning.

    Examples:
        pruning_modules=("mlp",)
            -> mlp_layers={block.mlp.fc1 for each transformer block}
            -> attention_proj_layers=set()
            -> num_heads={}

        pruning_modules=("head", "mlp")
            -> mlp_layers={block.mlp.fc1 for each transformer block}
            -> attention_proj_layers={block.attn.proj for each transformer block}
            -> num_heads={block.attn.qkv: block.attn.num_heads for each transformer block}
    """

    if not hasattr(model.encoder, "blocks"):
        raise ValueError("This model does not expose transformer blocks for structured pruning.")

    mlp_layers = []
    attention_proj_layers = []
    num_heads = {}
    prune_attention_heads = "head" in pruning_modules
    selected_block_indices = _normalize_target_block_indices(
        target_block_indices,
        num_blocks=len(model.encoder.blocks),
    )

    for block_idx, block in enumerate(model.encoder.blocks):
        if selected_block_indices is not None and block_idx not in selected_block_indices:
            continue
        if prune_attention_heads:
            # Torch-Pruning's MHA path roots head pruning at proj.in_features,
            # then propagates matching q/k/v output pruning through the graph.
            attention_proj_layers.append(block.attn.proj)
            num_heads[block.attn.qkv] = block.attn.num_heads
        if "mlp" in pruning_modules:
            # fc1.out_features is the MLP hidden width. fc2.out_features is the
            # residual stream width and must stay fixed for post-training pruning.
            mlp_layers.append(block.mlp.fc1)

    return PruningTargets(
        mlp_layers=tuple(mlp_layers),
        attention_proj_layers=tuple(attention_proj_layers),
        num_heads=num_heads,
    )


def _count_ops_and_params(model, example_inputs):
    """Return MAC and parameter counts for the current model structure."""

    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return int(macs), int(params)


def _build_importance(importance, activation_taylor_scores=None, gate_taylor_scores=None):
    """Create the Torch-Pruning importance criterion.

    Magnitude uses only weights. Taylor uses weight * gradient, so gradients
    must be populated before pruner.step() asks this object to score groups.
    Activation Taylor uses precomputed fc2-input activation scores. Gate Taylor
    uses precomputed explicit gate-gradient scores.
    """

    importance = (importance or "magnitude").strip().lower()
    common_kwargs = {
        "group_reduction": "mean",
        "normalizer": "mean",
    }
    gate_taylor_kwargs = {
        "group_reduction": "mean",
        "normalizer": None,
    }

    if importance == "magnitude":
        return tp.importance.MagnitudeImportance(p=2, **common_kwargs), importance
    if importance == "taylor":
        return tp.importance.TaylorImportance(**common_kwargs), importance
    if importance == "activation_taylor":
        if activation_taylor_scores is None:
            raise ValueError("activation_taylor_scores is required for activation_taylor.")
        return MLPActivationTaylorImportance(activation_taylor_scores, **common_kwargs), importance
    if importance == "gate_taylor":
        if gate_taylor_scores is None:
            raise ValueError("gate_taylor_scores is required for gate_taylor.")
        return MLPActivationTaylorImportance(gate_taylor_scores, **gate_taylor_kwargs), importance

    raise ValueError(
        "importance must be 'magnitude', 'taylor', 'activation_taylor', or 'gate_taylor'."
    )


def _build_pruning_scope_config(
    targets: PruningTargets,
    pruning_modules: tuple[str, ...],
    pruning_ratio: float,
    global_pruning: bool,
) -> PruningScopeConfig:
    """Build explicit pruning scopes for Torch-Pruning.

    MLP hidden units and attention heads are scoped separately. When
    global_pruning=True, MLP layers compete with other selected MLP layers, and
    attention heads compete with other selected attention heads. MLP unit scores
    and attention-head scores are not mixed.
    """

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


def _build_pruner(
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
    """Create a Torch-Pruning pruner for the selected target scopes.

    The returned pruner is the object that traces the model with example_inputs,
    builds the dependency graph, estimates channel importance, and applies
    dependency-aware pruning.

    Example for pruning_modules=("mlp",):
        targets.mlp_layers = {
            model.encoder.blocks[0].mlp.fc1,
            model.encoder.blocks[1].mlp.fc1,
            ...
        }
        targets.attention_proj_layers = set()
        targets.num_heads = {}
    """

    importance, importance_type = _build_importance(
        importance,
        activation_taylor_scores=activation_taylor_scores,
        gate_taylor_scores=gate_taylor_scores,
    )
    ignored_layers = [model.classifier]
    root_module_types = [nn.Linear]
    targets = _collect_pruning_targets(model, pruning_modules, target_block_indices)
    scope_config = _build_pruning_scope_config(
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
        ignored_layers=ignored_layers,
        round_to=round_to,
        root_module_types=root_module_types,
        num_heads=targets.num_heads,
        prune_head_dims=False,
        prune_num_heads=("head" in pruning_modules),
        head_pruning_ratio=scope_config.head_pruning_ratio,
        head_pruning_ratio_dict=scope_config.head_pruning_ratio_dict,
    )
    return pruner, importance_type


def compute_taylor_gradients(
    model,
    calibration_dataset,
    calibration_batch_size,
    calibration_batches,
    calibration_split,
    num_workers,
    data_root,
    device,
    calibration_seed=None,
    activation_taylor_collector=None,
    gate_taylor_collector=None,
):
    """Run supervised batches so Taylor pruning criteria can read gradients."""

    if calibration_dataset is None:
        raise ValueError("Taylor pruning needs calibration_dataset.")

    calibration_batches = _normalize_calibration_batches(calibration_batches)
    generator = None
    if calibration_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(calibration_seed))

    dataloader = get_loader(
        calibration_dataset,
        calibration_batch_size,
        mode="test",
        train=(calibration_split == "train"),
        shuffle=(calibration_split == "train"),
        drop_last=False,
        num_workers=num_workers,
        data_root=data_root,
        generator=generator,
    )

    model.eval()
    model.zero_grad(set_to_none=True)

    processed_batches = 0
    total_examples = 0
    for batch_idx, batch in enumerate(dataloader):
        if calibration_batches is not None and batch_idx >= calibration_batches:
            break
        images, labels = move_to_device(batch, device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        # Do not optimizer.step(). Taylor pruning only needs d(loss)/d(weight).
        # Torch-Pruning reads parameter gradients later for weight Taylor. For
        # activation/gate Taylor, the collectors read retained activation
        # gradients or explicit gate gradients.
        loss.backward()
        if activation_taylor_collector is not None:
            activation_taylor_collector.accumulate_batch()
        if gate_taylor_collector is not None:
            gate_taylor_collector.accumulate_batch()

        processed_batches += 1
        total_examples += labels.size(0)

    if processed_batches == 0:
        raise ValueError("Taylor pruning did not process any calibration batches.")

    calibration_config = {
        "dataset": calibration_dataset,
        "batch_size": calibration_batch_size,
        "requested_batches": calibration_batches
        if calibration_batches is not None
        else "full",
        "split": calibration_split,
        "transform_mode": "test",
        "loss_reduction": "sum",
        "seed": calibration_seed,
        "processed_batches": processed_batches,
        "processed_examples": total_examples,
    }
    if activation_taylor_collector is not None:
        calibration_config["activation_taylor_reduction"] = (
            activation_taylor_collector.reduction
        )
    if gate_taylor_collector is not None:
        calibration_config["gate_taylor_reduction"] = gate_taylor_collector.reduction
        calibration_config["gate_taylor_location"] = gate_taylor_collector.gate_location
        calibration_config["gate_taylor_score_mode"] = gate_taylor_collector.score_mode
    return calibration_config


def _linear_shape(layer):
    return (layer.in_features, layer.out_features)


def _attention_metadata(block):
    attn = block.attn
    return {
        "num_heads": int(attn.num_heads),
        "head_dim": int(attn.head_dim),
        "attn_dim": int(getattr(attn, "attn_dim", attn.num_heads * attn.head_dim)),
        "qkv": _linear_shape(attn.qkv),
        "proj": _linear_shape(attn.proj),
    }


def _collect_attention_metadata(model, target_block_indices=None):
    metadata = {}
    selected_block_indices = _normalize_target_block_indices(
        target_block_indices,
        num_blocks=len(model.encoder.blocks),
    )
    for block_idx, block in enumerate(model.encoder.blocks):
        if selected_block_indices is not None and block_idx not in selected_block_indices:
            continue
        metadata[f"blocks.{block_idx}.attn"] = _attention_metadata(block)
    return metadata


def _collect_target_shapes(model, pruning_modules, target_block_indices=None):
    shapes = {}
    if not pruning_modules:
        return shapes

    selected_block_indices = _normalize_target_block_indices(
        target_block_indices,
        num_blocks=len(model.encoder.blocks),
    )
    for block_idx, block in enumerate(model.encoder.blocks):
        if selected_block_indices is not None and block_idx not in selected_block_indices:
            continue
        if "mlp" in pruning_modules:
            shapes[f"blocks.{block_idx}.mlp.fc1"] = _linear_shape(block.mlp.fc1)
            shapes[f"blocks.{block_idx}.mlp.fc2"] = _linear_shape(block.mlp.fc2)
        if "head" in pruning_modules:
            shapes[f"blocks.{block_idx}.attn.qkv"] = _linear_shape(block.attn.qkv)
            shapes[f"blocks.{block_idx}.attn.proj"] = _linear_shape(block.attn.proj)
            shapes[f"blocks.{block_idx}.attn.num_heads"] = block.attn.num_heads
            shapes[f"blocks.{block_idx}.attn.head_dim"] = block.attn.head_dim
            shapes[f"blocks.{block_idx}.attn.attn_dim"] = getattr(
                block.attn,
                "attn_dim",
                block.attn.num_heads * block.attn.head_dim,
            )
    return shapes


def _print_shape_changes(before_shapes, after_shapes, max_lines=24):
    def _format_shape(value):
        return f"Linear{value}" if isinstance(value, tuple) else str(value)

    changed = [
        (name, before_shapes[name], after_shapes.get(name))
        for name in before_shapes
        if after_shapes.get(name) != before_shapes[name]
    ]
    print(f"[Pruning][Inspect] changed target layers: {len(changed)}")
    for name, before, after in changed[:max_lines]:
        print(f"  {name}: {_format_shape(before)} -> {_format_shape(after)}")
    if len(changed) > max_lines:
        print(f"  ... {len(changed) - max_lines} more changed layers")


def _ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def _build_target_pruning_summary(before_shapes, after_shapes):
    """Summarize target layer sparsity from before/after structural metadata."""

    summary = {
        "overall": {},
        "by_layer": {},
    }
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
                "pruned_ratio": _ratio(pruned_hidden, hidden_before),
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
                "pruned_ratio": _ratio(pruned_heads, before),
            }
            heads_before += before
            heads_after += after

    if mlp_before:
        pruned_hidden = mlp_before - mlp_after
        summary["overall"]["mlp"] = {
            "hidden_before": mlp_before,
            "hidden_after": mlp_after,
            "pruned_hidden": pruned_hidden,
            "pruned_ratio": _ratio(pruned_hidden, mlp_before),
        }
    if heads_before:
        pruned_heads = heads_before - heads_after
        summary["overall"]["head"] = {
            "heads_before": heads_before,
            "heads_after": heads_after,
            "pruned_heads": pruned_heads,
            "pruned_ratio": _ratio(pruned_heads, heads_before),
        }
    return summary


def _print_pruning_summary(summary, max_lines=24):
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


def _refresh_attention_metadata(model, attention_metadata_before, pruning_modules):
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


def _build_pruning_artifact(
    model,
    model_config,
    source_info,
    importance,
    calibration_config,
    pruning_modules,
    target_block_indices,
    pruning_ratio,
    iterative_steps,
    global_pruning,
    round_to,
    importance_group_reduction,
    importance_normalizer,
    activation_taylor_reduction,
    gate_taylor_reduction,
    gate_taylor_location,
    base_macs,
    base_params,
    pruned_macs,
    pruned_params,
    num_pruned_groups,
    target_pruning_summary,
    attention_metadata_before=None,
    attention_metadata_after=None,
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
            "iterative_steps": iterative_steps,
            "global_pruning": global_pruning,
            "round_to": round_to,
            "importance_group_reduction": importance_group_reduction,
            "importance_normalizer": importance_normalizer,
            "activation_taylor_reduction": activation_taylor_reduction,
            "gate_taylor_reduction": gate_taylor_reduction,
            "gate_taylor_location": gate_taylor_location,
            "calibration": calibration_config,
        },
        "pruning_stats": {
            "base_macs": base_macs,
            "pruned_macs": pruned_macs,
            "base_params": base_params,
            "pruned_params": pruned_params,
            "num_pruned_groups": num_pruned_groups,
            "target_pruning_summary": target_pruning_summary,
            "attention_metadata_before": attention_metadata_before,
            "attention_metadata_after": attention_metadata_after,
        },
    }


def prune_model(
    model,
    model_config,
    source_info,
    output_dir,
    output_path=None,
    importance="magnitude",
    pruning_ratio=0.2,
    pruning_modules=None,
    target_block_indices=None,
    iterative_steps=1,
    global_pruning=False,
    round_to=None,
    calibration_dataset=None,
    calibration_batch_size=64,
    calibration_batches=1,
    calibration_split="train",
    calibration_seed=None,
    activation_taylor_reduction="sum_abs",
    gate_taylor_reduction="sum_abs",
    gate_taylor_location="fc1_out",
    num_workers=4,
    data_root="./data",
    inspect_groups=False,
    use_existing_taylor_gradients=False,
    existing_calibration_config=None,
    existing_activation_taylor_scores=None,
    existing_gate_taylor_scores=None,
    save_artifact=True,
    verbose=True,
    device="cpu",
):
    """Prune an already-built dense timm classifier.

    target_block_indices limits pruning roots to selected transformer blocks.
    use_existing_taylor_gradients is for sweep jobs that already populated
    parameter.grad or activation scores and want to avoid repeated calibration.
    save_artifact=False is useful for sensitivity sweeps that only need metrics.
    verbose=False suppresses per-trial logs during large sweeps.
    """

    model = model.to(device)
    model.eval()

    # Torch-Pruning needs a representative input shape to trace the model and
    # build a dependency graph. The values are random because pruning only needs
    # the forward graph and tensor shapes here, not real dataset samples.
    # The same input is also used to report MACs before and after pruning.
    example_inputs = torch.randn(
        1,
        3,
        model_config["img_size"],
        model_config["img_size"],
        device=device,
    )

    normalized_modules = _normalize_pruning_modules(pruning_modules)
    normalized_target_block_indices = _normalize_target_block_indices(
        target_block_indices,
        num_blocks=len(model.encoder.blocks),
    )

    before_shapes = _collect_target_shapes(
        model,
        normalized_modules,
        normalized_target_block_indices,
    )
    attention_metadata_before = (
        _collect_attention_metadata(model, normalized_target_block_indices)
        if "head" in normalized_modules
        else None
    )
    base_macs, base_params = _count_ops_and_params(model, example_inputs)
    calibration_config = None
    importance_type = (importance or "magnitude").strip().lower()
    importance_group_reduction = "mean"
    importance_normalizer = None if importance_type == "gate_taylor" else "mean"
    activation_taylor_scores = None
    gate_taylor_scores = None
    if importance_type == "activation_taylor":
        if activation_taylor_reduction not in VALID_ACTIVATION_TAYLOR_REDUCTIONS:
            raise ValueError(
                "activation_taylor_reduction must be one of "
                f"{sorted(VALID_ACTIVATION_TAYLOR_REDUCTIONS)}, "
                f"got {activation_taylor_reduction!r}."
            )
        if normalized_modules != ("mlp",):
            raise ValueError(
                "activation_taylor currently supports pruning_modules='mlp' only."
            )
        activation_taylor_scores = (
            {} if existing_activation_taylor_scores is None else existing_activation_taylor_scores
        )
    if importance_type == "gate_taylor":
        if gate_taylor_reduction not in VALID_GATE_TAYLOR_REDUCTIONS:
            raise ValueError(
                "gate_taylor reduction must be one of "
                f"{sorted(VALID_GATE_TAYLOR_REDUCTIONS)}, "
                f"got {gate_taylor_reduction!r}."
            )
        if gate_taylor_location not in VALID_GATE_TAYLOR_LOCATIONS:
            raise ValueError(
                "gate_taylor_location must be one of "
                f"{sorted(VALID_GATE_TAYLOR_LOCATIONS)}, got {gate_taylor_location!r}."
            )
        if normalized_modules != ("mlp",):
            raise ValueError("gate_taylor currently supports pruning_modules='mlp' only.")
        gate_taylor_scores = (
            {} if existing_gate_taylor_scores is None else existing_gate_taylor_scores
        )
    if normalized_modules:
        # The pruner builds the dependency graph and decides which channel groups
        # can be removed together. The importance object only decides the ranking.
        pruner, importance_type = _build_pruner(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_ratio=pruning_ratio,
            pruning_modules=normalized_modules,
            target_block_indices=normalized_target_block_indices,
            iterative_steps=iterative_steps,
            global_pruning=global_pruning,
            round_to=round_to,
            activation_taylor_scores=activation_taylor_scores,
            gate_taylor_scores=gate_taylor_scores,
        )
        if importance_type in {"taylor", "activation_taylor", "gate_taylor"}:
            # Taylor scores are based on weight * gradient, so run calibration
            # backward passes before pruner.step() asks for channel scores.
            if iterative_steps != 1:
                raise ValueError(
                    "Taylor pruning currently supports iterative_steps=1. "
                    "For iterative Taylor pruning, recompute gradients before each step."
                )
            if use_existing_taylor_gradients:
                # The caller is responsible for restoring parameter.grad before
                # prune_model is called, or for passing activation scores.
                if existing_calibration_config is None:
                    raise ValueError(
                        "existing_calibration_config is required when "
                        "use_existing_taylor_gradients=True."
                    )
                if importance_type == "activation_taylor" and not activation_taylor_scores:
                    raise ValueError(
                        "existing_activation_taylor_scores is required when "
                        "activation_taylor uses existing calibration."
                    )
                if importance_type == "gate_taylor" and not gate_taylor_scores:
                    raise ValueError(
                        "existing_gate_taylor_scores is required when "
                        "gate_taylor uses existing calibration."
                    )
                calibration_config = existing_calibration_config
            else:
                activation_taylor_collector = None
                gate_taylor_collector = None
                if importance_type == "activation_taylor":
                    activation_taylor_collector = MLPActivationTaylorCollector(
                        model=model,
                        target_block_indices=normalized_target_block_indices,
                        reduction=activation_taylor_reduction,
                    )
                if importance_type == "gate_taylor":
                    gate_taylor_collector = MLPGateTaylorCollector(
                        model=model,
                        target_block_indices=normalized_target_block_indices,
                        reduction=gate_taylor_reduction,
                        gate_location=gate_taylor_location,
                    )
                try:
                    calibration_config = compute_taylor_gradients(
                        model=model,
                        calibration_dataset=calibration_dataset,
                        calibration_batch_size=calibration_batch_size,
                        calibration_batches=calibration_batches,
                        calibration_split=calibration_split,
                        num_workers=num_workers,
                        data_root=data_root,
                        device=device,
                        calibration_seed=calibration_seed,
                        activation_taylor_collector=activation_taylor_collector,
                        gate_taylor_collector=gate_taylor_collector,
                    )
                    if activation_taylor_collector is not None:
                        activation_taylor_scores.update(
                            activation_taylor_collector.final_scores()
                        )
                    if gate_taylor_collector is not None:
                        gate_taylor_scores.update(gate_taylor_collector.final_scores())
                finally:
                    if activation_taylor_collector is not None:
                        activation_taylor_collector.remove()
                    if gate_taylor_collector is not None:
                        gate_taylor_collector.remove()
        history_before = len(pruner.pruning_history())
        pruner.step()
        num_pruned_groups = len(pruner.pruning_history()) - history_before
        _refresh_attention_metadata(
            model,
            attention_metadata_before=attention_metadata_before,
            pruning_modules=normalized_modules,
        )
    else:
        # No module was explicitly selected, so leave the model unchanged.
        num_pruned_groups = 0
    # Taylor pruning leaves calibration gradients on parameters. They are useful
    # only while pruner.step() is choosing channels, so clear them before saving.
    model.zero_grad(set_to_none=True)
    after_shapes = _collect_target_shapes(
        model,
        normalized_modules,
        normalized_target_block_indices,
    )
    target_pruning_summary = _build_target_pruning_summary(before_shapes, after_shapes)
    if inspect_groups:
        _print_shape_changes(before_shapes, after_shapes)
        _print_pruning_summary(target_pruning_summary)
    pruned_macs, pruned_params = _count_ops_and_params(model, example_inputs)
    attention_metadata_after = (
        _collect_attention_metadata(model, normalized_target_block_indices)
        if "head" in normalized_modules
        else None
    )

    artifact = _build_pruning_artifact(
        model=model.cpu(),
        model_config=model_config,
        source_info=source_info,
        importance=importance_type,
        calibration_config=calibration_config,
        pruning_modules=normalized_modules,
        target_block_indices=normalized_target_block_indices,
        pruning_ratio=pruning_ratio,
        iterative_steps=iterative_steps,
        global_pruning=global_pruning,
        round_to=round_to,
        importance_group_reduction=importance_group_reduction,
        importance_normalizer=importance_normalizer,
        activation_taylor_reduction=(
            activation_taylor_reduction
            if importance_type == "activation_taylor"
            else None
        ),
        gate_taylor_reduction=gate_taylor_reduction if importance_type == "gate_taylor" else None,
        gate_taylor_location=gate_taylor_location if importance_type == "gate_taylor" else None,
        base_macs=base_macs,
        base_params=base_params,
        pruned_macs=pruned_macs,
        pruned_params=pruned_params,
        num_pruned_groups=num_pruned_groups,
        target_pruning_summary=target_pruning_summary,
        attention_metadata_before=attention_metadata_before,
        attention_metadata_after=attention_metadata_after,
    )

    if output_path is None:
        output_path = os.path.join(output_dir, "pruned_timm_classifier.pth")
    if save_artifact:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(artifact, output_path)

    if verbose:
        print(f"[Pruning] source: {source_info}")
        print(f"[Pruning] importance: {importance_type}")
        print(f"[Pruning] importance group reduction: {importance_group_reduction}")
        print(f"[Pruning] importance normalizer: {importance_normalizer}")
        if calibration_config is not None:
            print(f"[Pruning] calibration: {calibration_config}")
        print(f"[Pruning] modules: {list(normalized_modules)}")
        print(f"[Pruning] target blocks: {normalized_target_block_indices}")
        print(f"[Pruning] ratio: {pruning_ratio}")
        print(f"[Pruning] groups pruned: {num_pruned_groups}")
        _print_pruning_summary(target_pruning_summary, max_lines=0)
        print(f"[Pruning] MACs: {base_macs:,} -> {pruned_macs:,}")
        print(f"[Pruning] Params: {base_params:,} -> {pruned_params:,}")
        if save_artifact:
            print(f"[Pruning] saved to: {output_path}")

    return artifact
