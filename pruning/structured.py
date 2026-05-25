"""Structured pruning pipeline for dense timm ViT-style classifiers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

from datasets import get_loader
from utils import move_to_device


VALID_PRUNING_MODULES = {"head", "mlp"}


@dataclass(frozen=True)
class PruningTargets:
    """Module sets used to filter Torch-Pruning's proposed pruning groups."""

    mlp_layers: set[nn.Module]
    attention_proj_layers: set[nn.Module]
    num_heads: dict[nn.Module, int]


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

    mlp_layers = set()
    attention_proj_layers = set()
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
    """Return MAC and parameter counts for the current model structure."""

    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return int(macs), int(params)


def _build_importance(importance):
    """Create the Torch-Pruning importance criterion.

    Magnitude uses only weights. Taylor uses weight * gradient, so gradients
    must be populated before pruner.step() asks this object to score groups.
    """

    importance = (importance or "magnitude").strip().lower()
    common_kwargs = {
        "group_reduction": "mean",
        "normalizer": "mean",
    }

    if importance == "magnitude":
        return tp.importance.MagnitudeImportance(p=2, **common_kwargs), importance
    if importance == "taylor":
        return tp.importance.TaylorImportance(**common_kwargs), importance

    raise ValueError("importance must be 'magnitude' or 'taylor'.")


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
):
    """Create a Torch-Pruning meta pruner and the target filter metadata.

    The returned pruner is the object that traces the model with example_inputs,
    builds the dependency graph, estimates channel importance, and proposes
    pruning groups.

    Example for pruning_modules=("mlp",):
        targets.mlp_layers = {
            model.encoder.blocks[0].mlp.fc1,
            model.encoder.blocks[1].mlp.fc1,
            ...
        }
        targets.attention_proj_layers = set()
        targets.num_heads = {}
    """

    importance, importance_type = _build_importance(importance)
    ignored_layers = [model.classifier]
    root_module_types = [nn.Linear]
    # targets is not the pruned model. It is a filter that says which modules are
    # allowed to act as pruning roots when Torch-Pruning proposes dependency groups.
    targets = _collect_pruning_targets(model, pruning_modules, target_block_indices)

    pruner = tp.pruner.MetaPruner(
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
        prune_head_dims=False,
        prune_num_heads=("head" in pruning_modules),
        head_pruning_ratio=pruning_ratio if "head" in pruning_modules else 0.0,
    )
    return pruner, targets, importance_type


def compute_taylor_gradients(
    model,
    calibration_dataset,
    calibration_batch_size,
    calibration_batches,
    calibration_split,
    num_workers,
    data_root,
    device,
):
    """Run a few supervised batches so TaylorImportance can read .grad fields."""

    if calibration_dataset is None:
        raise ValueError("Taylor pruning needs calibration_dataset.")

    dataloader = get_loader(
        calibration_dataset,
        calibration_batch_size,
        mode="test",
        train=(calibration_split == "train"),
        shuffle=(calibration_split == "train"),
        drop_last=False,
        num_workers=num_workers,
        data_root=data_root,
    )

    model.eval()
    model.zero_grad(set_to_none=True)

    processed_batches = 0
    total_examples = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= calibration_batches:
            break
        images, labels = move_to_device(batch, device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        # Do not optimizer.step(). Taylor pruning only needs d(loss)/d(weight).
        # Torch-Pruning reads these gradients later inside pruner.step().
        loss.backward()

        processed_batches += 1
        total_examples += labels.size(0)

    if processed_batches == 0:
        raise ValueError("Taylor pruning did not process any calibration batches.")

    return {
        "dataset": calibration_dataset,
        "batch_size": calibration_batch_size,
        "requested_batches": calibration_batches,
        "split": calibration_split,
        "transform_mode": "test",
        "processed_batches": processed_batches,
        "processed_examples": total_examples,
    }


def _is_target_group(dep, targets: PruningTargets, dependency_graph) -> bool:
    """Check whether a Torch-Pruning dependency group starts from an allowed target.

    For MLP pruning, the allowed root is:
        layer == block.mlp.fc1
        handler == output-channel pruning

    This means the pruning group is allowed to start by removing fc1 hidden
    channels. Torch-Pruning will then handle dependent changes, such as removing
    the same hidden indices from fc2 input channels.
    """

    layer = dep.layer
    handler = dep.handler
    if layer in targets.mlp_layers:
        # MLP pruning should reduce fc1.out_features, not fc1.in_features.
        return dependency_graph.is_out_channel_pruning_fn(handler)
    if layer in targets.attention_proj_layers:
        # Attention head pruning is rooted at proj.in_features so that the
        # matching qkv output channels can be propagated by Torch-Pruning.
        return dependency_graph.is_in_channel_pruning_fn(handler)
    return False


def _execute_targeted_pruning(
    pruner,
    targets: PruningTargets,
    inspect_groups=False,
    max_inspect_groups=3,
):
    """Run interactive pruning and apply only groups that match configured targets.

    pruner.step(interactive=True) yields dependency groups one by one instead
    of pruning immediately. Each group contains all operations that must happen
    together to keep tensor shapes consistent.

    Example MLP group:
        root operation: prune block.mlp.fc1 output channels [idxs]
        dependent op:   prune block.mlp.fc2 input channels [same idxs]
    """

    pruned_groups = []
    inspected_groups = 0
    for group_idx, group in enumerate(pruner.step(interactive=True)):
        # group[0] is the root pruning operation proposed by Torch-Pruning.
        # dep describes the root layer and pruning direction, while idxs are the
        # channel indices selected by the importance criterion.
        dep, idxs = group[0]
        is_target = _is_target_group(dep, targets, pruner.DG)
        if inspect_groups and is_target and inspected_groups < max_inspect_groups:
            print(f"\n[Pruning][Inspect group {inspected_groups + 1}/{max_inspect_groups}]")
            print(f"proposal_index: {group_idx}")
            print(f"root_layer: {dep.layer}")
            print(f"root_handler: {dep.handler}")
            print(f"num_pruned_indices: {len(idxs)}")
            print(f"first_pruned_indices: {list(idxs)[:20]}")
            print(group)
            inspected_groups += 1

        if not is_target:
            # The group may be valid for the model, but it was not explicitly
            # selected by pruning_modules, so leave it untouched.
            continue
        if len(idxs) == 0:
            # A zero pruning ratio can still exercise the dependency graph and
            # target filtering path. Empty proposals are valid no-ops.
            continue
        # This mutates the model in place. Torch-Pruning applies every operation
        # in the dependency group, not only group[0].
        group.prune()
        pruned_groups.append((dep.layer, dep.handler, idxs))
    return pruned_groups


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
    base_macs,
    base_params,
    pruned_macs,
    pruned_params,
    pruned_groups,
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
            "calibration": calibration_config,
        },
        "pruning_stats": {
            "base_macs": base_macs,
            "pruned_macs": pruned_macs,
            "base_params": base_params,
            "pruned_params": pruned_params,
            "num_pruned_groups": len(pruned_groups),
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
    num_workers=4,
    data_root="./data",
    inspect_groups=False,
    max_inspect_groups=3,
    use_existing_taylor_gradients=False,
    existing_calibration_config=None,
    save_artifact=True,
    verbose=True,
    device="cpu",
):
    """Prune an already-built dense timm classifier.

    target_block_indices limits pruning roots to selected transformer blocks.
    use_existing_taylor_gradients is for sweep jobs that already populated
    parameter.grad and want to avoid repeated Taylor calibration passes.
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

    before_shapes = (
        _collect_target_shapes(model, normalized_modules, normalized_target_block_indices)
        if inspect_groups
        else None
    )
    attention_metadata_before = (
        _collect_attention_metadata(model, normalized_target_block_indices)
        if "head" in normalized_modules
        else None
    )
    base_macs, base_params = _count_ops_and_params(model, example_inputs)
    calibration_config = None
    importance_type = (importance or "magnitude").strip().lower()
    if normalized_modules:
        # The pruner builds the dependency graph and decides which channel groups
        # can be removed together. The importance object only decides the ranking.
        pruner, targets, importance_type = _build_pruner(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_ratio=pruning_ratio,
            pruning_modules=normalized_modules,
            target_block_indices=normalized_target_block_indices,
            iterative_steps=iterative_steps,
            global_pruning=global_pruning,
            round_to=round_to,
        )
        if importance_type == "taylor":
            # Taylor scores are based on weight * gradient, so run calibration
            # backward passes before pruner.step() asks for channel scores.
            if iterative_steps != 1:
                raise ValueError(
                    "Taylor pruning currently supports iterative_steps=1. "
                    "For iterative Taylor pruning, recompute gradients before each step."
                )
            if use_existing_taylor_gradients:
                # The caller is responsible for restoring parameter.grad before
                # prune_model is called. TaylorImportance reads those gradients
                # during pruner.step().
                if existing_calibration_config is None:
                    raise ValueError(
                        "existing_calibration_config is required when "
                        "use_existing_taylor_gradients=True."
                    )
                calibration_config = existing_calibration_config
            else:
                calibration_config = compute_taylor_gradients(
                    model=model,
                    calibration_dataset=calibration_dataset,
                    calibration_batch_size=calibration_batch_size,
                    calibration_batches=calibration_batches,
                    calibration_split=calibration_split,
                    num_workers=num_workers,
                    data_root=data_root,
                    device=device,
                )
        pruned_groups = _execute_targeted_pruning(
            pruner,
            targets,
            inspect_groups=inspect_groups,
            max_inspect_groups=max_inspect_groups,
        )
        _refresh_attention_metadata(
            model,
            attention_metadata_before=attention_metadata_before,
            pruning_modules=normalized_modules,
        )
    else:
        # No module was explicitly selected, so leave the model unchanged.
        pruned_groups = []
    # Taylor pruning leaves calibration gradients on parameters. They are useful
    # only while pruner.step() is choosing channels, so clear them before saving.
    model.zero_grad(set_to_none=True)
    if inspect_groups and before_shapes is not None:
        after_shapes = _collect_target_shapes(
            model,
            normalized_modules,
            normalized_target_block_indices,
        )
        _print_shape_changes(before_shapes, after_shapes)
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
        base_macs=base_macs,
        base_params=base_params,
        pruned_macs=pruned_macs,
        pruned_params=pruned_params,
        pruned_groups=pruned_groups,
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
        if calibration_config is not None:
            print(f"[Pruning] calibration: {calibration_config}")
        print(f"[Pruning] modules: {list(normalized_modules)}")
        print(f"[Pruning] target blocks: {normalized_target_block_indices}")
        print(f"[Pruning] ratio: {pruning_ratio}")
        print(f"[Pruning] groups pruned: {len(pruned_groups)}")
        print(f"[Pruning] MACs: {base_macs:,} -> {pruned_macs:,}")
        print(f"[Pruning] Params: {base_params:,} -> {pruned_params:,}")
        if save_artifact:
            print(f"[Pruning] saved to: {output_path}")

    return artifact
