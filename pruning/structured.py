"""Structured pruning pipeline for dense timm ViT-style classifiers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch_pruning as tp


VALID_PRUNING_MODULES = {"qkv", "mlp"}


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
        "qkv,mlp" -> ("qkv", "mlp")
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


def _collect_pruning_targets(model, pruning_modules: tuple[str, ...]) -> PruningTargets:
    """Collect concrete nn.Module objects that should be allowed to trigger pruning.

    Examples:
        pruning_modules=("mlp",)
            -> mlp_layers={block.mlp.fc1 for each transformer block}
            -> attention_proj_layers=set()
            -> num_heads={}

        pruning_modules=("qkv", "mlp")
            -> mlp_layers={block.mlp.fc1 for each transformer block}
            -> attention_proj_layers={block.attn.proj for each transformer block}
            -> num_heads={block.attn.qkv: block.attn.num_heads for each transformer block}
    """

    if not hasattr(model.encoder, "blocks"):
        raise ValueError("This model does not expose transformer blocks for structured pruning.")

    mlp_layers = set()
    attention_proj_layers = set()
    num_heads = {}
    prune_attention = "qkv" in pruning_modules

    for block in model.encoder.blocks:
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
    """Return MAC and parameter counts for the current model structure."""

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
    """Create a Torch-Pruning magnitude pruner and the target filter metadata.

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

    importance = tp.importance.MagnitudeImportance(
        p=2,
        # Combine importance scores from all parameterized ops in a dependency
        # group by averaging them. For MLP pruning, this means the fc1 output-row
        # score and the dependent fc2 input-column score are averaged per hidden
        # channel.
        group_reduction="mean",
        # Normalize each group's channel scores by that group's mean score. This
        # does not change the ranking within a group, but it makes scores more
        # comparable across groups when global_pruning=True.
        normalizer="mean",
    )
    ignored_layers = [model.classifier]
    root_module_types = [nn.Linear]
    # targets is not the pruned model. It is a filter that says which modules are
    # allowed to act as pruning roots when Torch-Pruning proposes dependency groups.
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
        # Attention width pruning is rooted at proj.in_features so that the
        # matching qkv/head dimensions can be propagated by Torch-Pruning.
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
        # This mutates the model in place. Torch-Pruning applies every operation
        # in the dependency group, not only group[0].
        group.prune()
        pruned_groups.append((dep.layer, dep.handler, idxs))
    return pruned_groups


def _linear_shape(layer):
    return (layer.in_features, layer.out_features)


def _collect_target_shapes(model, pruning_modules):
    shapes = {}
    if not pruning_modules:
        return shapes

    for block_idx, block in enumerate(model.encoder.blocks):
        if "mlp" in pruning_modules:
            shapes[f"blocks.{block_idx}.mlp.fc1"] = _linear_shape(block.mlp.fc1)
            shapes[f"blocks.{block_idx}.mlp.fc2"] = _linear_shape(block.mlp.fc2)
        if "qkv" in pruning_modules:
            shapes[f"blocks.{block_idx}.attn.qkv"] = _linear_shape(block.attn.qkv)
            shapes[f"blocks.{block_idx}.attn.proj"] = _linear_shape(block.attn.proj)
    return shapes


def _print_shape_changes(before_shapes, after_shapes, max_lines=24):
    changed = [
        (name, before_shapes[name], after_shapes.get(name))
        for name in before_shapes
        if after_shapes.get(name) != before_shapes[name]
    ]
    print(f"[Pruning][Inspect] changed target layers: {len(changed)}")
    for name, before, after in changed[:max_lines]:
        print(f"  {name}: Linear{before} -> Linear{after}")
    if len(changed) > max_lines:
        print(f"  ... {len(changed) - max_lines} more changed layers")


def _refresh_attention_metadata(model):
    """Update timm attention metadata after qkv/head dimensions have changed."""

    for block in model.encoder.blocks:
        attn = block.attn
        if attn.qkv.out_features % (3 * attn.num_heads) != 0:
            raise ValueError(
                "Pruned qkv width is incompatible with the current number of attention heads."
            )
        attn.head_dim = attn.qkv.out_features // (3 * attn.num_heads)
        attn.attn_dim = attn.head_dim * attn.num_heads
        attn.scale = attn.head_dim ** -0.5


def _build_pruning_artifact(
    model,
    model_config,
    source_info,
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
    """Package the pruned model and pruning statistics into a serializable artifact."""

    return {
        "model": model,
        "source": source_info,
        "model_config": model_config,
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


def prune_model(
    model,
    model_config,
    source_info,
    output_dir,
    output_path=None,
    pruning_ratio=0.2,
    pruning_modules=None,
    iterative_steps=1,
    global_pruning=False,
    round_to=None,
    inspect_groups=False,
    max_inspect_groups=3,
    device="cpu",
):
    """Prune an already-built dense timm classifier."""

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
    before_shapes = _collect_target_shapes(model, normalized_modules) if inspect_groups else None
    base_macs, base_params = _count_ops_and_params(model, example_inputs)
    if normalized_modules:
        pruner, targets = _build_pruner(
            model=model,
            example_inputs=example_inputs,
            pruning_ratio=pruning_ratio,
            pruning_modules=normalized_modules,
            iterative_steps=iterative_steps,
            global_pruning=global_pruning,
            round_to=round_to,
        )
        pruned_groups = _execute_targeted_pruning(
            pruner,
            targets,
            inspect_groups=inspect_groups,
            max_inspect_groups=max_inspect_groups,
        )
        _refresh_attention_metadata(model)
    else:
        # No module was explicitly selected, so leave the model unchanged.
        pruned_groups = []
    if inspect_groups and before_shapes is not None:
        after_shapes = _collect_target_shapes(model, normalized_modules)
        _print_shape_changes(before_shapes, after_shapes)
    pruned_macs, pruned_params = _count_ops_and_params(model, example_inputs)

    artifact = _build_pruning_artifact(
        model=model.cpu(),
        model_config=model_config,
        source_info=source_info,
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

    print(f"[Pruning] source: {source_info}")
    print(f"[Pruning] modules: {list(normalized_modules)}")
    print(f"[Pruning] ratio: {pruning_ratio}")
    print(f"[Pruning] groups pruned: {len(pruned_groups)}")
    print(f"[Pruning] MACs: {base_macs:,} -> {pruned_macs:,}")
    print(f"[Pruning] Params: {base_params:,} -> {pruned_params:,}")
    print(f"[Pruning] saved to: {output_path}")

    return artifact
