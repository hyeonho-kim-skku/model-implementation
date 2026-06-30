"""Direct attention-head pruning helpers.

This module should keep head selection explicit and use Torch-Pruning only as
the dependency-aware structural deletion engine. The intended flow is:

1. Receive block-indexed head scores, shaped [num_heads] per selected block.
2. Select concrete head ids to prune, either globally across blocks or locally
   within each block, while preserving at least one head per block.
3. Convert selected head ids to attention projection input-channel indices.
4. Ask torch_pruning.DependencyGraph for the pruning group and call group.prune().
5. Return selected-head metadata for artifact logging and downstream analysis.

This deliberately avoids adapting head scores to Torch-Pruning's importance API.
The research code chooses concrete head ids itself; Torch-Pruning is only asked
to remove the corresponding qkv/proj tensor slices consistently.
"""

from __future__ import annotations

import math

import torch
import torch_pruning as tp


def select_attention_heads_by_score(
    scores,
    pruning_ratio=None,
    *,
    global_pruning=False,
    pruned_head_counts=None,
    min_heads_per_block=1,
):
    """Select low-scoring attention heads for explicit structural pruning.

    This is the ranking step for head_gate_taylor. The caller passes already
    calibrated head scores, so this function does not inspect model weights or
    gradients. It simply turns scores into a concrete list such as
    {0: [1, 5, 8]}, meaning "prune heads 1, 5, and 8 in block 0".

    Args:
        scores: dict mapping block index to a 1D tensor/list of per-head scores.
        pruning_ratio: Fraction of heads to prune. Uses ceil(ratio * heads) to
            match the existing head-sensitivity ratio convention.
        global_pruning: If True, rank heads across all scored blocks. If False,
            apply the budget independently within each block.
        pruned_head_counts: Optional exact budget. For local pruning this can be
            an int applied to every block or a dict of block index to count. For
            global pruning this can be an int total or a dict whose values are
            summed into a total budget.
        min_heads_per_block: Preserve at least this many heads in every block.

    Returns:
        dict[int, list[int]] mapping block index to sorted selected head ids.
    """

    normalized_scores = _normalize_score_dict(scores)
    if not normalized_scores:
        return {}
    if pruning_ratio is None and pruned_head_counts is None:
        raise ValueError("Either pruning_ratio or pruned_head_counts must be provided.")
    if min_heads_per_block < 1:
        raise ValueError("min_heads_per_block must be at least 1.")

    if global_pruning:
        return _select_global_attention_heads(
            normalized_scores,
            pruning_ratio=pruning_ratio,
            pruned_head_counts=pruned_head_counts,
            min_heads_per_block=int(min_heads_per_block),
        )
    return _select_local_attention_heads(
        normalized_scores,
        pruning_ratio=pruning_ratio,
        pruned_head_counts=pruned_head_counts,
        min_heads_per_block=int(min_heads_per_block),
    )


def head_ids_to_proj_in_idxs(head_ids, head_dim):
    """Return attn.proj input-channel indices for whole attention heads.

    In timm ViT attention, heads are concatenated before attn.proj, so head h
    occupies proj input channels [h * head_dim, (h + 1) * head_dim).
    These are the same slices that the head gate scores at proj_in.
    """

    idxs = []
    for head_id in _normalize_head_ids(head_ids):
        start = int(head_id) * int(head_dim)
        idxs.extend(range(start, start + int(head_dim)))
    return idxs


def head_ids_to_qkv_out_idxs(head_ids, num_heads, head_dim):
    """Return fused qkv output-channel indices for whole attention heads.

    timm ViT qkv output is laid out as [q heads][k heads][v heads], where each
    component has num_heads contiguous head_dim-sized chunks.
    """

    idxs = []
    component_width = int(num_heads) * int(head_dim)
    for head_id in _normalize_head_ids(head_ids):
        head_id = int(head_id)
        for component_idx in range(3):
            start = component_idx * component_width + head_id * int(head_dim)
            idxs.extend(range(start, start + int(head_dim)))
    return idxs


def prune_selected_attention_heads(
    model,
    example_inputs,
    selected_heads,
    *,
    root="proj_in",
    check_groups=True,
):
    """Structurally prune explicit attention heads via Torch-Pruning groups.

    The selected_heads argument is already the pruning decision. This helper
    converts each selected head into channel indices and asks
    DependencyGraph.get_pruning_group(...).prune() to apply the dependency-aware
    structural change. For the default root="proj_in", the root slice is
    attn.proj.in_features, and Torch-Pruning propagates the matching deletion
    back to qkv output channels.

    Args:
        model: A classifier exposing model.encoder.blocks.
        example_inputs: Representative input tensor for DependencyGraph tracing.
        selected_heads: dict mapping block index to iterable head ids.
        root: "proj_in" prunes attn.proj input slices. "qkv_out" is a fallback
            that prunes fused qkv output slices directly.
        check_groups: Validate each pruning group before pruning.

    Returns:
        A metadata dict with selected heads and before/after attention shapes.
    """

    _validate_model_blocks(model)
    if root not in {"proj_in", "qkv_out"}:
        raise ValueError(f"Unsupported attention head pruning root: {root!r}.")

    normalized_heads = {
        int(block_idx): _normalize_head_ids(head_ids)
        for block_idx, head_ids in selected_heads.items()
        if head_ids
    }
    if not normalized_heads:
        return {
            "root": root,
            "selected_heads": {},
            "attention_metadata_before": {},
            "attention_metadata_after": {},
            "num_pruned_heads": 0,
        }

    metadata_before = _collect_attention_metadata(model, normalized_heads)
    _validate_selected_heads(model, normalized_heads)

    dg = tp.DependencyGraph().build_dependency(
        model,
        example_inputs=example_inputs,
    )

    for block_idx, head_ids in normalized_heads.items():
        block = model.encoder.blocks[block_idx]
        attn = block.attn
        if root == "proj_in":
            # Preferred path: score and prune from the same conceptual place,
            # the concatenated head output entering attn.proj. DependencyGraph
            # then keeps qkv and proj dimensions consistent.
            pruning_layer = attn.proj
            pruning_fn = tp.prune_linear_in_channels
            pruning_idxs = head_ids_to_proj_in_idxs(head_ids, attn.head_dim)
        else:
            # Debug/fallback path. This starts from fused qkv output slices
            # ([q heads][k heads][v heads]) and lets DependencyGraph propagate
            # forward to attn.proj input.
            pruning_layer = attn.qkv
            pruning_fn = tp.prune_linear_out_channels
            pruning_idxs = head_ids_to_qkv_out_idxs(
                head_ids,
                attn.num_heads,
                attn.head_dim,
            )

        group = dg.get_pruning_group(pruning_layer, pruning_fn, pruning_idxs)
        if check_groups and not dg.check_pruning_group(group):
            raise ValueError(
                f"Invalid attention-head pruning group for block {block_idx}, "
                f"heads={head_ids}, root={root!r}."
            )
        group.prune()

    # timm stores num_heads/attn_dim metadata separately from Linear shapes.
    # After structural pruning, refresh those fields so future forward passes
    # reshape qkv using the new head count.
    _refresh_attention_metadata(model, metadata_before)
    metadata_after = _collect_attention_metadata(model, normalized_heads)
    return {
        "root": root,
        "selected_heads": {
            int(block_idx): list(head_ids)
            for block_idx, head_ids in normalized_heads.items()
        },
        "attention_metadata_before": metadata_before,
        "attention_metadata_after": metadata_after,
        "num_pruned_heads": sum(len(head_ids) for head_ids in normalized_heads.values()),
    }


def dry_run_prune_attention_head(
    model,
    example_inputs,
    *,
    block_idx=0,
    head_id=0,
    root="proj_in",
):
    """Prune one head and verify shape propagation plus a forward pass.

    This is intended as the first integration check before head-gate Taylor is
    wired into prune_model(). It mutates the provided model.
    """

    _validate_model_blocks(model)
    model.eval()
    selected_heads = {int(block_idx): [int(head_id)]}
    metadata = prune_selected_attention_heads(
        model,
        example_inputs,
        selected_heads,
        root=root,
    )
    with torch.no_grad():
        output = model(example_inputs)
    metadata["forward_output_shape"] = tuple(output.shape)
    _validate_pruned_attention_shapes(metadata, expected_pruned_heads=1)
    return metadata


def _normalize_head_ids(head_ids):
    return tuple(dict.fromkeys(int(head_id) for head_id in head_ids))


def _normalize_score_dict(scores):
    normalized = {}
    for block_idx, score in scores.items():
        score = torch.as_tensor(score).detach().float().cpu()
        if score.ndim != 1:
            raise ValueError(f"Head scores for block {block_idx} must be 1D.")
        if score.numel() == 0:
            raise ValueError(f"Head scores for block {block_idx} must not be empty.")
        if not torch.isfinite(score).all():
            raise ValueError(f"Head scores for block {block_idx} contain non-finite values.")
        normalized[int(block_idx)] = score.flatten()
    return normalized


def _budget_from_ratio(num_heads, pruning_ratio, min_heads_per_block):
    if pruning_ratio is None:
        raise ValueError("pruning_ratio is required when no exact head count is provided.")
    pruning_ratio = float(pruning_ratio)
    if pruning_ratio < 0.0 or pruning_ratio >= 1.0:
        raise ValueError(f"pruning_ratio must be in [0, 1), got {pruning_ratio}.")
    max_prunable = max(0, int(num_heads) - int(min_heads_per_block))
    return min(max_prunable, int(math.ceil(pruning_ratio * int(num_heads))))


def _local_budget(block_idx, num_heads, pruning_ratio, pruned_head_counts, min_heads_per_block):
    if pruned_head_counts is None:
        budget = _budget_from_ratio(num_heads, pruning_ratio, min_heads_per_block)
    elif isinstance(pruned_head_counts, dict):
        budget = int(pruned_head_counts.get(int(block_idx), 0))
    else:
        budget = int(pruned_head_counts)
    max_prunable = max(0, int(num_heads) - int(min_heads_per_block))
    if budget < 0:
        raise ValueError(f"Pruned head count must be non-negative, got {budget}.")
    if budget > max_prunable:
        raise ValueError(
            f"Block {block_idx} can prune at most {max_prunable} heads while "
            f"preserving {min_heads_per_block}; requested {budget}."
        )
    return budget


def _select_local_attention_heads(
    scores,
    *,
    pruning_ratio,
    pruned_head_counts,
    min_heads_per_block,
):
    selected = {}
    for block_idx, score in scores.items():
        budget = _local_budget(
            block_idx,
            score.numel(),
            pruning_ratio,
            pruned_head_counts,
            min_heads_per_block,
        )
        if budget == 0:
            continue
        selected[int(block_idx)] = torch.argsort(score)[:budget].sort().values.tolist()
    return selected


def _global_budget(scores, pruning_ratio, pruned_head_counts, min_heads_per_block):
    if pruned_head_counts is None:
        total_heads = sum(score.numel() for score in scores.values())
        budget = _budget_from_ratio(total_heads, pruning_ratio, min_heads_per_block=0)
    elif isinstance(pruned_head_counts, dict):
        budget = sum(int(count) for count in pruned_head_counts.values())
    else:
        budget = int(pruned_head_counts)
    if budget < 0:
        raise ValueError(f"Pruned head count must be non-negative, got {budget}.")
    max_prunable = sum(
        max(0, score.numel() - int(min_heads_per_block))
        for score in scores.values()
    )
    if budget > max_prunable:
        raise ValueError(
            f"Global head budget {budget} exceeds max prunable heads {max_prunable} "
            f"with min_heads_per_block={min_heads_per_block}."
        )
    return budget


def _select_global_attention_heads(
    scores,
    *,
    pruning_ratio,
    pruned_head_counts,
    min_heads_per_block,
):
    budget = _global_budget(
        scores,
        pruning_ratio,
        pruned_head_counts,
        min_heads_per_block,
    )
    if budget == 0:
        return {}

    selected = {int(block_idx): [] for block_idx in scores}
    remaining = {
        int(block_idx): int(score.numel())
        for block_idx, score in scores.items()
    }
    candidates = []
    for block_idx, score in scores.items():
        for head_id, value in enumerate(score.tolist()):
            candidates.append((float(value), int(block_idx), int(head_id)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    total_selected = 0
    for _value, block_idx, head_id in candidates:
        if total_selected >= budget:
            break
        if remaining[block_idx] <= min_heads_per_block:
            continue
        selected[block_idx].append(head_id)
        remaining[block_idx] -= 1
        total_selected += 1

    if total_selected != budget:
        raise ValueError(
            f"Selected {total_selected} heads, but requested global budget {budget}."
        )
    return {
        block_idx: sorted(head_ids)
        for block_idx, head_ids in selected.items()
        if head_ids
    }


def _validate_model_blocks(model):
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "blocks"):
        raise ValueError("Attention head pruning needs model.encoder.blocks.")


def _validate_selected_heads(model, selected_heads):
    for block_idx, head_ids in selected_heads.items():
        if block_idx < 0 or block_idx >= len(model.encoder.blocks):
            raise ValueError(f"Block index {block_idx} is out of range.")
        attn = model.encoder.blocks[block_idx].attn
        invalid_heads = [
            head_id
            for head_id in head_ids
            if head_id < 0 or head_id >= int(attn.num_heads)
        ]
        if invalid_heads:
            raise ValueError(
                f"Block {block_idx} has {attn.num_heads} heads; invalid heads: "
                f"{invalid_heads}."
            )
        if len(head_ids) >= int(attn.num_heads):
            raise ValueError(
                f"Attention head pruning must leave at least one head in block {block_idx}."
            )


def _linear_shape(layer):
    return (int(layer.in_features), int(layer.out_features))


def _attention_metadata(block):
    attn = block.attn
    return {
        "num_heads": int(attn.num_heads),
        "head_dim": int(attn.head_dim),
        "attn_dim": int(getattr(attn, "attn_dim", attn.num_heads * attn.head_dim)),
        "qkv": _linear_shape(attn.qkv),
        "proj": _linear_shape(attn.proj),
    }


def _collect_attention_metadata(model, selected_heads):
    return {
        f"blocks.{int(block_idx)}.attn": _attention_metadata(
            model.encoder.blocks[int(block_idx)]
        )
        for block_idx in selected_heads
    }


def _refresh_attention_metadata(model, metadata_before):
    for block_name, before in metadata_before.items():
        block_idx = int(block_name.split(".")[1])
        attn = model.encoder.blocks[block_idx].attn
        original_head_dim = int(before["head_dim"])
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
        attn.num_heads = int(new_num_heads)
        attn.head_dim = original_head_dim
        attn.attn_dim = int(new_num_heads * original_head_dim)
        attn.scale = attn.head_dim ** -0.5


def _validate_pruned_attention_shapes(metadata, expected_pruned_heads):
    for block_name, before in metadata["attention_metadata_before"].items():
        after = metadata["attention_metadata_after"][block_name]
        pruned_heads = before["num_heads"] - after["num_heads"]
        if pruned_heads != expected_pruned_heads:
            raise ValueError(
                f"{block_name} pruned {pruned_heads} heads; expected "
                f"{expected_pruned_heads}."
            )
        expected_qkv_out = 3 * after["num_heads"] * before["head_dim"]
        expected_proj_in = after["num_heads"] * before["head_dim"]
        if after["qkv"][1] != expected_qkv_out:
            raise ValueError(
                f"{block_name} qkv.out_features={after['qkv'][1]}, expected "
                f"{expected_qkv_out}."
            )
        if after["proj"][0] != expected_proj_in:
            raise ValueError(
                f"{block_name} proj.in_features={after['proj'][0]}, expected "
                f"{expected_proj_in}."
            )
