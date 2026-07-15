"""Attention head-dimension pruning helpers for ragged fused-QKV attention."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from models.ragged_attention import RaggedFusedQKVAttention, convert_vit_attention_to_ragged


VALID_ATTENTION_DIM_TARGETS = {"v_proj", "qk_pair", "qkv_shared"}
VALID_ATTENTION_DIM_CONSTRAINTS = {"free", "equal_head_width"}


def ensure_ragged_attention(model):
    """Convert every encoder attention block to RaggedFusedQKVAttention."""

    if not hasattr(model, "encoder") or not hasattr(model.encoder, "blocks"):
        raise ValueError("Attention-dim pruning needs model.encoder.blocks.")
    return convert_vit_attention_to_ragged(model.encoder)


def capture_attention_dim_metadata(model, target_block_indices=None):
    _validate_model_blocks(model)
    selected = _normalize_target_block_indices(target_block_indices, len(model.encoder.blocks))
    metadata = {}
    for block_idx, block in enumerate(model.encoder.blocks):
        if selected is not None and block_idx not in selected:
            continue
        attn = block.attn
        if not isinstance(attn, RaggedFusedQKVAttention):
            raise TypeError(
                "Attention-dim metadata requires RaggedFusedQKVAttention. "
                "Call ensure_ragged_attention() first."
            )
        metadata[int(block_idx)] = attn.export_ragged_metadata()
    return metadata


def select_attention_dims_by_score(
    scores,
    model,
    *,
    target,
    attention_dim_constraint="free",
    pruning_ratio=None,
    global_pruning=False,
    pruned_dim_counts=None,
    min_qk_dim_per_head=1,
    min_v_dim_per_head=1,
    target_block_indices=None,
):
    """Select low-scoring attention dimensions for explicit pruning.

    Args:
        scores: dict[int, Tensor] with shape [num_heads, original_head_dim].
        model: model with ragged attention metadata.
        target: one of v_proj, qk_pair, qkv_shared.
    """

    if target not in VALID_ATTENTION_DIM_TARGETS:
        raise ValueError(f"Unsupported attention dim target: {target!r}.")
    if attention_dim_constraint not in VALID_ATTENTION_DIM_CONSTRAINTS:
        raise ValueError(
            f"Unsupported attention dim constraint: {attention_dim_constraint!r}."
        )
    if pruning_ratio is None and pruned_dim_counts is None:
        raise ValueError("Either pruning_ratio or pruned_dim_counts must be provided.")

    _validate_model_blocks(model)
    selected_blocks = _normalize_target_block_indices(
        target_block_indices,
        len(model.encoder.blocks),
    )
    normalized_scores = _normalize_score_dict(scores)
    active = _active_dim_lookup(model, target, selected_blocks)

    if attention_dim_constraint == "equal_head_width":
        return _select_equal_head_width_dims(
            normalized_scores,
            active,
            target=target,
            pruning_ratio=pruning_ratio,
            pruned_dim_counts=pruned_dim_counts,
            min_qk_dim_per_head=min_qk_dim_per_head,
            min_v_dim_per_head=min_v_dim_per_head,
        )

    if global_pruning:
        return _select_global_dims(
            normalized_scores,
            active,
            target=target,
            pruning_ratio=pruning_ratio,
            pruned_dim_counts=pruned_dim_counts,
            min_qk_dim_per_head=min_qk_dim_per_head,
            min_v_dim_per_head=min_v_dim_per_head,
        )
    return _select_local_dims(
        normalized_scores,
        active,
        target=target,
        pruning_ratio=pruning_ratio,
        pruned_dim_counts=pruned_dim_counts,
        min_qk_dim_per_head=min_qk_dim_per_head,
        min_v_dim_per_head=min_v_dim_per_head,
    )


def prune_selected_attention_dims(
    model,
    selected_dims,
    *,
    target,
    target_block_indices=None,
):
    """Apply explicit attention-dim structural pruning to ragged attention."""

    if target not in VALID_ATTENTION_DIM_TARGETS:
        raise ValueError(f"Unsupported attention dim target: {target!r}.")
    _validate_model_blocks(model)
    selected_blocks = _normalize_target_block_indices(
        target_block_indices,
        len(model.encoder.blocks),
    )
    normalized = _normalize_selected_dims(selected_dims)
    if selected_blocks is not None:
        normalized = {
            block_idx: dims
            for block_idx, dims in normalized.items()
            if block_idx in selected_blocks
        }
    if not normalized:
        return {
            "target": target,
            "selected_attention_dims": {},
            "attention_dim_metadata_before": {},
            "attention_dim_metadata_after": {},
            "num_pruned_dims": 0,
        }

    metadata_before = capture_attention_dim_metadata(model, normalized.keys())
    for block_idx, dims in normalized.items():
        attn = model.encoder.blocks[int(block_idx)].attn
        if not isinstance(attn, RaggedFusedQKVAttention):
            raise TypeError(
                f"Block {block_idx} attention is not RaggedFusedQKVAttention."
            )
        _prune_block_attention_dims(attn, dims, target=target)

    metadata_after = capture_attention_dim_metadata(model, normalized.keys())
    return {
        "target": target,
        "selected_attention_dims": _serializable_selected_dims(normalized),
        "attention_dim_metadata_before": metadata_before,
        "attention_dim_metadata_after": metadata_after,
        "num_pruned_dims": sum(len(dims) for dims in normalized.values()),
    }


def validate_equal_head_width_metadata(metadata, *, target):
    """Validate that pruned metadata satisfies block-local equal head width."""

    if target not in VALID_ATTENTION_DIM_TARGETS:
        raise ValueError(f"Unsupported attention dim target: {target!r}.")
    invalid = {}
    for block_idx, block_metadata in metadata.items():
        qk_dims = [int(value) for value in block_metadata["qk_head_dims"]]
        v_dims = [int(value) for value in block_metadata["v_head_dims"]]
        block_errors = {}
        if target in {"qk_pair", "qkv_shared"} and len(set(qk_dims)) != 1:
            block_errors["qk_head_dims"] = qk_dims
        if target in {"v_proj", "qkv_shared"} and len(set(v_dims)) != 1:
            block_errors["v_head_dims"] = v_dims
        if block_errors:
            invalid[int(block_idx)] = block_errors
    if invalid:
        raise ValueError(f"Equal-head-width metadata check failed: {invalid}")
    return True


def dry_run_prune_attention_dim(
    model,
    example_inputs,
    *,
    block_idx=0,
    head_idx=0,
    dim_idx=0,
    target="v_proj",
):
    ensure_ragged_attention(model)
    model.eval()
    metadata = prune_selected_attention_dims(
        model,
        {int(block_idx): [{"head_idx": int(head_idx), "dim_idx": int(dim_idx)}]},
        target=target,
    )
    with torch.no_grad():
        output = model(example_inputs)
    metadata["forward_output_shape"] = tuple(output.shape)
    return metadata


def compute_attention_dim_mask_equivalence(
    model,
    example_inputs,
    selected_dims,
    *,
    target,
    target_block_indices=None,
):
    """Compare activation-mask pruning with structural pruning for the same dims."""

    if target not in VALID_ATTENTION_DIM_TARGETS:
        raise ValueError(f"Unsupported attention dim target: {target!r}.")

    mask_model = copy.deepcopy(model)
    structured_model = copy.deepcopy(model)
    ensure_ragged_attention(mask_model)
    ensure_ragged_attention(structured_model)
    mask_model.eval()
    structured_model.eval()

    normalized = _normalize_selected_dims(selected_dims)
    selected_blocks = _normalize_target_block_indices(
        target_block_indices,
        len(mask_model.encoder.blocks),
    )
    if selected_blocks is not None:
        normalized = {
            block_idx: dims
            for block_idx, dims in normalized.items()
            if block_idx in selected_blocks
        }

    handles = _register_attention_dim_activation_masks(
        mask_model,
        normalized,
        target=target,
    )
    try:
        prune_selected_attention_dims(
            structured_model,
            normalized,
            target=target,
            target_block_indices=target_block_indices,
        )
        with torch.no_grad():
            mask_output = mask_model(example_inputs)
            structured_output = structured_model(example_inputs)
    finally:
        for handle in handles:
            handle.remove()

    diff = (mask_output - structured_output).detach().abs()
    return {
        "target": target,
        "selected_attention_dims": _serializable_selected_dims(normalized),
        "max_diff": float(diff.max().item()),
        "mean_diff": float(diff.mean().item()),
        "mask_output_shape": tuple(mask_output.shape),
        "structured_output_shape": tuple(structured_output.shape),
    }


def _register_attention_dim_activation_masks(model, selected_dims, *, target):
    _validate_model_blocks(model)
    handles = []
    for block_idx, dims in selected_dims.items():
        attn = model.encoder.blocks[int(block_idx)].attn
        if not isinstance(attn, RaggedFusedQKVAttention):
            raise TypeError(
                f"Block {block_idx} attention is not RaggedFusedQKVAttention."
            )
        mask_columns = _qkv_mask_columns_for_selected_dims(attn, dims, target=target)
        if not mask_columns:
            continue

        def hook(_module, _inputs, output, columns=tuple(mask_columns)):
            masked = output.clone()
            masked[..., list(columns)] = 0
            return masked

        handles.append(attn.qkv.register_forward_hook(hook))
    return handles


def _qkv_mask_columns_for_selected_dims(attn, dims, *, target):
    selected_by_head = _selected_by_head(dims)
    columns = []
    q_rows = _component_row_indices(attn, "q")
    k_rows = _component_row_indices(attn, "k")
    v_rows = _component_row_indices(attn, "v")

    for head_idx, dim_idxs in selected_by_head.items():
        if head_idx < 0 or head_idx >= attn.num_heads:
            raise ValueError(f"head_idx out of range for attention mask: {head_idx}.")
        qk_dims = list(attn.qk_dim_indices[head_idx])
        v_dims = list(attn.v_dim_indices[head_idx])
        for dim_idx in dim_idxs:
            found = False
            if target in {"qk_pair", "qkv_shared"}:
                if dim_idx not in qk_dims:
                    raise ValueError(
                        f"Selected Q/K dim {dim_idx} is not active in head {head_idx}."
                    )
                pos = qk_dims.index(dim_idx)
                columns.extend([q_rows[head_idx][pos], k_rows[head_idx][pos]])
                found = True
            if target in {"v_proj", "qkv_shared"}:
                if dim_idx not in v_dims:
                    raise ValueError(
                        f"Selected V dim {dim_idx} is not active in head {head_idx}."
                    )
                pos = v_dims.index(dim_idx)
                columns.append(v_rows[head_idx][pos])
                found = True
            if not found:
                raise ValueError(f"No mask columns found for dim {dim_idx}.")
    return sorted(set(int(column) for column in columns))


def _prune_block_attention_dims(attn, dims, *, target):
    selected_by_head = _selected_by_head(dims)
    qk_remove = {head_idx: set() for head_idx in range(attn.num_heads)}
    v_remove = {head_idx: set() for head_idx in range(attn.num_heads)}
    for head_idx, dim_idxs in selected_by_head.items():
        if target in {"qk_pair", "qkv_shared"}:
            qk_remove[head_idx].update(dim_idxs)
        if target in {"v_proj", "qkv_shared"}:
            v_remove[head_idx].update(dim_idxs)

    old_q_indices = _component_row_indices(attn, "q")
    old_k_indices = _component_row_indices(attn, "k")
    old_v_indices = _component_row_indices(attn, "v")

    new_qk_dim_indices = []
    new_v_dim_indices = []
    keep_q_rows = []
    keep_k_rows = []
    keep_v_rows = []
    keep_proj_cols = []
    v_proj_cursor = 0

    for head_idx in range(attn.num_heads):
        qk_dims = list(attn.qk_dim_indices[head_idx])
        v_dims = list(attn.v_dim_indices[head_idx])
        q_rows = old_q_indices[head_idx]
        k_rows = old_k_indices[head_idx]
        v_rows = old_v_indices[head_idx]

        qk_keep_positions = [
            pos for pos, dim_idx in enumerate(qk_dims)
            if dim_idx not in qk_remove[head_idx]
        ]
        v_keep_positions = [
            pos for pos, dim_idx in enumerate(v_dims)
            if dim_idx not in v_remove[head_idx]
        ]
        if not qk_keep_positions:
            raise ValueError(f"Pruning would remove all Q/K dims from head {head_idx}.")
        if not v_keep_positions:
            raise ValueError(f"Pruning would remove all V dims from head {head_idx}.")

        new_qk_dim_indices.append([qk_dims[pos] for pos in qk_keep_positions])
        new_v_dim_indices.append([v_dims[pos] for pos in v_keep_positions])
        keep_q_rows.extend(q_rows[pos] for pos in qk_keep_positions)
        keep_k_rows.extend(k_rows[pos] for pos in qk_keep_positions)
        keep_v_rows.extend(v_rows[pos] for pos in v_keep_positions)

        keep_proj_cols.extend(v_proj_cursor + pos for pos in v_keep_positions)
        v_proj_cursor += len(v_dims)

    keep_qkv_rows = keep_q_rows + keep_k_rows + keep_v_rows
    attn.qkv = _prune_linear_out(attn.qkv, keep_qkv_rows)
    attn.proj = _prune_linear_in(attn.proj, keep_proj_cols)
    attn.qk_dim_indices = new_qk_dim_indices
    attn.v_dim_indices = new_v_dim_indices
    attn._refresh_metadata()
    attn._validate_shapes()


def _component_row_indices(attn, component):
    ranges = attn.head_component_ranges(component)
    return [list(range(start, end)) for start, end in ranges]


def _prune_linear_out(linear, keep_rows):
    keep_rows = [int(idx) for idx in keep_rows]
    new_linear = nn.Linear(
        linear.in_features,
        len(keep_rows),
        bias=linear.bias is not None,
    )
    new_linear.to(device=linear.weight.device, dtype=linear.weight.dtype)
    new_linear.weight.data.copy_(linear.weight.data[keep_rows, :])
    if linear.bias is not None:
        new_linear.bias.data.copy_(linear.bias.data[keep_rows])
    return new_linear


def _prune_linear_in(linear, keep_cols):
    keep_cols = [int(idx) for idx in keep_cols]
    new_linear = nn.Linear(
        len(keep_cols),
        linear.out_features,
        bias=linear.bias is not None,
    )
    new_linear.to(device=linear.weight.device, dtype=linear.weight.dtype)
    new_linear.weight.data.copy_(linear.weight.data[:, keep_cols])
    if linear.bias is not None:
        new_linear.bias.data.copy_(linear.bias.data)
    return new_linear


def _active_dim_lookup(model, target, selected_blocks):
    active = {}
    for block_idx, block in enumerate(model.encoder.blocks):
        if selected_blocks is not None and block_idx not in selected_blocks:
            continue
        attn = block.attn
        if not isinstance(attn, RaggedFusedQKVAttention):
            raise TypeError("Attention-dim pruning requires ragged attention.")
        block_active = []
        for head_idx in range(attn.num_heads):
            if target == "v_proj":
                dims = set(attn.v_dim_indices[head_idx])
            elif target == "qk_pair":
                dims = set(attn.qk_dim_indices[head_idx])
            else:
                dims = set(attn.qk_dim_indices[head_idx]) & set(attn.v_dim_indices[head_idx])
            block_active.append(sorted(dims))
        active[int(block_idx)] = block_active
    return active


def _select_global_dims(
    scores,
    active,
    *,
    target,
    pruning_ratio,
    pruned_dim_counts,
    min_qk_dim_per_head,
    min_v_dim_per_head,
):
    candidates = _candidate_dims(scores, active)
    budget = _global_budget(candidates, pruning_ratio, pruned_dim_counts)
    remaining = _remaining_counts(active)
    selected = {block_idx: [] for block_idx in active}

    total_selected = 0
    for _score, block_idx, head_idx, dim_idx in candidates:
        if total_selected >= budget:
            break
        if not _can_prune(
            remaining,
            block_idx,
            head_idx,
            target=target,
            min_qk_dim_per_head=min_qk_dim_per_head,
            min_v_dim_per_head=min_v_dim_per_head,
        ):
            continue
        selected[block_idx].append({"head_idx": head_idx, "dim_idx": dim_idx})
        _decrement_remaining(remaining, block_idx, head_idx, target)
        total_selected += 1
    if total_selected != budget:
        raise ValueError(f"Selected {total_selected} dims, requested {budget}.")
    return {block_idx: dims for block_idx, dims in selected.items() if dims}


def _select_local_dims(
    scores,
    active,
    *,
    target,
    pruning_ratio,
    pruned_dim_counts,
    min_qk_dim_per_head,
    min_v_dim_per_head,
):
    selected = {}
    for block_idx, block_active in active.items():
        block_candidates = _candidate_dims(
            {block_idx: scores[block_idx]},
            {block_idx: block_active},
        )
        budget = _local_budget(block_idx, block_candidates, pruning_ratio, pruned_dim_counts)
        remaining = _remaining_counts({block_idx: block_active})
        for _score, _block_idx, head_idx, dim_idx in block_candidates:
            if len(selected.get(block_idx, [])) >= budget:
                break
            if not _can_prune(
                remaining,
                block_idx,
                head_idx,
                target=target,
                min_qk_dim_per_head=min_qk_dim_per_head,
                min_v_dim_per_head=min_v_dim_per_head,
            ):
                continue
            selected.setdefault(block_idx, []).append({"head_idx": head_idx, "dim_idx": dim_idx})
            _decrement_remaining(remaining, block_idx, head_idx, target)
    return selected


def _select_equal_head_width_dims(
    scores,
    active,
    *,
    target,
    pruning_ratio,
    pruned_dim_counts,
    min_qk_dim_per_head,
    min_v_dim_per_head,
):
    selected = {}
    required_min = _required_min_dim_per_head(
        target,
        min_qk_dim_per_head=min_qk_dim_per_head,
        min_v_dim_per_head=min_v_dim_per_head,
    )
    for block_idx, block_active in active.items():
        active_widths = [len(dim_indices) for dim_indices in block_active]
        if len(set(active_widths)) != 1:
            raise ValueError(
                "equal_head_width selection requires every head in a block to "
                f"start with the same active width. Block {block_idx}: {active_widths}."
            )
        active_width = active_widths[0]
        budget = _equal_head_width_budget(
            block_idx,
            active_width,
            pruning_ratio=pruning_ratio,
            pruned_dim_counts=pruned_dim_counts,
        )
        max_budget = active_width - required_min
        if budget < 0 or budget > max_budget:
            raise ValueError(
                "Invalid equal_head_width per-head budget for "
                f"block {block_idx}: {budget}. Max allowed is {max_budget} "
                f"with active_width={active_width}, required_min={required_min}."
            )
        if budget == 0:
            continue
        score = scores.get(block_idx)
        if score is None:
            raise ValueError(f"Missing attention-dim scores for block {block_idx}.")
        for head_idx, dim_indices in enumerate(block_active):
            head_candidates = [
                (
                    float(score[head_idx, dim_idx].item()),
                    int(dim_idx),
                )
                for dim_idx in dim_indices
            ]
            head_candidates.sort(key=lambda item: (item[0], item[1]))
            for _score, dim_idx in head_candidates[:budget]:
                selected.setdefault(block_idx, []).append(
                    {"head_idx": int(head_idx), "dim_idx": int(dim_idx)}
                )
    return selected


def _equal_head_width_budget(
    block_idx,
    active_width,
    *,
    pruning_ratio,
    pruned_dim_counts,
):
    if pruned_dim_counts is not None:
        if isinstance(pruned_dim_counts, dict):
            return int(pruned_dim_counts.get(int(block_idx), 0))
        return int(pruned_dim_counts)
    ratio = float(pruning_ratio)
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"pruning_ratio must be in [0, 1), got {ratio}.")
    return int(math.ceil(ratio * int(active_width)))


def _required_min_dim_per_head(
    target,
    *,
    min_qk_dim_per_head,
    min_v_dim_per_head,
):
    required = 0
    if target in {"qk_pair", "qkv_shared"}:
        required = max(required, int(min_qk_dim_per_head))
    if target in {"v_proj", "qkv_shared"}:
        required = max(required, int(min_v_dim_per_head))
    return required


def _candidate_dims(scores, active):
    candidates = []
    for block_idx, block_active in active.items():
        score = scores.get(block_idx)
        if score is None:
            raise ValueError(f"Missing attention-dim scores for block {block_idx}.")
        for head_idx, dim_indices in enumerate(block_active):
            for dim_idx in dim_indices:
                candidates.append(
                    (
                        float(score[head_idx, dim_idx].item()),
                        int(block_idx),
                        int(head_idx),
                        int(dim_idx),
                    )
                )
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates


def _global_budget(candidates, pruning_ratio, pruned_dim_counts):
    if pruned_dim_counts is not None:
        budget = sum(int(value) for value in pruned_dim_counts.values()) if isinstance(pruned_dim_counts, dict) else int(pruned_dim_counts)
    else:
        ratio = float(pruning_ratio)
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError(f"pruning_ratio must be in [0, 1), got {ratio}.")
        budget = int(math.ceil(ratio * len(candidates)))
    if budget < 0 or budget > len(candidates):
        raise ValueError(f"Invalid global attention-dim budget: {budget}.")
    return budget


def _local_budget(block_idx, candidates, pruning_ratio, pruned_dim_counts):
    if pruned_dim_counts is not None:
        if isinstance(pruned_dim_counts, dict):
            return int(pruned_dim_counts.get(int(block_idx), 0))
        return int(pruned_dim_counts)
    ratio = float(pruning_ratio)
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"pruning_ratio must be in [0, 1), got {ratio}.")
    return int(math.ceil(ratio * len(candidates)))


def _remaining_counts(active):
    return {
        block_idx: {
            head_idx: {"qk": len(dim_indices), "v": len(dim_indices)}
            for head_idx, dim_indices in enumerate(block_active)
        }
        for block_idx, block_active in active.items()
    }


def _can_prune(
    remaining,
    block_idx,
    head_idx,
    *,
    target,
    min_qk_dim_per_head,
    min_v_dim_per_head,
):
    counts = remaining[block_idx][head_idx]
    if target in {"qk_pair", "qkv_shared"} and counts["qk"] <= int(min_qk_dim_per_head):
        return False
    if target in {"v_proj", "qkv_shared"} and counts["v"] <= int(min_v_dim_per_head):
        return False
    return True


def _decrement_remaining(remaining, block_idx, head_idx, target):
    if target in {"qk_pair", "qkv_shared"}:
        remaining[block_idx][head_idx]["qk"] -= 1
    if target in {"v_proj", "qkv_shared"}:
        remaining[block_idx][head_idx]["v"] -= 1


def _normalize_score_dict(scores):
    normalized = {}
    for block_idx, score in scores.items():
        score = torch.as_tensor(score).detach().float().cpu()
        if score.ndim != 2:
            raise ValueError(f"Attention-dim scores for block {block_idx} must be 2D.")
        if not torch.isfinite(score).all():
            raise ValueError(f"Attention-dim scores for block {block_idx} contain non-finite values.")
        normalized[int(block_idx)] = score
    return normalized


def _normalize_selected_dims(selected_dims):
    normalized = {}
    for block_idx, dims in selected_dims.items():
        block_dims = []
        for item in dims:
            block_dims.append(
                {
                    "head_idx": int(item["head_idx"]),
                    "dim_idx": int(item["dim_idx"]),
                }
            )
        normalized[int(block_idx)] = block_dims
    return normalized


def _selected_by_head(dims):
    selected = {}
    for item in dims:
        selected.setdefault(int(item["head_idx"]), set()).add(int(item["dim_idx"]))
    return selected


def _serializable_selected_dims(selected_dims):
    return {
        int(block_idx): [
            {"head_idx": int(item["head_idx"]), "dim_idx": int(item["dim_idx"])}
            for item in dims
        ]
        for block_idx, dims in selected_dims.items()
    }


def _normalize_target_block_indices(target_block_indices, num_blocks):
    if target_block_indices is None:
        return None
    if isinstance(target_block_indices, str):
        if not target_block_indices.strip():
            return None
        indices = tuple(int(item.strip()) for item in target_block_indices.split(",") if item.strip())
    else:
        indices = tuple(int(item) for item in target_block_indices)
    invalid = [idx for idx in indices if idx < 0 or idx >= int(num_blocks)]
    if invalid:
        raise ValueError(f"target_block_indices out of range: {invalid}.")
    return tuple(dict.fromkeys(indices))


def _validate_model_blocks(model):
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "blocks"):
        raise ValueError("Attention-dim pruning needs model.encoder.blocks.")
