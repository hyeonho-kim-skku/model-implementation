"""Joint MLP and whole-attention-head Gate-Taylor orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pruning.calibration import compute_taylor_gradients
from pruning.head_pruning import (
    prune_selected_attention_heads,
    select_attention_heads_by_score,
)
from pruning.head_taylor_cache import capture_head_taylor_scores
from pruning.importance import (
    AttentionHeadGateTaylorCollector,
    MLPGateTaylorCollector,
)
from pruning.structured_core import (
    build_pruner,
    collect_pruning_targets,
    validate_pruning_ratio,
)


JOINT_GATE_TAYLOR_MLP_CONFIG = {
    "gate_location": "fc2_in",
    "reduction": "sum_square",
    "aggregation": "samplewise",
}
JOINT_GATE_TAYLOR_HEAD_CONFIG = {
    "gate_location": "proj_in",
    "reduction": "sum_square",
    "aggregation": "samplewise",
}


@dataclass(frozen=True)
class JointSettings:
    mlp_pruning_ratio: float
    head_pruning_ratio: float
    round_to: int


@dataclass(frozen=True)
class JointPruningResult:
    calibration_config: dict
    selected_attention_heads: dict
    direct_head_pruning_metadata: dict
    num_pruned_mlp_groups: int
    num_pruned_heads: int

    @property
    def num_pruned_groups(self):
        return self.num_pruned_mlp_groups + self.num_pruned_heads


def normalize_joint_settings(
    *,
    pruning_modules,
    global_pruning,
    iterative_steps,
    head_pruning_root,
    mlp_pruning_ratio,
    head_pruning_ratio,
    round_to,
):
    if set(pruning_modules) != {"mlp", "head"} or len(pruning_modules) != 2:
        raise ValueError(
            "joint_gate_taylor requires pruning_modules='mlp,head' "
            "(order does not matter)."
        )
    if not global_pruning:
        raise ValueError("joint_gate_taylor requires global_pruning=True.")
    if iterative_steps != 1:
        raise ValueError("joint_gate_taylor currently supports iterative_steps=1.")
    if head_pruning_root not in {"proj_in", "qkv_out"}:
        raise ValueError(
            "head_pruning_root must be one of ['proj_in', 'qkv_out'], "
            f"got {head_pruning_root!r}."
        )
    return JointSettings(
        mlp_pruning_ratio=validate_pruning_ratio(
            "mlp_pruning_ratio",
            mlp_pruning_ratio,
        ),
        head_pruning_ratio=validate_pruning_ratio(
            "head_pruning_ratio",
            head_pruning_ratio,
        ),
        round_to=8 if round_to is None else round_to,
    )


def head_scores_for_selection(model, module_keyed_scores, target_block_indices=None):
    """Convert qkv-keyed head scores to block-index scores for selection."""

    block_scores = capture_head_taylor_scores(model, module_keyed_scores)
    if target_block_indices is None:
        if not block_scores:
            raise ValueError("Head gate Taylor scoring produced no attention-head scores.")
        return block_scores

    selected_scores = {}
    missing_blocks = []
    for block_idx in target_block_indices:
        block_idx = int(block_idx)
        score = block_scores.get(block_idx)
        if score is None:
            missing_blocks.append(block_idx)
            continue
        selected_scores[block_idx] = score
    if missing_blocks:
        raise ValueError(
            "Head gate Taylor scores are missing target blocks: "
            f"{missing_blocks}."
        )
    return selected_scores


def validate_joint_gate_taylor_scores(
    model,
    mlp_scores,
    head_scores,
    target_block_indices=None,
):
    """Validate dense-model score tensors before either structure is mutated."""

    targets = collect_pruning_targets(
        model,
        pruning_modules=("mlp", "head"),
        target_block_indices=target_block_indices,
    )
    for fc1 in targets.mlp_layers:
        score = mlp_scores.get(fc1)
        if score is None:
            raise ValueError("joint_gate_taylor is missing an MLP score tensor.")
        score = torch.as_tensor(score)
        if score.ndim != 1 or score.numel() != fc1.out_features:
            raise ValueError(
                "joint_gate_taylor MLP score shape must match fc1.out_features: "
                f"got {tuple(score.shape)}, expected ({fc1.out_features},)."
            )
        if not torch.isfinite(score).all():
            raise ValueError("joint_gate_taylor MLP scores contain non-finite values.")

    for qkv, num_heads in targets.num_heads.items():
        score = head_scores.get(qkv)
        if score is None:
            raise ValueError("joint_gate_taylor is missing an attention-head score tensor.")
        score = torch.as_tensor(score)
        if score.ndim != 1 or score.numel() != num_heads:
            raise ValueError(
                "joint_gate_taylor head score shape must match num_heads: "
                f"got {tuple(score.shape)}, expected ({num_heads},)."
            )
        if not torch.isfinite(score).all():
            raise ValueError("joint_gate_taylor head scores contain non-finite values.")


def run_joint_gate_taylor(
    *,
    model,
    example_inputs,
    target_block_indices,
    settings,
    head_pruning_root,
    calibration_dataset,
    calibration_batch_size,
    calibration_batches,
    calibration_split,
    calibration_seed,
    calibration_transform,
    calibration_objective,
    feature_dim_mask,
    feature_dim_mask_metadata,
    num_workers,
    data_root,
    device,
    use_existing_taylor_gradients,
    existing_calibration_config,
    existing_gate_taylor_scores,
    existing_head_gate_taylor_scores,
):
    """Collect both score families once, then prune MLP units and whole heads."""

    mlp_scores = (
        {} if existing_gate_taylor_scores is None else existing_gate_taylor_scores
    )
    head_scores = (
        {}
        if existing_head_gate_taylor_scores is None
        else existing_head_gate_taylor_scores
    )

    if use_existing_taylor_gradients:
        if existing_calibration_config is None:
            raise ValueError(
                "existing_calibration_config is required when "
                "joint_gate_taylor uses existing calibration."
            )
        if not mlp_scores or not head_scores:
            raise ValueError(
                "joint_gate_taylor existing calibration requires both "
                "existing_gate_taylor_scores and existing_head_gate_taylor_scores."
            )
        calibration_config = existing_calibration_config
    else:
        mlp_collector = MLPGateTaylorCollector(
            model=model,
            target_block_indices=target_block_indices,
            reduction=JOINT_GATE_TAYLOR_MLP_CONFIG["reduction"],
            gate_location=JOINT_GATE_TAYLOR_MLP_CONFIG["gate_location"],
            aggregation=JOINT_GATE_TAYLOR_MLP_CONFIG["aggregation"],
        )
        head_collector = AttentionHeadGateTaylorCollector(
            model=model,
            target_block_indices=target_block_indices,
            reduction=JOINT_GATE_TAYLOR_HEAD_CONFIG["reduction"],
            gate_location=JOINT_GATE_TAYLOR_HEAD_CONFIG["gate_location"],
            aggregation=JOINT_GATE_TAYLOR_HEAD_CONFIG["aggregation"],
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
                calibration_transform=calibration_transform,
                gate_taylor_collector=mlp_collector,
                head_gate_taylor_collector=head_collector,
                calibration_objective=calibration_objective,
                feature_dim_mask=feature_dim_mask,
                feature_dim_mask_metadata=feature_dim_mask_metadata,
            )
            mlp_scores.update(mlp_collector.final_scores())
            head_scores.update(head_collector.final_scores())
        finally:
            mlp_collector.remove()
            head_collector.remove()

    if not mlp_scores:
        raise ValueError("joint_gate_taylor calibration produced no MLP scores.")
    if not head_scores:
        raise ValueError(
            "joint_gate_taylor calibration produced no attention-head scores."
        )
    validate_joint_gate_taylor_scores(
        model,
        mlp_scores=mlp_scores,
        head_scores=head_scores,
        target_block_indices=target_block_indices,
    )

    block_head_scores = head_scores_for_selection(
        model,
        head_scores,
        target_block_indices,
    )
    selected_attention_heads = select_attention_heads_by_score(
        block_head_scores,
        pruning_ratio=settings.head_pruning_ratio,
        global_pruning=True,
        min_heads_per_block=1,
    )

    mlp_pruner, _ = build_pruner(
        model=model,
        example_inputs=example_inputs,
        importance="gate_taylor",
        pruning_ratio=settings.mlp_pruning_ratio,
        pruning_modules=("mlp",),
        target_block_indices=target_block_indices,
        iterative_steps=1,
        global_pruning=True,
        round_to=settings.round_to,
        gate_taylor_scores=mlp_scores,
    )
    history_before = len(mlp_pruner.pruning_history())
    mlp_pruner.step()
    num_pruned_mlp_groups = len(mlp_pruner.pruning_history()) - history_before

    direct_head_pruning_metadata = prune_selected_attention_heads(
        model=model,
        example_inputs=example_inputs,
        selected_heads=selected_attention_heads,
        root=head_pruning_root,
    )
    return JointPruningResult(
        calibration_config=calibration_config,
        selected_attention_heads=selected_attention_heads,
        direct_head_pruning_metadata=direct_head_pruning_metadata,
        num_pruned_mlp_groups=num_pruned_mlp_groups,
        num_pruned_heads=direct_head_pruning_metadata["num_pruned_heads"],
    )
