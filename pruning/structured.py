"""Structured pruning pipeline for dense timm ViT-style classifiers."""

from __future__ import annotations

import os

import torch

from pruning.artifact import (
    build_pruning_artifact as _build_pruning_artifact,
    build_target_pruning_summary as _build_target_pruning_summary,
    collect_attention_metadata as _collect_attention_metadata,
    collect_target_shapes as _collect_target_shapes,
    print_pruning_summary as _print_pruning_summary,
    print_shape_changes as _print_shape_changes,
    refresh_attention_metadata as _refresh_attention_metadata,
)
from pruning.calibration import (
    VALID_TAYLOR_CALIBRATION_OBJECTIVES,
    compute_taylor_gradients,
)
from pruning.head_pruning import (
    prune_selected_attention_heads,
    select_attention_heads_by_score,
)
from pruning.importance import (
    AttentionHeadGateTaylorCollector,
    MLPActivationTaylorCollector,
    MLPGateTaylorCollector,
    VALID_ACTIVATION_TAYLOR_REDUCTIONS,
    VALID_GATE_TAYLOR_AGGREGATIONS,
    VALID_GATE_TAYLOR_REDUCTIONS,
    VALID_GATE_TAYLOR_LOCATIONS,
    VALID_HEAD_GATE_TAYLOR_LOCATIONS,
)
from pruning.joint import (
    JOINT_GATE_TAYLOR_HEAD_CONFIG,
    JOINT_GATE_TAYLOR_MLP_CONFIG,
    head_scores_for_selection as _head_scores_for_selection,
    normalize_joint_settings,
    run_joint_gate_taylor,
    validate_joint_gate_taylor_scores as _validate_joint_gate_taylor_scores,
)
from pruning.structured_core import (
    build_pruner as _build_pruner,
    count_ops_and_params as _count_ops_and_params,
    normalize_pruning_modules as _normalize_pruning_modules,
    normalize_target_block_indices as _normalize_target_block_indices,
)
from pruning.isomorphic import prune_model_isomorphic


def prune_model(
    model,
    model_config,
    source_info,
    output_dir,
    output_path=None,
    importance="magnitude",
    pruning_ratio=0.2,
    mlp_pruning_ratio=0.4,
    head_pruning_ratio=0.4,
    isomorphic_pruning_ratio=0.2,
    isomorphic_head_pruning_ratio=0.2,
    isomorphic_head_dim_pruning_ratio=None,
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
    calibration_transform="default",
    activation_taylor_reduction="sum_abs",
    gate_taylor_reduction="sum_abs",
    gate_taylor_location="fc1_out",
    gate_taylor_aggregation="elementwise",
    head_gate_taylor_reduction="sum_abs",
    head_gate_taylor_location="proj_in",
    head_gate_taylor_aggregation="samplewise",
    head_pruning_root="proj_in",
    calibration_objective="ce",
    feature_dim_mask=None,
    feature_dim_mask_metadata=None,
    num_workers=4,
    data_root="./data",
    inspect_groups=False,
    use_existing_taylor_gradients=False,
    existing_calibration_config=None,
    existing_activation_taylor_scores=None,
    existing_gate_taylor_scores=None,
    existing_head_gate_taylor_scores=None,
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

    Three pruning styles live here:
    - Torch-Pruning ranked pruning for magnitude/taylor/MLP activation/gate
      Taylor. BasePruner owns both ranking and structural deletion.
    - Direct head_gate_taylor pruning. The project computes head scores and
      selects concrete head ids itself, then uses Torch-Pruning's
      DependencyGraph only to remove the matching qkv/proj slices safely.
    - Joint gate Taylor pruning. MLP and whole-head scores are collected in one
      calibration pass, ranked in separate global scopes, then structurally
      removed from the same dense source model.
    """

    model = model.to(device)
    model.eval()

    # Keep the paper method in its own package.  This early dispatch makes the
    # legacy magnitude/Taylor/gate/joint paths below byte-for-byte independent
    # of Isomorphic Pruning's GroupTaylor and DependencyGraph policy.
    if (importance or "").strip().lower() == "isomorphic_taylor":
        return prune_model_isomorphic(
            model=model,
            model_config=model_config,
            source_info=source_info,
            output_dir=output_dir,
            output_path=output_path,
            calibration_dataset=calibration_dataset,
            calibration_batch_size=calibration_batch_size,
            calibration_batches=calibration_batches,
            calibration_split=calibration_split,
            calibration_seed=calibration_seed,
            calibration_transform=calibration_transform,
            num_workers=num_workers,
            data_root=data_root,
            isomorphic_pruning_ratio=isomorphic_pruning_ratio,
            isomorphic_head_pruning_ratio=isomorphic_head_pruning_ratio,
            isomorphic_head_dim_pruning_ratio=isomorphic_head_dim_pruning_ratio,
            round_to=round_to,
            inspect_groups=inspect_groups,
            device=device,
        )

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
    joint_gate_taylor = importance_type == "joint_gate_taylor"
    joint_settings = None
    if joint_gate_taylor:
        joint_settings = normalize_joint_settings(
            pruning_modules=normalized_modules,
            global_pruning=global_pruning,
            iterative_steps=iterative_steps,
            head_pruning_root=head_pruning_root,
            mlp_pruning_ratio=mlp_pruning_ratio,
            head_pruning_ratio=head_pruning_ratio,
            round_to=round_to,
        )
        mlp_pruning_ratio = joint_settings.mlp_pruning_ratio
        head_pruning_ratio = joint_settings.head_pruning_ratio
        round_to = joint_settings.round_to

    importance_group_reduction = (
        None if importance_type == "head_gate_taylor" else "mean"
    )
    importance_normalizer = (
        None
        if importance_type in {"gate_taylor", "head_gate_taylor", "joint_gate_taylor"}
        else "mean"
    )
    activation_taylor_scores = None
    gate_taylor_scores = None
    head_gate_taylor_scores = None
    direct_head_pruning_metadata = None
    selected_attention_heads = None
    num_pruned_mlp_groups = None
    num_pruned_heads = None
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
    if calibration_objective not in VALID_TAYLOR_CALIBRATION_OBJECTIVES:
        raise ValueError(
            "calibration_objective must be one of "
            f"{sorted(VALID_TAYLOR_CALIBRATION_OBJECTIVES)}, got {calibration_objective!r}."
        )
    if calibration_objective == "feature_dim_masked_ce" and feature_dim_mask is None:
        raise ValueError("feature_dim_mask is required for feature_dim_masked_ce.")
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
        if gate_taylor_aggregation not in VALID_GATE_TAYLOR_AGGREGATIONS:
            raise ValueError(
                "gate_taylor_aggregation must be one of "
                f"{sorted(VALID_GATE_TAYLOR_AGGREGATIONS)}, "
                f"got {gate_taylor_aggregation!r}."
            )
        if normalized_modules != ("mlp",):
            raise ValueError("gate_taylor currently supports pruning_modules='mlp' only.")
        gate_taylor_scores = (
            {} if existing_gate_taylor_scores is None else existing_gate_taylor_scores
        )
    if importance_type == "head_gate_taylor":
        if head_gate_taylor_reduction not in VALID_GATE_TAYLOR_REDUCTIONS:
            raise ValueError(
                "head_gate_taylor reduction must be one of "
                f"{sorted(VALID_GATE_TAYLOR_REDUCTIONS)}, "
                f"got {head_gate_taylor_reduction!r}."
            )
        if head_gate_taylor_location not in VALID_HEAD_GATE_TAYLOR_LOCATIONS:
            raise ValueError(
                "head_gate_taylor_location must be one of "
                f"{sorted(VALID_HEAD_GATE_TAYLOR_LOCATIONS)}, "
                f"got {head_gate_taylor_location!r}."
            )
        if head_gate_taylor_aggregation not in VALID_GATE_TAYLOR_AGGREGATIONS:
            raise ValueError(
                "head_gate_taylor_aggregation must be one of "
                f"{sorted(VALID_GATE_TAYLOR_AGGREGATIONS)}, "
                f"got {head_gate_taylor_aggregation!r}."
            )
        if head_pruning_root not in {"proj_in", "qkv_out"}:
            raise ValueError(
                "head_pruning_root must be one of ['proj_in', 'qkv_out'], "
                f"got {head_pruning_root!r}."
            )
        if normalized_modules != ("head",):
            raise ValueError(
                "head_gate_taylor currently supports pruning_modules='head' only."
            )
        head_gate_taylor_scores = (
            {}
            if existing_head_gate_taylor_scores is None
            else existing_head_gate_taylor_scores
        )
    if normalized_modules:
        if joint_gate_taylor:
            joint_result = run_joint_gate_taylor(
                model=model,
                example_inputs=example_inputs,
                target_block_indices=normalized_target_block_indices,
                settings=joint_settings,
                head_pruning_root=head_pruning_root,
                calibration_dataset=calibration_dataset,
                calibration_batch_size=calibration_batch_size,
                calibration_batches=calibration_batches,
                calibration_split=calibration_split,
                calibration_seed=calibration_seed,
                calibration_transform=calibration_transform,
                calibration_objective=calibration_objective,
                feature_dim_mask=feature_dim_mask,
                feature_dim_mask_metadata=feature_dim_mask_metadata,
                num_workers=num_workers,
                data_root=data_root,
                device=device,
                use_existing_taylor_gradients=use_existing_taylor_gradients,
                existing_calibration_config=existing_calibration_config,
                existing_gate_taylor_scores=existing_gate_taylor_scores,
                existing_head_gate_taylor_scores=existing_head_gate_taylor_scores,
            )
            calibration_config = joint_result.calibration_config
            selected_attention_heads = joint_result.selected_attention_heads
            direct_head_pruning_metadata = (
                joint_result.direct_head_pruning_metadata
            )
            num_pruned_mlp_groups = joint_result.num_pruned_mlp_groups
            num_pruned_heads = joint_result.num_pruned_heads
            num_pruned_groups = joint_result.num_pruned_groups
        elif importance_type == "head_gate_taylor":
            # Direct attention-head path:
            # 1. Obtain qkv-keyed [num_heads] scores, either from calibration
            #    below or from an existing snapshot supplied by a sweep.
            # 2. Convert module-keyed scores to block-indexed scores so the
            #    target layer can be selected on copied trial models.
            # 3. Pick the lowest-scoring head ids.
            # 4. Apply dependency-aware structural deletion.
            #
            # This branch intentionally bypasses BasePruner.step(): the score
            # unit is a whole attention head, not an arbitrary channel group.
            if iterative_steps != 1:
                raise ValueError(
                    "Head gate Taylor pruning currently supports iterative_steps=1. "
                    "For iterative pruning, recompute head scores before each step."
                )
            if use_existing_taylor_gradients:
                # Sensitivity sweeps calibrate once on the dense model, then
                # restore the same head scores on every trial model. In that
                # mode this branch should not run the data loader again.
                if existing_calibration_config is None:
                    raise ValueError(
                        "existing_calibration_config is required when "
                        "use_existing_taylor_gradients=True."
                    )
                if not head_gate_taylor_scores:
                    raise ValueError(
                        "existing_head_gate_taylor_scores is required when "
                        "head_gate_taylor uses existing calibration."
                    )
                calibration_config = existing_calibration_config
            else:
                # One-shot pruning path: attach temporary gates to attn.proj
                # inputs, run calibration backward passes, and collect one score
                # per head in each selected block.
                head_gate_taylor_collector = AttentionHeadGateTaylorCollector(
                    model=model,
                    target_block_indices=normalized_target_block_indices,
                    reduction=head_gate_taylor_reduction,
                    gate_location=head_gate_taylor_location,
                    aggregation=head_gate_taylor_aggregation,
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
                        calibration_objective=calibration_objective,
                        feature_dim_mask=feature_dim_mask,
                        feature_dim_mask_metadata=feature_dim_mask_metadata,
                        head_gate_taylor_collector=head_gate_taylor_collector,
                    )
                    head_gate_taylor_scores.update(
                        head_gate_taylor_collector.final_scores()
                    )
                finally:
                    head_gate_taylor_collector.remove()

            block_head_scores = _head_scores_for_selection(
                model,
                head_gate_taylor_scores,
                normalized_target_block_indices,
            )
            # select_attention_heads_by_score returns explicit ids, e.g.
            # {11: [0, 2, 5]}. These ids are stored in the artifact so later
            # analysis can inspect exactly which heads were removed.
            selected_attention_heads = select_attention_heads_by_score(
                block_head_scores,
                pruning_ratio=pruning_ratio,
                global_pruning=global_pruning,
            )
            direct_head_pruning_metadata = prune_selected_attention_heads(
                model=model,
                example_inputs=example_inputs,
                selected_heads=selected_attention_heads,
                root=head_pruning_root,
            )
            num_pruned_groups = direct_head_pruning_metadata["num_pruned_heads"]
            num_pruned_heads = direct_head_pruning_metadata["num_pruned_heads"]
        else:
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
                            aggregation=gate_taylor_aggregation,
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
                            activation_taylor_collector=activation_taylor_collector,
                            gate_taylor_collector=gate_taylor_collector,
                            calibration_objective=calibration_objective,
                            feature_dim_mask=feature_dim_mask,
                            feature_dim_mask_metadata=feature_dim_mask_metadata,
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
            if "mlp" in normalized_modules:
                num_pruned_mlp_groups = num_pruned_groups
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
        pruning_ratio=None if joint_gate_taylor else pruning_ratio,
        mlp_pruning_ratio=mlp_pruning_ratio if joint_gate_taylor else None,
        head_pruning_ratio=head_pruning_ratio if joint_gate_taylor else None,
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
        gate_taylor_reduction=(
            JOINT_GATE_TAYLOR_MLP_CONFIG["reduction"]
            if joint_gate_taylor
            else gate_taylor_reduction if importance_type == "gate_taylor" else None
        ),
        gate_taylor_location=(
            JOINT_GATE_TAYLOR_MLP_CONFIG["gate_location"]
            if joint_gate_taylor
            else gate_taylor_location if importance_type == "gate_taylor" else None
        ),
        gate_taylor_aggregation=(
            JOINT_GATE_TAYLOR_MLP_CONFIG["aggregation"]
            if joint_gate_taylor
            else gate_taylor_aggregation if importance_type == "gate_taylor" else None
        ),
        head_gate_taylor_reduction=(
            JOINT_GATE_TAYLOR_HEAD_CONFIG["reduction"]
            if joint_gate_taylor
            else head_gate_taylor_reduction if importance_type == "head_gate_taylor" else None
        ),
        head_gate_taylor_location=(
            JOINT_GATE_TAYLOR_HEAD_CONFIG["gate_location"]
            if joint_gate_taylor
            else head_gate_taylor_location if importance_type == "head_gate_taylor" else None
        ),
        head_gate_taylor_aggregation=(
            JOINT_GATE_TAYLOR_HEAD_CONFIG["aggregation"]
            if joint_gate_taylor
            else head_gate_taylor_aggregation if importance_type == "head_gate_taylor" else None
        ),
        head_pruning_root=(
            head_pruning_root
            if importance_type in {"head_gate_taylor", "joint_gate_taylor"}
            else None
        ),
        base_macs=base_macs,
        base_params=base_params,
        pruned_macs=pruned_macs,
        pruned_params=pruned_params,
        num_pruned_groups=num_pruned_groups,
        target_pruning_summary=target_pruning_summary,
        attention_metadata_before=attention_metadata_before,
        attention_metadata_after=attention_metadata_after,
        selected_attention_heads=selected_attention_heads,
        direct_head_pruning_metadata=direct_head_pruning_metadata,
        num_pruned_mlp_groups=num_pruned_mlp_groups,
        num_pruned_heads=num_pruned_heads,
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
        if joint_gate_taylor:
            print(
                "[Pruning] joint ratios: "
                f"mlp={mlp_pruning_ratio}, head={head_pruning_ratio}"
            )
        else:
            print(f"[Pruning] ratio: {pruning_ratio}")
        print(f"[Pruning] groups pruned: {num_pruned_groups}")
        _print_pruning_summary(target_pruning_summary, max_lines=0)
        print(f"[Pruning] MACs: {base_macs:,} -> {pruned_macs:,}")
        print(f"[Pruning] Params: {base_params:,} -> {pruned_params:,}")
        if save_artifact:
            print(f"[Pruning] saved to: {output_path}")

    return artifact
