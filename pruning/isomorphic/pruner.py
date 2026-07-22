"""Artifact-aware adapter around Torch-Pruning's Isomorphic Pruning engine."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch_pruning as tp

from pruning.artifact import build_pruning_artifact
from pruning.isomorphic.adapter import (
    build_structure_summary,
    collect_vit_attention_qkv,
    collect_vit_structure,
    refresh_vit_attention_metadata,
)
from pruning.isomorphic.calibration import accumulate_group_taylor_gradients
from pruning.structured_core import count_ops_and_params


REFERENCE_REPOSITORY = "https://github.com/VainF/Isomorphic-Pruning"
REFERENCE_METHOD = "ECCV 2024 Isomorphic Pruning; Torch-Pruning MetaPruner(isomorphic=True)"


def _ratio(name, value):
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}.")
    return value


def _build_pruner(model, example_inputs, *, pruning_ratio, head_pruning_ratio, head_dim_pruning_ratio, round_to):
    qkv_by_block = collect_vit_attention_qkv(model)
    qkv_layers = tuple(qkv_by_block.values())
    num_heads = {qkv: model.encoder.blocks[index].attn.num_heads for index, qkv in qkv_by_block.items()}
    ratio_dict = None
    if head_dim_pruning_ratio is not None:
        ratio_dict = {qkv_layers: _ratio("isomorphic_head_dim_pruning_ratio", head_dim_pruning_ratio)}
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=tp.importance.GroupTaylorImportance(),
        global_pruning=True,
        isomorphic=True,
        pruning_ratio=_ratio("isomorphic_pruning_ratio", pruning_ratio),
        pruning_ratio_dict=ratio_dict,
        ignored_layers=[model.classifier],
        round_to=round_to,
        root_module_types=[nn.Linear],
        num_heads=num_heads,
        prune_head_dims=True,
        prune_num_heads=True,
        head_pruning_ratio=_ratio("isomorphic_head_pruning_ratio", head_pruning_ratio),
    )
    return pruner, qkv_by_block


def prune_model_isomorphic(
    *,
    model,
    model_config,
    source_info,
    output_dir,
    output_path=None,
    calibration_dataset,
    calibration_batch_size=64,
    calibration_batches=100,
    calibration_split="train",
    calibration_seed=None,
    calibration_transform="default",
    num_workers=4,
    data_root="./data",
    isomorphic_pruning_ratio=0.2,
    isomorphic_head_pruning_ratio=0.2,
    isomorphic_head_dim_pruning_ratio=None,
    round_to=2,
    inspect_groups=False,
    device="cpu",
):
    """Run the original method's full ViT structural pruning policy.

    ``isomorphic_pruning_ratio`` affects the normal isomorphic scopes (including
    embedding width and FFN hidden dimensions).  Head count and head dimension
    use the separate knobs exposed by the reference repository.
    """

    model = model.to(device).eval()
    example_inputs = torch.randn(1, 3, model_config["img_size"], model_config["img_size"], device=device)
    before = collect_vit_structure(model)
    base_macs, base_params = count_ops_and_params(model, example_inputs)
    pruner, qkv_by_block = _build_pruner(
        model,
        example_inputs,
        pruning_ratio=isomorphic_pruning_ratio,
        head_pruning_ratio=isomorphic_head_pruning_ratio,
        head_dim_pruning_ratio=isomorphic_head_dim_pruning_ratio,
        round_to=round_to,
    )
    calibration = accumulate_group_taylor_gradients(
        model,
        dataset=calibration_dataset,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        split=calibration_split,
        seed=calibration_seed,
        transform=calibration_transform,
        num_workers=num_workers,
        data_root=data_root,
        device=device,
    )
    history_before = len(pruner.pruning_history())
    pruner.step()
    groups_pruned = len(pruner.pruning_history()) - history_before
    refresh_vit_attention_metadata(model, pruner.num_heads)
    model.zero_grad(set_to_none=True)
    # A post-pruning forward is deliberately performed before serialization: it
    # catches stale static timm metadata immediately.
    with torch.no_grad():
        logits = model(example_inputs)
    if logits.shape != (1, model_config["num_classes"]):
        raise RuntimeError(f"Pruned Isomorphic model returned invalid logits shape {tuple(logits.shape)}.")
    after = collect_vit_structure(model)
    pruned_macs, pruned_params = count_ops_and_params(model, example_inputs)
    structure_summary = build_structure_summary(before, after)

    # Keep the common artifact contract (model/source/model_config/pruning_stats)
    # so eval_pruned.py and TIMMPrunedLoRA can consume this artifact unchanged.
    artifact = build_pruning_artifact(
        model=model.cpu(),
        model_config=model_config,
        source_info=source_info,
        importance="isomorphic_taylor",
        calibration_config=calibration,
        pruning_modules=(),
        target_block_indices=None,
        pruning_ratio=None,
        mlp_pruning_ratio=None,
        head_pruning_ratio=None,
        iterative_steps=1,
        global_pruning=True,
        round_to=round_to,
        importance_group_reduction="mean",
        importance_normalizer="mean",
        activation_taylor_reduction=None,
        gate_taylor_reduction=None,
        gate_taylor_location=None,
        gate_taylor_aggregation=None,
        head_gate_taylor_reduction=None,
        head_gate_taylor_location=None,
        head_gate_taylor_aggregation=None,
        head_pruning_root=None,
        base_macs=base_macs,
        base_params=base_params,
        pruned_macs=pruned_macs,
        pruned_params=pruned_params,
        num_pruned_groups=groups_pruned,
        target_pruning_summary={},
    )
    artifact["pruning_config"]["isomorphic"] = {
        "enabled": True,
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_method": REFERENCE_METHOD,
        "engine": f"torch_pruning {getattr(tp, '__version__', 'unknown')}",
        "global_pruning": True,
        "pruning_ratio": float(isomorphic_pruning_ratio),
        "head_pruning_ratio": float(isomorphic_head_pruning_ratio),
        "head_dim_pruning_ratio": (
            None if isomorphic_head_dim_pruning_ratio is None else float(isomorphic_head_dim_pruning_ratio)
        ),
        "prune_head_dims": True,
        "prune_num_heads": True,
    }
    artifact["pruning_stats"]["isomorphic_structure"] = structure_summary
    artifact["pruning_stats"]["isomorphic_qkv_blocks"] = list(qkv_by_block)
    if output_path is None:
        output_path = os.path.join(output_dir, "pruned_timm_classifier.pth")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(artifact, output_path)

    print(f"[Isomorphic] source: {source_info}")
    print(f"[Isomorphic] calibration: {calibration}")
    print("[Isomorphic] ratios: " f"structure={isomorphic_pruning_ratio}, " f"heads={isomorphic_head_pruning_ratio}, " f"head_dim={isomorphic_head_dim_pruning_ratio}")
    print(f"[Isomorphic] groups pruned: {groups_pruned}")
    print(f"[Isomorphic] MACs: {base_macs:,} -> {pruned_macs:,}")
    print(f"[Isomorphic] Params: {base_params:,} -> {pruned_params:,}")
    if inspect_groups:
        print(f"[Isomorphic][Inspect] {structure_summary}")
    print(f"[Isomorphic] saved to: {output_path}")
    return artifact
