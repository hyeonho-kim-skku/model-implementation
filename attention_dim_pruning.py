"""Attention-dim pruning sweep using ragged fused-QKV attention."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from datasets import get_loader
from engine import evaluate_classifier
from models.ragged_attention import RaggedFusedQKVAttention
from pruning.attention_dim_cache import (
    load_attention_dim_score_cache,
    save_attention_dim_score_cache,
    validate_attention_dim_score_cache,
)
from pruning.attention_dim_importance import AttentionDimGateTaylorCollector
from pruning.attention_dim_pruning import (
    capture_attention_dim_metadata,
    compute_attention_dim_mask_equivalence,
    ensure_ragged_attention,
    prune_selected_attention_dims,
    select_attention_dims_by_score,
    validate_equal_head_width_metadata,
)
from pruning.source import build_pruning_source
from pruning.structured import compute_taylor_gradients


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_RATIOS = "0.02,0.04,0.06,0.08,0.10"


def build_parser():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add("--config", type=str)
    add("--source-type", dest="source_type", choices=["checkpoint", "timm"])
    add("--checkpoint-path", dest="checkpoint_path")
    add("--backbone-name", dest="backbone_name")
    add("--num-classes", dest="num_classes", type=int)
    add("--img-size", dest="img_size", type=int)
    add("--pretrained", action=argparse.BooleanOptionalAction, default=True)

    add("--dataset", type=str, required=False)
    add("--batch-size", dest="batch_size", type=int, default=64)
    add("--split", type=str, default="test")
    add("--num-workers", dest="num_workers", type=int, default=4)
    add("--max-batches", dest="max_batches", type=int, default=None)
    add("--data-root", dest="data_root", type=str, default="./data")

    add("--ratios", default=DEFAULT_RATIOS)
    add("--attention-dim-target", dest="attention_dim_target", choices=["v_proj", "qk_pair", "qkv_shared"], default="v_proj")
    add("--attention-dim-constraint", dest="attention_dim_constraint", choices=["free", "equal_head_width"], default="free")
    add("--target-block-indices", dest="target_block_indices", default=None)
    add("--global-pruning", dest="global_pruning", action=argparse.BooleanOptionalAction, default=True)
    add("--min-qk-dim-per-head", dest="min_qk_dim_per_head", type=int, default=1)
    add("--min-v-dim-per-head", dest="min_v_dim_per_head", type=int, default=1)

    add("--attention-dim-reduction", dest="attention_dim_reduction", choices=["signed_damage", "sum_abs", "sum_square"], default="sum_abs")
    add("--attention-dim-aggregation", dest="attention_dim_aggregation", choices=["elementwise", "samplewise"], default="samplewise")
    add("--attention-dim-gate-location", dest="attention_dim_gate_location", choices=["proj_in", "qk_pair", "qkv_shared"], default=None)
    add("--calibration-dataset", dest="calibration_dataset", type=str, default=None)
    add("--calibration-batch-size", dest="calibration_batch_size", type=int, default=64)
    add("--calibration-batches", dest="calibration_batches", default=None)
    add("--calibration-split", dest="calibration_split", choices=["train", "test"], default="train")
    add("--calibration-seed", dest="calibration_seed", type=int, default=42)

    add("--score-cache-path", dest="score_cache_path", type=str, default=None)
    add("--force-recompute-cache", dest="force_recompute_cache", action="store_true")
    add("--results-path", dest="results_path", type=str, default=None)
    add("--artifact-dir", dest="artifact_dir", type=str, default=None)
    add("--save-artifacts", dest="save_artifacts", action=argparse.BooleanOptionalAction, default=True)
    add("--skip-equivalence-check", dest="skip_equivalence_check", action="store_true")
    add("--check-mask-equivalence", dest="check_mask_equivalence", action="store_true")
    return parser


def parse_ratios(value):
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_calibration_batches(value):
    if value is None or str(value).strip().lower() in {"", "none", "null", "full", "all"}:
        return None
    return int(value)


def parse_target_block_indices(value):
    if value is None or str(value).strip().lower() in {"", "none", "null", "all"}:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def normalize_args(args):
    if args.dataset is None:
        raise ValueError("--dataset is required.")
    if args.calibration_dataset is None:
        args.calibration_dataset = args.dataset
    args.ratios = parse_ratios(args.ratios)
    args.calibration_batches = parse_calibration_batches(args.calibration_batches)
    args.target_block_indices = parse_target_block_indices(args.target_block_indices)
    if (args.results_path is None) != (args.artifact_dir is None):
        raise ValueError("--results-path and --artifact-dir must be provided together.")
    dataset = str(args.dataset).replace("-", "_")
    gate_location = args.attention_dim_gate_location or default_attention_dim_gate_location(
        args.attention_dim_target
    )
    args.attention_dim_gate_location = gate_location
    args.experiment_prefix = (
        f"vit_base_{dataset}_lora50_attention_dim_gate_taylor_"
        f"{args.attention_dim_target}_{gate_location}_"
        f"{args.attention_dim_reduction}_{args.attention_dim_aggregation}"
    )
    if args.attention_dim_constraint != "free":
        args.experiment_prefix = (
            f"{args.experiment_prefix}_{args.attention_dim_constraint}"
        )
    args.score_cache_path = args.score_cache_path or (
        f"./pruned/cache/{args.experiment_prefix}_full_scores.pth"
    )
    return args


def default_attention_dim_gate_location(target):
    if target == "qk_pair":
        return "qk_pair"
    if target == "qkv_shared":
        return "qkv_shared"
    return "proj_in"


def make_eval_loader(args):
    return get_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        mode="test",
        train=(args.split == "train"),
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )


def output_paths_for_ratio(args, ratio):
    if args.results_path is not None and args.artifact_dir is not None:
        return args.results_path, args.artifact_dir
    tag = ratio_tag(ratio)
    return (
        f"./pruned/{args.experiment_prefix}_global{tag}/results.jsonl",
        f"./pruned/{args.experiment_prefix}_global{tag}/artifacts",
    )


def ratio_tag(ratio):
    return f"{int(round(float(ratio) * 100)):03d}"


def check_attention_equivalence(model, *, device):
    max_diff = 0.0
    for block in model.encoder.blocks:
        old_attn = copy.deepcopy(block.attn).to(device).eval()
        ragged_attn = RaggedFusedQKVAttention.from_timm_attention(
            copy.deepcopy(block.attn)
        ).to(device).eval()
        dim = old_attn.qkv.in_features
        x = torch.randn(2, 17, dim, device=device)
        with torch.no_grad():
            old_output = old_attn(x)
            ragged_output = ragged_attn(x)
        max_diff = max(max_diff, float((old_output - ragged_output).abs().max().item()))
    return max_diff


def infer_image_size(args, model):
    if args.img_size is not None:
        return int(args.img_size)
    model_img_size = getattr(model, "img_size", None)
    if model_img_size is not None:
        return _first_image_size_value(model_img_size)
    patch_embed = getattr(getattr(model, "encoder", None), "patch_embed", None)
    patch_img_size = getattr(patch_embed, "img_size", None)
    if patch_img_size is not None:
        return _first_image_size_value(patch_img_size)
    return 224


def _first_image_size_value(value):
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def run_mask_equivalence_checks(args, model, *, device):
    block_idx = args.target_block_indices[0] if args.target_block_indices else 0
    selected_dims = {int(block_idx): [{"head_idx": 0, "dim_idx": 0}]}
    image_size = infer_image_size(args, model)
    dtype = next(model.parameters()).dtype
    example_inputs = torch.randn(2, 3, image_size, image_size, device=device, dtype=dtype)
    results = {}
    for target in ("v_proj", "qk_pair", "qkv_shared"):
        result = compute_attention_dim_mask_equivalence(
            model,
            example_inputs,
            selected_dims,
            target=target,
        )
        results[target] = result
        print(
            "[AttentionDim] mask equivalence "
            f"{target}: max_diff={result['max_diff']:.6g}, "
            f"mean_diff={result['mean_diff']:.6g}"
        )
    return results


def cache_metadata(args, source, calibration_config, scores):
    return {
        "dataset": args.calibration_dataset,
        "checkpoint_path": args.checkpoint_path,
        "source": source.source_info,
        "model_config": source.model_config,
        "importance": "attention_dim_gate_taylor",
        "attention_dim_target": args.attention_dim_target,
        "attention_dim_reduction": args.attention_dim_reduction,
        "attention_dim_aggregation": args.attention_dim_aggregation,
        "attention_dim_gate_location": args.attention_dim_gate_location,
        "attention_dim_score_mode": "attention_dim_gate_grad",
        "calibration_split": args.calibration_split,
        "calibration_batches": args.calibration_batches if args.calibration_batches is not None else "full",
        "calibration_seed": args.calibration_seed,
        "loss_reduction": "sum",
        "num_blocks": len(source.model.encoder.blocks),
        "target_block_indices": args.target_block_indices,
        "score_shapes": {int(idx): list(score.shape) for idx, score in scores.items()},
        "calibration_config": calibration_config,
    }


def get_attention_dim_scores(args, source):
    cache_path = Path(args.score_cache_path)
    if cache_path.exists() and not args.force_recompute_cache:
        scores, metadata = load_attention_dim_score_cache(cache_path)
        validate_attention_dim_score_cache(
            source.model,
            scores,
            metadata,
            dataset=args.calibration_dataset,
            checkpoint_path=args.checkpoint_path,
            attention_dim_target=args.attention_dim_target,
            attention_dim_reduction=args.attention_dim_reduction,
            attention_dim_aggregation=args.attention_dim_aggregation,
            attention_dim_gate_location=args.attention_dim_gate_location,
            calibration_split=args.calibration_split,
            calibration_batches=args.calibration_batches,
            calibration_seed=args.calibration_seed,
            target_block_indices=args.target_block_indices,
        )
        print(f"[AttentionDim] loaded score cache: {cache_path}")
        return scores, metadata, True

    collector = AttentionDimGateTaylorCollector(
        source.model,
        target=args.attention_dim_target,
        target_block_indices=args.target_block_indices,
        reduction=args.attention_dim_reduction,
        gate_location=args.attention_dim_gate_location,
        aggregation=args.attention_dim_aggregation,
    )
    try:
        calibration_config = compute_taylor_gradients(
            model=source.model,
            calibration_dataset=args.calibration_dataset,
            calibration_batch_size=args.calibration_batch_size,
            calibration_batches=args.calibration_batches,
            calibration_split=args.calibration_split,
            num_workers=args.num_workers,
            data_root=args.data_root,
            device=DEVICE,
            calibration_seed=args.calibration_seed,
            head_gate_taylor_collector=collector,
        )
        scores = collector.final_scores()
    finally:
        collector.remove()

    metadata = cache_metadata(args, source, calibration_config, scores)
    save_attention_dim_score_cache(cache_path, scores, metadata)
    validate_attention_dim_score_cache(
        source.model,
        scores,
        metadata,
        dataset=args.calibration_dataset,
        checkpoint_path=args.checkpoint_path,
        attention_dim_target=args.attention_dim_target,
        attention_dim_reduction=args.attention_dim_reduction,
        attention_dim_aggregation=args.attention_dim_aggregation,
        attention_dim_gate_location=args.attention_dim_gate_location,
        calibration_split=args.calibration_split,
        calibration_batches=args.calibration_batches,
        calibration_seed=args.calibration_seed,
        target_block_indices=args.target_block_indices,
    )
    print(f"[AttentionDim] saved score cache: {cache_path}")
    return scores, metadata, False


def count_params(model):
    return int(sum(param.numel() for param in model.parameters()))


def run_trial(args, source, base_model, scores, cache_info, ratio, eval_loader):
    trial_model = copy.deepcopy(base_model)
    before_metadata = capture_attention_dim_metadata(trial_model, args.target_block_indices)
    selected_dims = select_attention_dims_by_score(
        scores,
        trial_model,
        target=args.attention_dim_target,
        attention_dim_constraint=args.attention_dim_constraint,
        pruning_ratio=ratio,
        global_pruning=args.global_pruning,
        min_qk_dim_per_head=args.min_qk_dim_per_head,
        min_v_dim_per_head=args.min_v_dim_per_head,
        target_block_indices=args.target_block_indices,
    )
    pruning_metadata = prune_selected_attention_dims(
        trial_model,
        selected_dims,
        target=args.attention_dim_target,
        target_block_indices=args.target_block_indices,
    )
    after_metadata = capture_attention_dim_metadata(trial_model, args.target_block_indices)
    if args.attention_dim_constraint == "equal_head_width":
        validate_equal_head_width_metadata(
            after_metadata,
            target=args.attention_dim_target,
        )
    metrics = evaluate_classifier(trial_model.to(DEVICE), eval_loader, DEVICE, args.max_batches)

    results_path, artifact_dir = output_paths_for_ratio(args, ratio)
    artifact_path = os.path.join(artifact_dir, ratio_tag(ratio), "pruned_timm_classifier.pth")
    artifact = {
        "model": trial_model.cpu(),
        "source": source.source_info,
        "model_config": source.model_config,
        "pruning_config": {
            "importance": "attention_dim_gate_taylor",
            "pruning_modules": ["attention_dim"],
            "attention_dim_target": args.attention_dim_target,
            "attention_dim_constraint": args.attention_dim_constraint,
            "attention_dim_reduction": args.attention_dim_reduction,
            "attention_dim_aggregation": args.attention_dim_aggregation,
            "attention_dim_gate_location": args.attention_dim_gate_location,
            "pruning_ratio": ratio,
            "global_pruning": args.global_pruning,
            "target_block_indices": args.target_block_indices,
            "min_qk_dim_per_head": args.min_qk_dim_per_head,
            "min_v_dim_per_head": args.min_v_dim_per_head,
            "calibration": dict(cache_info.get("calibration_config", {})),
        },
        "pruning_stats": {
            "base_params": count_params(base_model),
            "pruned_params": count_params(trial_model),
            "attention_dim_metadata_before": before_metadata,
            "attention_dim_metadata_after": after_metadata,
            "direct_attention_dim_pruning_metadata": pruning_metadata,
        },
    }
    if args.save_artifacts:
        os.makedirs(os.path.dirname(artifact_path) or ".", exist_ok=True)
        torch.save(artifact, artifact_path)

    return results_path, {
        "type": "trial",
        "ratio": ratio,
        "metrics": metrics,
        "artifact_path": artifact_path if args.save_artifacts else None,
        "model_config": source.model_config,
        "source": source.source_info,
        "pruning_config": artifact["pruning_config"],
        "pruning_stats": artifact["pruning_stats"],
    }


def write_jsonl(path, row, mode="a"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode) as file:
        file.write(json.dumps(row) + "\n")


def metadata_row(
    args,
    baseline_metrics,
    cache_info,
    cache_loaded,
    ratio,
    equivalence_max_diff,
    mask_equivalence=None,
):
    return {
        "type": "metadata",
        "config": {
            "source_type": args.source_type,
            "checkpoint_path": args.checkpoint_path,
            "dataset": args.dataset,
            "split": args.split,
            "ratios": args.ratios,
            "current_ratio": ratio,
            "importance": "attention_dim_gate_taylor",
            "attention_dim_target": args.attention_dim_target,
            "attention_dim_constraint": args.attention_dim_constraint,
            "attention_dim_reduction": args.attention_dim_reduction,
            "attention_dim_aggregation": args.attention_dim_aggregation,
            "attention_dim_gate_location": args.attention_dim_gate_location,
            "global_pruning": args.global_pruning,
            "score_cache_path": args.score_cache_path,
            "score_cache_loaded": cache_loaded,
            "reference_baseline_metrics": baseline_metrics,
            "ragged_attention_equivalence_max_diff": equivalence_max_diff,
            "mask_equivalence": mask_equivalence,
            "calibration": cache_info.get("calibration_config", {}),
            "save_artifacts": args.save_artifacts,
        },
    }


def main(args):
    args = normalize_args(args)
    source = build_pruning_source(vars(args), device=DEVICE)
    source = replace(source, model=source.model.to(DEVICE))
    source.model.eval()

    equivalence_max_diff = None
    if not args.skip_equivalence_check:
        equivalence_max_diff = check_attention_equivalence(source.model, device=DEVICE)
        print(f"[AttentionDim] ragged attention equivalence max diff={equivalence_max_diff:.6g}")
    converted = ensure_ragged_attention(source.model)
    print(f"[AttentionDim] converted attention blocks: {converted}")
    mask_equivalence = None
    if args.check_mask_equivalence:
        mask_equivalence = run_mask_equivalence_checks(args, source.model, device=DEVICE)

    eval_loader = make_eval_loader(args)
    baseline = evaluate_classifier(source.model, eval_loader, DEVICE, args.max_batches)
    print(f"[AttentionDim] reference baseline acc={baseline['acc']:.2f}%")

    scores, cache_info, cache_loaded = get_attention_dim_scores(args, source)
    source.model.zero_grad(set_to_none=True)
    base_model = source.model.cpu()

    for idx, ratio in enumerate(args.ratios, start=1):
        print(f"[AttentionDim] trial {idx}/{len(args.ratios)} ratio={ratio:.2f}")
        results_path, row = run_trial(args, source, base_model, scores, cache_info, ratio, eval_loader)
        write_jsonl(
            results_path,
            metadata_row(
                args,
                baseline,
                cache_info,
                cache_loaded,
                ratio,
                equivalence_max_diff,
                mask_equivalence=mask_equivalence,
            ),
            mode="w",
        )
        write_jsonl(results_path, row)
        print(f"[AttentionDim] acc={row['metrics']['acc']:.2f}%")


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as file:
            parser.set_defaults(**yaml.safe_load(file))
    main(parser.parse_args())
