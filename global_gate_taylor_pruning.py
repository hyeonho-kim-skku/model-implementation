"""Global MLP pruning sweep using cached gate Taylor channel scores."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import yaml

from datasets import get_loader
from engine import evaluate_classifier
from pruning.gate_taylor_cache import (
    capture_mlp_taylor_scores,
    load_gate_taylor_score_cache,
    restore_mlp_taylor_scores,
    save_gate_taylor_score_cache,
    validate_gate_taylor_score_cache,
)
from pruning.importance import MLPGateTaylorCollector
from pruning.source import build_pruning_source
from pruning.structured import compute_taylor_gradients, prune_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_RATIOS = "0.4,0.5,0.6"


def parse_ratios(value):
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_calibration_batches(value):
    if value is None or str(value).strip().lower() in {"", "none", "null", "full", "all"}:
        return None
    return int(value)


def build_parser():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add("--config", type=str, help="Path to a YAML config file")
    add("--source-type", dest="source_type", choices=["checkpoint", "timm"])
    add("--checkpoint-path", dest="checkpoint_path")
    add("--backbone-name", dest="backbone_name")
    add("--num-classes", dest="num_classes", type=int)
    add("--img-size", dest="img_size", type=int)
    add("--pretrained", action=argparse.BooleanOptionalAction, default=True)

    add("--dataset", type=str, help="Evaluation dataset")
    add("--batch-size", dest="batch_size", type=int, default=64)
    add("--split", type=str, default="test")
    add("--num-workers", dest="num_workers", type=int, default=4)
    add("--max-batches", dest="max_batches", type=int, default=None)
    add("--data-root", dest="data_root", type=str, default="./data")

    add("--ratios", type=str, default=DEFAULT_RATIOS)
    add("--pruning-modules", dest="pruning_modules", type=str, default="mlp")
    add("--round-to", dest="round_to", type=int, default=None)
    add("--inspect-groups", dest="inspect_groups", action="store_true")

    add("--gate-taylor-location", dest="gate_taylor_location", type=str, default="fc2_in")
    add(
        "--gate-taylor-reduction",
        dest="gate_taylor_reduction",
        choices=["signed_damage", "sum_abs", "sum_square"],
        default="sum_square",
    )
    add(
        "--gate-taylor-aggregation",
        dest="gate_taylor_aggregation",
        choices=["elementwise", "samplewise"],
        default="elementwise",
    )
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
    return parser


def normalize_args(args):
    if args.dataset is None:
        raise ValueError("--dataset is required.")
    if args.calibration_dataset is None:
        raise ValueError("--calibration-dataset is required.")

    args.calibration_batches = parse_calibration_batches(args.calibration_batches)
    args.ratios = parse_ratios(args.ratios)

    pruning_modules = tuple(
        item.strip().lower() for item in str(args.pruning_modules).split(",") if item.strip()
    )
    if pruning_modules != ("mlp",):
        raise ValueError("Global gate Taylor pruning currently supports --pruning-modules mlp only.")
    args.pruning_modules = "mlp"

    bad_ratios = [ratio for ratio in args.ratios if ratio < 0.0 or ratio >= 1.0]
    if bad_ratios:
        raise ValueError(f"Ratios must be in [0.0, 1.0): {bad_ratios}")

    if (args.results_path is None) != (args.artifact_dir is None):
        raise ValueError("--results-path and --artifact-dir must be provided together.")
    if len(args.ratios) > 1 and (args.results_path is not None or args.artifact_dir is not None):
        raise ValueError(
            "Multiple ratios write one output directory per ratio. "
            "Omit --results-path/--artifact-dir, or run a single ratio."
        )

    dataset = str(args.dataset).replace("-", "_")
    aggregation_suffix = (
        "" if args.gate_taylor_aggregation == "elementwise" else f"_{args.gate_taylor_aggregation}"
    )
    args.experiment_prefix = (
        f"vit_base_{dataset}_lora50_gate_taylor_"
        f"{args.gate_taylor_location}_{args.gate_taylor_reduction}"
        f"{aggregation_suffix}"
    )
    if len(args.ratios) == 1:
        args.results_path, args.artifact_dir = output_paths_for_ratio(args, args.ratios[0])
    args.score_cache_path = args.score_cache_path or (
        f"./pruned/cache/vit_base_{dataset}_lora50_gate_taylor_"
        f"{args.gate_taylor_location}_{args.gate_taylor_reduction}"
        f"{aggregation_suffix}_full_scores.pth"
    )
    return args


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


def cache_metadata(args, source, calibration_config, scores):
    return {
        "dataset": args.calibration_dataset,
        "checkpoint_path": args.checkpoint_path,
        "source": source.source_info,
        "model_config": source.model_config,
        "importance": "gate_taylor",
        "gate_taylor_location": args.gate_taylor_location,
        "gate_taylor_reduction": args.gate_taylor_reduction,
        "gate_taylor_aggregation": args.gate_taylor_aggregation,
        "gate_taylor_score_mode": "elementwise_gate_grad",
        "calibration_split": args.calibration_split,
        "calibration_batches": args.calibration_batches if args.calibration_batches is not None else "full",
        "calibration_seed": args.calibration_seed,
        "loss_reduction": "sum",
        "num_blocks": len(source.model.encoder.blocks),
        "score_shapes": {int(idx): list(score.shape) for idx, score in scores.items()},
        "calibration_config": calibration_config,
    }


def validate_cache(args, source, scores, metadata):
    validate_gate_taylor_score_cache(
        source.model,
        scores,
        metadata,
        dataset=args.calibration_dataset,
        checkpoint_path=args.checkpoint_path,
        gate_taylor_location=args.gate_taylor_location,
        gate_taylor_reduction=args.gate_taylor_reduction,
        gate_taylor_aggregation=args.gate_taylor_aggregation,
        calibration_split=args.calibration_split,
        calibration_batches=args.calibration_batches,
        calibration_seed=args.calibration_seed,
    )


def get_gate_taylor_scores(args, source):
    """Load cached scores, or compute them once and save them."""

    cache_path = Path(args.score_cache_path)
    if cache_path.exists() and not args.force_recompute_cache:
        scores, metadata = load_gate_taylor_score_cache(cache_path)
        validate_cache(args, source, scores, metadata)
        print(f"[GlobalGateTaylor] loaded score cache: {cache_path}")
        return scores, metadata, True

    collector = MLPGateTaylorCollector(
        model=source.model,
        reduction=args.gate_taylor_reduction,
        gate_location=args.gate_taylor_location,
        aggregation=args.gate_taylor_aggregation,
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
            gate_taylor_collector=collector,
        )
        scores = capture_mlp_taylor_scores(source.model, collector.final_scores())
    finally:
        collector.remove()

    if not scores:
        raise ValueError("Gate Taylor calibration completed, but no MLP scores were found.")

    metadata = cache_metadata(args, source, calibration_config, scores)
    save_gate_taylor_score_cache(cache_path, scores, metadata)
    validate_cache(args, source, scores, metadata)
    print(f"[GlobalGateTaylor] saved score cache: {cache_path}")
    return scores, metadata, False


def write_jsonl(path, row, mode="a"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode) as file:
        file.write(json.dumps(row) + "\n")


def ratio_tag(ratio):
    return f"ratio{int(round(ratio * 100)):03d}"


def output_paths_for_ratio(args, ratio):
    if args.results_path is not None and args.artifact_dir is not None:
        return args.results_path, args.artifact_dir
    experiment = f"{args.experiment_prefix}_global{int(round(ratio * 100)):03d}"
    return f"./pruned/{experiment}/results.jsonl", f"./pruned/{experiment}/artifacts"


def run_pruning_trial(args, source, base_model, scores, cache_info, ratio, eval_loader):
    trial_model = copy.deepcopy(base_model)
    results_path, artifact_dir = output_paths_for_ratio(args, ratio)
    artifact_path = os.path.join(artifact_dir, ratio_tag(ratio), "pruned_timm_classifier.pth")
    calibration = dict(cache_info.get("calibration_config", {}))
    calibration.update({"score_cache_path": args.score_cache_path, "score_cache_loaded": True})

    artifact = prune_model(
        model=trial_model,
        model_config=source.model_config,
        source_info=source.source_info,
        output_dir=os.path.dirname(artifact_path),
        output_path=artifact_path,
        importance="gate_taylor",
        pruning_ratio=ratio,
        pruning_modules="mlp",
        target_block_indices=None,
        iterative_steps=1,
        global_pruning=True,
        round_to=args.round_to,
        gate_taylor_reduction=args.gate_taylor_reduction,
        gate_taylor_location=args.gate_taylor_location,
        gate_taylor_aggregation=args.gate_taylor_aggregation,
        inspect_groups=args.inspect_groups,
        use_existing_taylor_gradients=True,
        existing_calibration_config=calibration,
        existing_gate_taylor_scores=restore_mlp_taylor_scores(trial_model, scores),
        save_artifact=args.save_artifacts,
        verbose=False,
        device=DEVICE,
    )
    metrics = evaluate_classifier(artifact["model"].to(DEVICE), eval_loader, DEVICE, args.max_batches)
    return results_path, {
        "type": "trial",
        "ratio": ratio,
        "metrics": metrics,
        "artifact_path": artifact_path if args.save_artifacts else None,
        "model_config": source.model_config,
        "source": source.source_info,
        "pruning_config": artifact.get("pruning_config", {}),
        "pruning_stats": artifact.get("pruning_stats", {}),
    }


def metadata_row(args, baseline_metrics, cache_info, cache_loaded, ratio):
    return {
        "type": "metadata",
        "config": {
            "source_type": args.source_type,
            "checkpoint_path": args.checkpoint_path,
            "dataset": args.dataset,
            "split": args.split,
            "ratios": args.ratios,
            "current_ratio": ratio,
            "importance": "gate_taylor",
            "pruning_modules": "mlp",
            "global_pruning": True,
            "gate_taylor_location": args.gate_taylor_location,
            "gate_taylor_reduction": args.gate_taylor_reduction,
            "gate_taylor_aggregation": args.gate_taylor_aggregation,
            "score_cache_path": args.score_cache_path,
            "score_cache_loaded": cache_loaded,
            "calibration": cache_info.get("calibration_config", {}),
            "reference_baseline_metrics": baseline_metrics,
            "save_artifacts": args.save_artifacts,
        },
    }


def main(args):
    args = normalize_args(args)
    source = build_pruning_source(vars(args), device=DEVICE)
    base_model = source.model.to(DEVICE)

    print(f"[GlobalGateTaylor] device={DEVICE}")
    print(f"[GlobalGateTaylor] ratios={args.ratios}")
    print(f"[GlobalGateTaylor] score_cache_path={args.score_cache_path}")

    eval_loader = make_eval_loader(args)
    baseline = evaluate_classifier(base_model, eval_loader, DEVICE, args.max_batches)
    print(f"[GlobalGateTaylor] reference baseline acc={baseline['acc']:.2f}%")

    scores, cache_info, cache_loaded = get_gate_taylor_scores(args, source)
    base_model.zero_grad(set_to_none=True)

    for idx, ratio in enumerate(args.ratios, start=1):
        print(f"[GlobalGateTaylor] trial {idx}/{len(args.ratios)}: ratio={ratio:.2f}")
        results_path, row = run_pruning_trial(
            args, source, base_model, scores, cache_info, ratio, eval_loader
        )
        row.update({"trial_index": idx, "total_trials": len(args.ratios)})
        write_jsonl(
            results_path,
            metadata_row(args, baseline, cache_info, cache_loaded, ratio),
            mode="w",
        )
        write_jsonl(results_path, row)
        print(f"[GlobalGateTaylor] acc={row['metrics']['acc']:.2f}%")

    print("[GlobalGateTaylor] done")


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as file:
            parser.set_defaults(**yaml.safe_load(file))
    main(parser.parse_args())
