"""Shared progressive pruning loop.

This module owns experiment orchestration only. It reuses the existing pruning
core for scoring collectors, dependency-aware pruning, and artifact packaging.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import torch

from engine import evaluate_classifier
from datasets import get_loader
from pruning.gate_taylor_cache import capture_mlp_taylor_scores, restore_mlp_taylor_scores
from pruning.importance import MLPGateTaylorCollector
from pruning.structured import compute_taylor_gradients, prune_model


def parse_target_ratios(value):
    """Parse comma-separated or list-style cumulative pruning targets."""

    if isinstance(value, (list, tuple)):
        ratios = [float(item) for item in value]
    else:
        ratios = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not ratios:
        raise ValueError("target_ratios must not be empty.")
    previous = 0.0
    for ratio in ratios:
        if ratio <= previous or ratio >= 1.0:
            raise ValueError(
                "target_ratios must be strictly increasing values in (0, 1); "
                f"got {ratios}."
            )
        previous = ratio
    return ratios


def cumulative_to_step_ratio(previous_target, target_ratio):
    """Convert an original-model cumulative target to current-model pruning ratio."""

    previous_target = float(previous_target)
    target_ratio = float(target_ratio)
    if previous_target < 0.0 or previous_target >= 1.0:
        raise ValueError(f"previous_target must be in [0, 1), got {previous_target}.")
    if target_ratio <= previous_target or target_ratio >= 1.0:
        raise ValueError(
            "target_ratio must be greater than previous_target and less than 1; "
            f"got previous={previous_target}, target={target_ratio}."
        )
    return (target_ratio - previous_target) / (1.0 - previous_target)


def ratio_tag(ratio):
    return f"target{int(round(float(ratio) * 100)):03d}"


def write_jsonl(path, row, mode="a"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode) as file:
        file.write(json.dumps(row) + "\n")


def make_eval_loader(config):
    dataset = config.get("eval_dataset") or config.get("dataset")
    if dataset is None:
        return None
    batch_size = config.get("eval_batch_size") or config.get("batch_size") or 64
    split = config.get("eval_split", "test")
    return get_loader(
        dataset,
        batch_size,
        mode="test",
        train=(split == "train"),
        shuffle=False,
        drop_last=False,
        num_workers=config.get("num_workers", 4),
        data_root=config.get("data_root", "./data"),
    )


def compute_gate_taylor_scores(model, objective, config, device):
    """Compute one current-model gate Taylor score snapshot."""

    if getattr(objective, "calibration_objective", None) != "ce":
        raise NotImplementedError(
            "Only the CE baseline objective is implemented for scoring."
        )

    # TODO(progressive_pruning): route the Prototype/SupCon objective through an
    # objective.loss callback instead of structured.py's built-in CE branch.
    collector = MLPGateTaylorCollector(
        model=model,
        reduction=config.get("gate_taylor_reduction", "sum_square"),
        gate_location=config.get("gate_taylor_location", "fc2_in"),
        aggregation=config.get("gate_taylor_aggregation", "elementwise"),
    )
    try:
        calibration_config = compute_taylor_gradients(
            model=model,
            calibration_dataset=config.get("calibration_dataset") or config.get("dataset"),
            calibration_batch_size=config.get("calibration_batch_size")
            or config.get("batch_size", 64),
            calibration_batches=config.get("calibration_batches", None),
            calibration_split=config.get("calibration_split", "train"),
            num_workers=config.get("num_workers", 4),
            data_root=config.get("data_root", "./data"),
            device=device,
            calibration_seed=config.get("calibration_seed"),
            gate_taylor_collector=collector,
            calibration_objective=objective.calibration_objective,
        )
        scores = capture_mlp_taylor_scores(model, collector.final_scores())
    finally:
        collector.remove()

    if not scores:
        raise ValueError("Gate Taylor scoring completed, but no MLP scores were captured.")
    return scores, calibration_config


def artifact_path_for(output_dir, target_ratio):
    return str(Path(output_dir) / ratio_tag(target_ratio) / "pruned_timm_classifier.pth")


def total_mlp_hidden(model):
    if not hasattr(model.encoder, "blocks"):
        raise ValueError("Progressive pruning needs model.encoder.blocks.")
    return int(sum(block.mlp.fc1.out_features for block in model.encoder.blocks))


def run_progressive_pruning(source, objective, config, device):
    """Run score -> prune -> rescore progressive pruning without mid-step training."""

    # Recovery is intentionally excluded here; it should run later over the
    # saved artifacts.
    target_ratios = parse_target_ratios(config.get("target_ratios"))
    output_dir = config.get("output_dir")
    if output_dir is None:
        raise ValueError("output_dir is required.")
    results_path = config.get("results_path") or str(Path(output_dir) / "results.jsonl")
    save_artifacts = bool(config.get("save_artifacts", True))
    eval_loader = make_eval_loader(config)

    current_model = source.model.to(device)
    current_model.eval()
    original_mlp_hidden = total_mlp_hidden(current_model)
    previous_target = 0.0
    rows = []

    metadata = {
        "type": "metadata",
        "source": source.source_info,
        "model_config": source.model_config,
        "objective": objective.metadata(),
        "target_ratios": target_ratios,
        "config": {
            "pruning_modules": config.get("pruning_modules", "mlp"),
            "global_pruning": config.get("global_pruning", True),
            "gate_taylor_location": config.get("gate_taylor_location", "fc2_in"),
            "gate_taylor_reduction": config.get("gate_taylor_reduction", "sum_square"),
            "gate_taylor_aggregation": config.get("gate_taylor_aggregation", "elementwise"),
            "calibration_dataset": config.get("calibration_dataset") or config.get("dataset"),
            "calibration_split": config.get("calibration_split", "train"),
            "calibration_batches": config.get("calibration_batches", None),
            "save_artifacts": save_artifacts,
        },
    }
    write_jsonl(results_path, metadata, mode="w")

    for step_index, target_ratio in enumerate(target_ratios, start=1):
        step_ratio = cumulative_to_step_ratio(previous_target, target_ratio)
        scores, calibration_config = compute_gate_taylor_scores(
            current_model,
            objective,
            config,
            device,
        )

        output_path = artifact_path_for(output_dir, target_ratio)
        calibration = dict(calibration_config)
        calibration.update(
            {
                "progressive_step_index": step_index,
                "progressive_previous_target_ratio": previous_target,
                "progressive_target_ratio": target_ratio,
                "progressive_step_ratio": step_ratio,
            }
        )
        source_info = deepcopy(source.source_info)
        source_info["progressive_pruning"] = {
            "step_index": step_index,
            "previous_target_ratio": previous_target,
            "target_ratio": target_ratio,
            "step_ratio": step_ratio,
        }

        artifact = prune_model(
            model=current_model,
            model_config=source.model_config,
            source_info=source_info,
            output_dir=os.path.dirname(output_path),
            output_path=output_path,
            importance="gate_taylor",
            pruning_ratio=step_ratio,
            pruning_modules=config.get("pruning_modules", "mlp"),
            target_block_indices=config.get("target_block_indices"),
            iterative_steps=1,
            global_pruning=config.get("global_pruning", True),
            round_to=config.get("round_to"),
            gate_taylor_reduction=config.get("gate_taylor_reduction", "sum_square"),
            gate_taylor_location=config.get("gate_taylor_location", "fc2_in"),
            gate_taylor_aggregation=config.get("gate_taylor_aggregation", "elementwise"),
            inspect_groups=config.get("inspect_groups", False),
            use_existing_taylor_gradients=True,
            existing_calibration_config=calibration,
            existing_gate_taylor_scores=restore_mlp_taylor_scores(current_model, scores),
            save_artifact=save_artifacts,
            verbose=config.get("verbose", True),
            device=device,
        )
        current_mlp_hidden = total_mlp_hidden(artifact["model"])
        cumulative_pruned_hidden = original_mlp_hidden - current_mlp_hidden
        cumulative_stats = {
            "original_mlp_hidden": original_mlp_hidden,
            "current_mlp_hidden": current_mlp_hidden,
            "cumulative_pruned_hidden": cumulative_pruned_hidden,
            "cumulative_pruned_ratio": cumulative_pruned_hidden / original_mlp_hidden,
            "target_ratio": target_ratio,
        }
        artifact["progressive_pruning"] = {
            "step_index": step_index,
            "previous_target_ratio": previous_target,
            "target_ratio": target_ratio,
            "step_ratio": step_ratio,
            "cumulative_stats": cumulative_stats,
        }
        artifact.setdefault("pruning_stats", {})[
            "progressive_cumulative_stats"
        ] = cumulative_stats
        if save_artifacts:
            torch.save(artifact, output_path)

        metrics = None
        if eval_loader is not None and config.get("eval_each_step", True):
            metrics = evaluate_classifier(
                artifact["model"].to(device),
                eval_loader,
                device,
                max_batches=config.get("max_batches"),
            )

        row = {
            "type": "trial",
            "step_index": step_index,
            "target_ratio": target_ratio,
            "step_ratio": step_ratio,
            "artifact_path": output_path if save_artifacts else None,
            "metrics": metrics,
            "progressive_cumulative_stats": cumulative_stats,
            "pruning_config": artifact.get("pruning_config", {}),
            "pruning_stats": artifact.get("pruning_stats", {}),
        }
        write_jsonl(results_path, row)
        rows.append(row)

        current_model = artifact["model"].to(device)
        current_model.eval()
        previous_target = target_ratio

    return rows
