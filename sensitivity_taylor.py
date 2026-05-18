"""Layer-wise Taylor pruning sensitivity entrypoint.

Default experiment for ViT-Base:
  layers 0..11 x ratios 0.0,0.1,...,0.9 = 120 trials.
  Each trial prunes only one transformer block's MLP hidden width.

Pipeline:
  load dense checkpoint -> eval reference baseline -> compute Taylor gradients
  once -> snapshot parameter.grad -> for each (layer, ratio), deepcopy the
  unpruned model, restore grads, call prune_model(target_block_indices=[layer]),
  eval, append one JSONL row.

ratio=0.0 is a no-op prune, but it still exercises the pruning pipeline. Every
trial starts from the same unpruned model and same calibration gradients; never
prune cumulatively across layers or ratios.
"""

import argparse
import copy
import json
import os

import torch
import yaml

from datasets import get_loader
from engine import evaluate_classifier
from pruning.source import build_pruning_source
from pruning.structured import compute_taylor_gradients, prune_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_RATIOS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"


def parse_float_list(value):
    """Accept YAML lists or comma strings such as '0.0,0.1,0.2'."""

    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_int_list(value):
    """Return None for an empty layer list so callers can choose all layers."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    value = str(value).strip()
    return None if not value else [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser():
    """Define CLI flags shared by YAML configs and command-line overrides."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a YAML config file")
    parser.add_argument("--source-type", dest="source_type", type=str, choices=["checkpoint", "timm"], help="Model source for pruning")
    parser.add_argument("--checkpoint-path", dest="checkpoint_path", type=str, help="Path to a checkpoint source")
    parser.add_argument("--backbone-name", dest="backbone_name", type=str, help="timm backbone name for source_type=timm")
    parser.add_argument("--num-classes", dest="num_classes", type=int, help="Number of classes for source_type=timm")
    parser.add_argument("--img-size", dest="img_size", type=int, help="Input image size for source_type=timm")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Load pretrained timm weights for source_type=timm")

    parser.add_argument("--dataset", type=str, help="Evaluation dataset name")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--split", type=str, default="test", help="Evaluation split: train or test")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--max-batches", dest="max_batches", type=int, default=None, help="Optional quick eval batch limit")
    parser.add_argument("--data-root", dest="data_root", type=str, default="./data", help="Dataset root directory")

    parser.add_argument("--results-path", dest="results_path", type=str, default="./pruned/taylor_layer_sensitivity/results.jsonl", help="JSONL path for sensitivity results")
    parser.add_argument("--artifact-dir", dest="artifact_dir", type=str, default="./pruned/taylor_layer_sensitivity/artifacts", help="Directory for optional artifacts")
    parser.add_argument("--save-artifacts", dest="save_artifacts", action=argparse.BooleanOptionalAction, default=False, help="Save each pruned trial artifact")
    parser.add_argument("--ratios", type=str, default=DEFAULT_RATIOS, help="Comma-separated pruning ratios")
    parser.add_argument("--target-layers", dest="target_layers", type=str, default=None, help="Comma-separated block indices; default uses all blocks")

    parser.add_argument("--pruning-modules", dest="pruning_modules", type=str, default="mlp", help="Comma-separated pruning targets: qkv,mlp")
    parser.add_argument("--global-pruning", dest="global_pruning", action=argparse.BooleanOptionalAction, default=False, help="Use global pruning across target modules")
    parser.add_argument("--round-to", dest="round_to", type=int, default=None, help="Round pruned dimensions to a multiple")
    parser.add_argument("--calibration-dataset", dest="calibration_dataset", type=str, default=None, help="Dataset used to compute Taylor gradients")
    parser.add_argument("--calibration-batch-size", dest="calibration_batch_size", type=int, default=64, help="Batch size for Taylor calibration")
    parser.add_argument("--calibration-batches", dest="calibration_batches", type=int, default=10, help="Number of Taylor calibration batches")
    parser.add_argument("--calibration-split", dest="calibration_split", type=str, choices=["train", "test"], default="train", help="Dataset split for Taylor calibration")
    parser.add_argument("--inspect-groups", dest="inspect_groups", action="store_true", help="Print selected Torch-Pruning dependency groups")
    parser.add_argument("--max-inspect-groups", dest="max_inspect_groups", type=int, default=3, help="Maximum inspected groups per trial")
    return parser


def make_eval_loader(args):
    """Build the evaluation loader used for baseline and all pruned trials."""

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


def num_blocks(model):
    """Return the number of ViT transformer blocks exposed by timm."""

    if not hasattr(model.encoder, "blocks"):
        raise ValueError("This model does not expose encoder.blocks.")
    return len(model.encoder.blocks)


def capture_gradients(model):
    """Deep-copy does not preserve parameter.grad, so keep grads explicitly."""

    # Dict comprehension: collect only parameters that received calibration
    # gradients, keyed by their stable named_parameters() name.
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def restore_gradients(model, gradients):
    """Attach the saved Taylor gradients to a copied trial model."""

    for name, parameter in model.named_parameters():
        grad = gradients.get(name)
        # Missing gradients stay None; existing gradients are copied to the
        # trial model's current device so TaylorImportance can read them.
        parameter.grad = None if grad is None else grad.to(parameter.device).clone()


def artifact_path(args, layer_idx, ratio):
    """Deterministic artifact path for optional --save-artifacts output."""

    ratio_tag = f"ratio{int(round(ratio * 100)):03d}"
    return os.path.join(args.artifact_dir, f"layer_{layer_idx:02d}", ratio_tag, "pruned_timm_classifier.pth")


def write_jsonl(path, row):
    """Append one JSON object per line so partial long runs remain readable."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as file:
        file.write(json.dumps(row) + "\n")


def reset_results(path, args, ratios, layers, calibration_config, baseline_metrics):
    """Start a fresh results file with one metadata row."""

    metadata = {
        "type": "metadata",
        "config": {
            "source_type": args.source_type,
            "checkpoint_path": args.checkpoint_path,
            "dataset": args.dataset,
            "split": args.split,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "pruning_modules": args.pruning_modules,
            "ratios": ratios,
            "target_layers": layers,
            "calibration": calibration_config,
            "reference_baseline_metrics": baseline_metrics,
            "save_artifacts": args.save_artifacts,
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as file:
        file.write(json.dumps(metadata) + "\n")


def make_result_row(source, args, layer_idx, ratio, metrics, artifact=None, path=None):
    """Create the JSONL row for either a pruned artifact or a fallback row."""

    pruning_config = {
        "importance": "taylor",
        "pruning_modules": args.pruning_modules,
        "target_block_indices": [layer_idx],
        "pruning_ratio": ratio,
        "global_pruning": args.global_pruning,
        "round_to": args.round_to,
    }
    pruning_stats = {
        "base_macs": None,
        "pruned_macs": None,
        "base_params": None,
        "pruned_params": None,
        "num_pruned_groups": 0,
    }
    if artifact is not None:
        # Prefer the actual pruning metadata from prune_model(); fallback values
        # are only used if a caller records metrics without a pruning artifact.
        pruning_config = artifact.get("pruning_config", pruning_config)
        pruning_stats = artifact.get("pruning_stats", pruning_stats)

    return {
        "type": "trial",
        "layer_idx": layer_idx,
        "ratio": ratio,
        "metrics": metrics,
        "artifact_path": path,
        "model_config": source.model_config,
        "source": source.source_info,
        "pruning_config": pruning_config,
        "pruning_stats": pruning_stats,
    }


def validate_sweep(model, ratios, layers):
    """Fail early for impossible ratios or layer indices."""

    invalid_ratios = [ratio for ratio in ratios if ratio < 0.0 or ratio >= 1.0]
    if invalid_ratios:
        raise ValueError(f"Ratios must be in [0.0, 1.0): {invalid_ratios}")

    block_count = num_blocks(model)
    invalid_layers = [idx for idx in layers if idx < 0 or idx >= block_count]
    if invalid_layers:
        raise ValueError(f"target_layers contains out-of-range indices: {invalid_layers}")


def run_pruned_trial(args, source, base_model, gradients, layer_idx, ratio, eval_loader):
    """Run one independent (layer, ratio) pruning and evaluation trial."""

    trial_model = copy.deepcopy(base_model)
    restore_gradients(trial_model, gradients)

    path = artifact_path(args, layer_idx, ratio)
    artifact = prune_model(
        model=trial_model,
        model_config=source.model_config,
        source_info=source.source_info,
        output_dir=os.path.dirname(path),
        output_path=path,
        importance="taylor",
        pruning_ratio=ratio,
        pruning_modules=args.pruning_modules,
        target_block_indices=[layer_idx],
        iterative_steps=1,
        global_pruning=args.global_pruning,
        round_to=args.round_to,
        inspect_groups=args.inspect_groups,
        max_inspect_groups=args.max_inspect_groups,
        use_existing_taylor_gradients=True,
        existing_calibration_config=args.calibration_config,
        save_artifact=args.save_artifacts,
        verbose=False,
        device=DEVICE,
    )
    metrics = evaluate_classifier(artifact["model"].to(DEVICE), eval_loader, DEVICE, args.max_batches)
    return make_result_row(
        source=source,
        args=args,
        layer_idx=layer_idx,
        ratio=ratio,
        metrics=metrics,
        artifact=artifact,
        path=path if args.save_artifacts else None,
    )


def main(args):
    """Load config, prepare shared calibration state, and run the full sweep."""

    if args.dataset is None:
        raise ValueError("--dataset is required for evaluation.")
    if args.calibration_dataset is None:
        raise ValueError("--calibration-dataset is required for Taylor calibration.")

    source = build_pruning_source(vars(args), device=DEVICE)
    base_model = source.model.to(DEVICE)
    ratios = parse_float_list(args.ratios)
    # Empty target_layers means "all transformer blocks".
    layers = parse_int_list(args.target_layers) or list(range(num_blocks(base_model)))
    validate_sweep(base_model, ratios, layers)

    print(f"[Sensitivity] device={DEVICE}, blocks={num_blocks(base_model)}")
    print(f"[Sensitivity] layers={layers}")
    print(f"[Sensitivity] ratios={ratios}")

    eval_loader = make_eval_loader(args)
    baseline_metrics = evaluate_classifier(base_model, eval_loader, DEVICE, args.max_batches)
    print(f"[Sensitivity] reference baseline acc={baseline_metrics['acc']:.2f}%")

    # Taylor calibration is the expensive part. Run it once on the unpruned
    # model, then restore the same gradient snapshot for every independent trial.
    args.calibration_config = compute_taylor_gradients(
        model=base_model,
        calibration_dataset=args.calibration_dataset,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        calibration_split=args.calibration_split,
        num_workers=args.num_workers,
        data_root=args.data_root,
        device=DEVICE,
    )
    gradients = capture_gradients(base_model)
    if not gradients:
        raise ValueError("Taylor calibration completed, but no parameter gradients were found.")
    print(f"[Sensitivity] calibration={args.calibration_config}")
    print(f"[Sensitivity] gradient tensors={len(gradients)}")

    reset_results(args.results_path, args, ratios, layers, args.calibration_config, baseline_metrics)

    total = len(layers) * len(ratios)
    # The inner generator yields (layer, ratio) pairs without building a
    # separate list. enumerate(..., start=1) only changes the trial counter.
    for trial_idx, (layer_idx, ratio) in enumerate(
        ((layer_idx, ratio) for layer_idx in layers for ratio in ratios),
        start=1,
    ):
        print(f"[Sensitivity] trial {trial_idx}/{total}: layer={layer_idx}, ratio={ratio:.2f}")
        row = run_pruned_trial(args, source, base_model, gradients, layer_idx, ratio, eval_loader)
        row.update({"trial_index": trial_idx, "total_trials": total})
        write_jsonl(args.results_path, row)
        print(f"[Sensitivity] acc={row['metrics']['acc']:.2f}%")

    print(f"[Sensitivity] done: {args.results_path}")


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as file:
            parser.set_defaults(**yaml.safe_load(file))
    main(parser.parse_args())
