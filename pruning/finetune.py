"""Single-GPU and DDP full fine-tuning for structured-pruning artifacts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from timm.data import Mixup, create_transform, resolve_model_data_config

from datasets import build_timm_eval_transform, get_loader
from pruning.eval import load_pruned_artifact
from utils import build_seeded_generator, load_optimizer, load_scheduler, seed_worker, set_seed


KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class DistributedRuntime:
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_distributed(self):
        return self.world_size > 1

    @property
    def is_main_process(self):
        return self.rank == 0


def initialize_distributed_runtime():
    """Initialize NCCL only when launched through torchrun."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Pruned ImageNet fine-tuning requires a CUDA GPU.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return DistributedRuntime(
        device=torch.device("cuda", local_rank),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def destroy_distributed_runtime(runtime):
    if runtime.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def kst_now():
    return datetime.now(KST).isoformat(timespec="seconds")


def resolve_local_batch_size(global_batch_size, world_size):
    global_batch_size = int(global_batch_size)
    world_size = int(world_size)
    if global_batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch_size={global_batch_size} must be divisible by world_size={world_size}."
        )
    return global_batch_size // world_size


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _artifact_model(artifact_path, device):
    artifact = load_pruned_artifact(artifact_path, map_location="cpu")
    model = artifact["model"]
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.freeze_encoder = False
    return artifact, model.to(device)


def set_drop_path(model, drop_path):
    drop_path = float(drop_path)
    if not 0.0 <= drop_path <= 1.0:
        raise ValueError("drop_path must be in [0, 1].")
    for module in model.modules():
        if module.__class__.__name__ == "DropPath" and hasattr(module, "drop_prob"):
            module.drop_prob = drop_path


def build_finetune_train_transform(model):
    backbone = getattr(unwrap_model(model), "encoder", unwrap_model(model))
    data_config = resolve_model_data_config(backbone)
    transform = create_transform(
        input_size=data_config["input_size"],
        is_training=True,
        interpolation="bicubic",
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=data_config["mean"],
        std=data_config["std"],
    )
    return transform, dict(data_config)


def build_finetune_loaders(config, model, runtime):
    dataset = config.get("dataset", "imagenet")
    if dataset != "imagenet":
        raise ValueError("Pruned full fine-tuning currently supports dataset='imagenet' only.")
    global_batch_size = int(config["batch_size"])
    local_batch_size = resolve_local_batch_size(global_batch_size, runtime.world_size)
    num_workers = int(config.get("num_workers", 8))
    data_root = config["data_root"]
    seed = int(config.get("seed", 42))
    train_transform, train_data_config = build_finetune_train_transform(model)
    val_transform, val_data_config = build_timm_eval_transform(unwrap_model(model))
    loader_kwargs = {
        "distributed": runtime.is_distributed,
        "rank": runtime.rank,
        "world_size": runtime.world_size,
        "pin_memory": True,
        "persistent_workers": True,
    }
    train_loader = get_loader(
        dataset_name=dataset,
        batch_size=local_batch_size,
        mode="supervised",
        train=True,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        data_root=data_root,
        generator=build_seeded_generator(seed + runtime.rank),
        worker_init_fn=seed_worker,
        transform=train_transform,
        repeat_aug_reps=int(config.get("repeated_augmentation_reps", 3)),
        **loader_kwargs,
    )
    val_loader = get_loader(
        dataset_name=dataset,
        batch_size=local_batch_size,
        mode="test",
        train=False,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        data_root=data_root,
        transform=val_transform,
        **loader_kwargs,
    )
    return train_loader, val_loader, {
        "train": {
            "preset": "deit_style",
            "data_config": train_data_config,
            "interpolation": "bicubic",
            "auto_augment": "rand-m9-mstd0.5-inc1",
            "random_erasing": 0.25,
            "repeated_augmentation_reps": int(config.get("repeated_augmentation_reps", 3)),
        },
        "validation": {"preset": "timm_pretrained", "data_config": dict(val_data_config)},
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
        "world_size": runtime.world_size,
    }


def build_mixup(config):
    return Mixup(
        mixup_alpha=float(config.get("mixup", 0.8)),
        cutmix_alpha=float(config.get("cutmix", 1.0)),
        label_smoothing=float(config.get("label_smoothing", 0.1)),
        num_classes=int(config.get("num_classes", 1000)),
    )


def _reduce_totals(totals, runtime):
    values = torch.tensor(totals, dtype=torch.float64, device=runtime.device)
    if runtime.is_distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values.tolist()


def train_one_epoch(model, loader, optimizer, mixup_fn, runtime, max_batches=None):
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(runtime.device, non_blocking=True)
        labels = labels.to(runtime.device, non_blocking=True)
        images, labels = mixup_fn(images, labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * images.size(0)
        total_examples += images.size(0)
        total_batches += 1
    if total_batches == 0:
        raise ValueError("No training batches were processed.")
    total_loss, total_examples = _reduce_totals((total_loss, total_examples), runtime)
    # Every rank executes the same number of optimizer updates. Keep this as
    # the local count rather than summing ranks, so the metric means updates
    # per epoch in both single-GPU and DDP runs.
    return {"loss": total_loss / total_examples, "batches": total_batches}


@torch.no_grad()
def evaluate_finetune_classifier(model, loader, runtime, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(runtime.device, non_blocking=True)
        labels = labels.to(runtime.device, non_blocking=True)
        logits = model(images)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    if total == 0:
        raise ValueError("No evaluation batches were processed.")
    total_loss, correct, total = _reduce_totals((total_loss, correct, total), runtime)
    return {"loss": total_loss / total, "acc": 100.0 * correct / total}


def checkpoint_payload(*, artifact_path, model, optimizer, scheduler, epoch, best_top1, config, history, transform_metadata):
    return {
        "checkpoint_type": "pruned_full_finetune",
        "artifact_path": str(artifact_path),
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_top1": float(best_top1),
        "config": dict(config),
        "history": list(history),
        "transform_metadata": transform_metadata,
    }


def save_checkpoint(path, **kwargs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(**kwargs), path)


def load_resume_checkpoint(resume_path, device):
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != "pruned_full_finetune":
        raise ValueError("--resume must point to a pruned_full_finetune checkpoint.")
    artifact_path = checkpoint.get("artifact_path")
    if not artifact_path:
        raise ValueError("Resume checkpoint is missing artifact_path.")
    artifact, model = _artifact_model(artifact_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, artifact, model


def validate_resume_config(config, checkpoint):
    saved_epochs = checkpoint.get("config", {}).get("epochs")
    if saved_epochs is None:
        raise ValueError("Resume checkpoint is missing its original epochs setting.")
    if int(config["epochs"]) != int(saved_epochs):
        raise ValueError(
            "Cannot resume with a different epochs value because the saved cosine "
            f"schedule was created for epochs={saved_epochs}. Start a new run instead."
        )


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(value, file, indent=2)


def _prepare_output_dir(output_dir, resume_path, runtime):
    output_dir = Path(output_dir)
    error = 0
    if runtime.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not resume_path and any(
            path.exists() for path in (output_dir / "metrics.jsonl", output_dir / "latest.pth", output_dir / "best.pth")
        ):
            error = 1
    error = int(_reduce_totals((error,), runtime)[0])
    if error:
        raise FileExistsError(
            f"Output directory already contains a fine-tuning run: {output_dir}. "
            "Choose a new --output-dir or pass --resume."
        )
    if runtime.is_distributed:
        dist.barrier()
    return output_dir


def run_finetune(config, runtime, resume_path=None):
    config = dict(config)
    set_seed(int(config.get("seed", 42)) + runtime.rank)
    started_at_kst = kst_now()
    started_at_monotonic = time.perf_counter()
    output_dir = _prepare_output_dir(config["output_dir"], resume_path, runtime)

    if resume_path:
        resume, artifact, model = load_resume_checkpoint(resume_path, runtime.device)
        validate_resume_config(config, resume)
        artifact_path, start_epoch = resume["artifact_path"], int(resume["epoch"])
        best_top1, history = float(resume["best_top1"]), list(resume.get("history", []))
    else:
        artifact_path = config["artifact_path"]
        artifact, model = _artifact_model(artifact_path, runtime.device)
        start_epoch, best_top1, history = 0, float("-inf"), []

    set_drop_path(model, config.get("drop_path", 0.0))
    if runtime.is_distributed:
        model = DistributedDataParallel(model, device_ids=[runtime.local_rank], output_device=runtime.local_rank)
    train_loader, val_loader, transform_metadata = build_finetune_loaders(config, model, runtime)
    optimizer = load_optimizer("AdamW", model, float(config.get("learning_rate", 3e-4)), float(config.get("weight_decay", 0.05)))
    scheduler = load_scheduler("CosineAnnealingLR", optimizer, int(config["epochs"]), 0)
    if resume_path:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
    mixup_fn = build_mixup(config)
    max_train_batches = config.get("max_train_batches")
    max_val_batches = config.get("max_val_batches")
    epochs = int(config["epochs"])
    if start_epoch >= epochs:
        raise ValueError(f"Resume checkpoint starts at epoch {start_epoch}, but epochs={epochs}.")

    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(int(config.get("seed", 0)) + epoch)
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(model, train_loader, optimizer, mixup_fn, runtime, max_train_batches)
        val_metrics = evaluate_finetune_classifier(model, val_loader, runtime, max_val_batches)
        scheduler.step()
        if runtime.is_main_process:
            record = {
                "epoch": epoch + 1,
                "completed_at_kst": kst_now(),
                "learning_rate": epoch_learning_rate,
                "train_loss": train_metrics["loss"],
                "train_batches": train_metrics["batches"],
                "validation_loss": val_metrics["loss"],
                "top1": val_metrics["acc"],
            }
            history.append(record)
            with open(metrics_path, "a") as metrics_file:
                metrics_file.write(json.dumps(record) + "\n")
            payload_args = dict(
                artifact_path=artifact_path, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch + 1, best_top1=max(best_top1, val_metrics["acc"]), config=config,
                history=history, transform_metadata=transform_metadata,
            )
            save_checkpoint(output_dir / "latest.pth", **payload_args)
            if val_metrics["acc"] > best_top1:
                best_top1 = val_metrics["acc"]
                payload_args["best_top1"] = best_top1
                save_checkpoint(output_dir / "best.pth", **payload_args)
            print(f"[FineTune] epoch {epoch + 1}/{epochs} train_loss={record['train_loss']:.4f} val_loss={record['validation_loss']:.4f} top1={record['top1']:.3f}%")
        if runtime.is_distributed:
            dist.barrier()

    if not runtime.is_main_process:
        return None
    summary = {
        "artifact_path": str(artifact_path),
        "pruning_only_top1": float(config.get("pruning_only_top1", 14.042)),
        "best_top1": best_top1,
        "last": history[-1],
        "recovery_from_pruning_only": best_top1 - float(config.get("pruning_only_top1", 14.042)),
        "epochs_completed": len(history),
        "started_at_kst": started_at_kst,
        "completed_at_kst": kst_now(),
        "elapsed_seconds": time.perf_counter() - started_at_monotonic,
        "world_size": runtime.world_size,
        "global_batch_size": int(config["batch_size"]),
        "local_batch_size": resolve_local_batch_size(config["batch_size"], runtime.world_size),
        "transform_metadata": transform_metadata,
        "config": config,
        "pruning_config": artifact.get("pruning_config", {}),
        "pruning_stats": artifact.get("pruning_stats", {}),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
