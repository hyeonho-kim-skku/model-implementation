"""Full fine-tuning utilities for serialized structured-pruning artifacts."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.nn.functional as F
from timm.data import Mixup, create_transform, resolve_model_data_config

from datasets import build_timm_eval_transform, get_loader
from engine import evaluate_classifier
from pruning.eval import load_pruned_artifact
from utils import build_seeded_generator, load_optimizer, load_scheduler, seed_worker, set_seed


KST = ZoneInfo("Asia/Seoul")


def kst_now():
    return datetime.now(KST).isoformat(timespec="seconds")


def _artifact_model(artifact_path, device):
    artifact = load_pruned_artifact(artifact_path, map_location="cpu")
    model = artifact["model"]
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.freeze_encoder = False
    return artifact, model.to(device)


def set_drop_path(model, drop_path):
    """Set timm DropPath probabilities without changing the pruned structure."""

    drop_path = float(drop_path)
    if not 0.0 <= drop_path <= 1.0:
        raise ValueError("drop_path must be in [0, 1].")
    for module in model.modules():
        if module.__class__.__name__ == "DropPath" and hasattr(module, "drop_prob"):
            module.drop_prob = drop_path


def build_finetune_train_transform(model):
    """Build the fixed DeiT-style ImageNet fine-tuning preprocessing."""

    backbone = getattr(model, "encoder", model)
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


def build_finetune_loaders(config, model):
    dataset = config.get("dataset", "imagenet")
    if dataset != "imagenet":
        raise ValueError("Pruned full fine-tuning currently supports dataset='imagenet' only.")
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 8))
    data_root = config["data_root"]
    seed = config.get("seed", 42)
    train_transform, train_data_config = build_finetune_train_transform(model)
    val_transform, val_data_config = build_timm_eval_transform(model)
    train_loader = get_loader(
        dataset_name=dataset,
        batch_size=batch_size,
        mode="supervised",
        train=True,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        data_root=data_root,
        generator=build_seeded_generator(seed),
        worker_init_fn=seed_worker,
        transform=train_transform,
        repeat_aug_reps=int(config.get("repeated_augmentation_reps", 3)),
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = get_loader(
        dataset_name=dataset,
        batch_size=batch_size,
        mode="test",
        train=False,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        data_root=data_root,
        transform=val_transform,
        pin_memory=True,
        persistent_workers=True,
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
    }


def build_mixup(config):
    return Mixup(
        mixup_alpha=float(config.get("mixup", 0.8)),
        cutmix_alpha=float(config.get("cutmix", 1.0)),
        label_smoothing=float(config.get("label_smoothing", 0.1)),
        num_classes=int(config.get("num_classes", 1000)),
    )


def train_one_epoch(model, loader, optimizer, mixup_fn, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images, labels = mixup_fn(images, labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())
        total_batches += 1
    if total_batches == 0:
        raise ValueError("No training batches were processed.")
    return {"loss": total_loss / total_batches, "batches": total_batches}


def checkpoint_payload(*, artifact_path, model, optimizer, scheduler, epoch, best_top1, config, history, transform_metadata):
    return {
        "checkpoint_type": "pruned_full_finetune",
        "artifact_path": str(artifact_path),
        "model": model.state_dict(),
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
    """Prevent resuming a cosine schedule with a different total epoch count."""

    saved_config = checkpoint.get("config", {})
    saved_epochs = saved_config.get("epochs")
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


def run_finetune(config, device="cuda", resume_path=None):
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Pruned ImageNet fine-tuning requires a CUDA GPU.")
    config = dict(config)
    set_seed(config.get("seed", 42))
    started_at_kst = kst_now()
    started_at_monotonic = time.perf_counter()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_path:
        resume, artifact, model = load_resume_checkpoint(resume_path, device)
        validate_resume_config(config, resume)
        artifact_path = resume["artifact_path"]
        start_epoch = int(resume["epoch"])
        best_top1 = float(resume["best_top1"])
        history = list(resume.get("history", []))
    else:
        artifact_path = config["artifact_path"]
        artifact, model = _artifact_model(artifact_path, device)
        start_epoch = 0
        best_top1 = float("-inf")
        history = []

    set_drop_path(model, config.get("drop_path", 0.0))
    train_loader, val_loader, transform_metadata = build_finetune_loaders(config, model)
    optimizer = load_optimizer(
        "AdamW", model, float(config.get("learning_rate", 3e-4)),
        float(config.get("weight_decay", 0.05)),
    )
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
    if not resume_path and any(
        path.exists() for path in (metrics_path, output_dir / "latest.pth", output_dir / "best.pth")
    ):
        raise FileExistsError(
            f"Output directory already contains a fine-tuning run: {output_dir}. "
            "Choose a new --output-dir or pass --resume."
        )
    with open(metrics_path, "a") as metrics_file:
        for epoch in range(start_epoch, epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(int(config.get("seed", 0)) + epoch)
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, mixup_fn, device, max_train_batches,
            )
            val_metrics = evaluate_classifier(model, val_loader, device, max_batches=max_val_batches)
            scheduler.step()
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
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            payload_args = dict(
                artifact_path=artifact_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_top1=max(best_top1, val_metrics["acc"]),
                config=config,
                history=history,
                transform_metadata=transform_metadata,
            )
            save_checkpoint(output_dir / "latest.pth", **payload_args)
            if val_metrics["acc"] > best_top1:
                best_top1 = val_metrics["acc"]
                payload_args["best_top1"] = best_top1
                save_checkpoint(output_dir / "best.pth", **payload_args)
            print(
                f"[FineTune] epoch {epoch + 1}/{epochs} "
                f"train_loss={record['train_loss']:.4f} "
                f"val_loss={record['validation_loss']:.4f} top1={record['top1']:.3f}%"
            )

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
        "transform_metadata": transform_metadata,
        "config": config,
        "pruning_config": artifact.get("pruning_config", {}),
        "pruning_stats": artifact.get("pruning_stats", {}),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
