"""Intermediate recovery helpers for prune-recover progressive pruning."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_loader
from models.lora import FusedQKVLoRA, LoRAWrappedLinear
from models.timm_lora import count_parameters, inject_lora_into_vit
from utils import (
    build_seeded_generator,
    load_optimizer,
    load_scheduler,
    move_to_device,
    seed_worker,
    set_seed,
)


class IntermediateLoRARecoveryModel(nn.Module):
    """Temporarily attach trainable LoRA adapters to a plain pruned model."""

    def __init__(
        self,
        model,
        rank,
        alpha,
        lora_modules,
        qkv_lora_components,
    ):
        super().__init__()
        self.model = model
        self.model.freeze_encoder = True
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

        self.injected_module_names = inject_lora_into_vit(
            self.model.encoder,
            rank=rank,
            alpha=alpha,
            qkv_lora_components=qkv_lora_components,
            lora_modules=lora_modules,
        )
        if not self.injected_module_names:
            raise ValueError("Intermediate recovery did not inject any LoRA modules.")

    def forward(self, images):
        return self.model(images)

    def merge_to_plain_model(self):
        """Merge LoRA updates and return the underlying plain classifier model."""

        merged_encoder = copy.deepcopy(self.model.encoder)
        for block in merged_encoder.blocks:
            if isinstance(block.attn.qkv, FusedQKVLoRA):
                block.attn.qkv = block.attn.qkv.to_merged_linear()
            if isinstance(block.attn.proj, LoRAWrappedLinear):
                block.attn.proj = block.attn.proj.to_merged_linear()
            if isinstance(block.mlp.fc1, LoRAWrappedLinear):
                block.mlp.fc1 = block.mlp.fc1.to_merged_linear()
            if isinstance(block.mlp.fc2, LoRAWrappedLinear):
                block.mlp.fc2 = block.mlp.fc2.to_merged_linear()

        self.model.encoder = merged_encoder
        self.model.freeze_encoder = False
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = True
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True
        self.model.zero_grad(set_to_none=True)
        return self.model


def _recovery_value(config, name, default=None):
    return config.get(f"intermediate_recovery_{name}", default)


def _make_train_loader(config, seed):
    dataset = _recovery_value(config, "dataset") or config.get("dataset")
    if not dataset:
        raise ValueError("dataset is required for intermediate recovery.")
    batch_size = int(
        _recovery_value(config, "batch_size")
        or config.get("batch_size")
        or 64
    )
    return get_loader(
        dataset,
        batch_size,
        mode=_recovery_value(config, "mode", "supervised"),
        train=True,
        shuffle=True,
        drop_last=True,
        num_workers=config.get("num_workers", 4),
        data_root=config.get("data_root", "./data"),
        generator=build_seeded_generator(seed),
        worker_init_fn=seed_worker if seed is not None else None,
    )


def recover_model_with_lora(model, config, device, step_index):
    """Run short CE recovery and return a merged plain model plus metadata."""

    epochs = int(_recovery_value(config, "epochs", 1))
    if epochs <= 0:
        raise ValueError("intermediate_recovery_epochs must be positive.")

    base_seed = _recovery_value(config, "seed")
    if base_seed is None:
        base_seed = config.get("seed", 42)
    stage_seed = None if base_seed is None else int(base_seed) + int(step_index) - 1
    set_seed(stage_seed)

    rank = int(_recovery_value(config, "lora_rank", 4))
    alpha = _recovery_value(config, "lora_alpha")
    lora_modules = _recovery_value(config, "lora_modules", "qkv,proj,mlp")
    qkv_components = _recovery_value(
        config,
        "qkv_lora_components",
        "q,k,v",
    )
    recovery_model = IntermediateLoRARecoveryModel(
        model=model,
        rank=rank,
        alpha=alpha,
        lora_modules=lora_modules,
        qkv_lora_components=qkv_components,
    ).to(device)
    trainable_params, total_params = count_parameters(recovery_model)

    optimizer_name = _recovery_value(config, "optimizer", "AdamW")
    lr = float(_recovery_value(config, "lr", 5e-4))
    weight_decay = float(_recovery_value(config, "weight_decay", 0.05))
    momentum = float(_recovery_value(config, "momentum", 0.9))
    nesterov = bool(_recovery_value(config, "nesterov", False))
    classifier_lr = _recovery_value(config, "classifier_lr")
    if classifier_lr is not None:
        classifier_lr = float(classifier_lr)

    optimizer = load_optimizer(
        optimizer_name,
        recovery_model,
        lr,
        weight_decay,
        momentum,
        nesterov,
        classifier_lr=classifier_lr,
    )
    scheduler_name = _recovery_value(config, "scheduler", "CosineAnnealingLR")
    scheduler = load_scheduler(
        scheduler_name,
        optimizer,
        epochs,
        int(_recovery_value(config, "warmup_epochs", 0)),
    )
    train_loader = _make_train_loader(config, stage_seed)
    max_batches = _recovery_value(config, "batches")
    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches <= 0:
            raise ValueError("intermediate_recovery_batches must be positive or null.")

    epoch_losses = []
    epoch_batch_counts = []
    recovery_model.train()
    for _ in range(epochs):
        total_loss = 0.0
        processed_batches = 0
        for batch_index, batch in enumerate(train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(recovery_model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            processed_batches += 1
        if processed_batches == 0:
            raise ValueError("Intermediate recovery did not process any batches.")
        epoch_losses.append(total_loss / processed_batches)
        epoch_batch_counts.append(processed_batches)
        scheduler.step()

    merged_model = recovery_model.merge_to_plain_model().to(device)
    merged_model.eval()
    metadata = {
        "epochs": epochs,
        "epoch_losses": epoch_losses,
        "epoch_batch_counts": epoch_batch_counts,
        "final_train_loss": epoch_losses[-1],
        "seed": stage_seed,
        "optimizer": optimizer_name,
        "lr": lr,
        "classifier_lr": classifier_lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "nesterov": nesterov,
        "scheduler": scheduler_name,
        "warmup_epochs": int(_recovery_value(config, "warmup_epochs", 0)),
        "batch_size": int(
            _recovery_value(config, "batch_size")
            or config.get("batch_size")
            or 64
        ),
        "requested_batches_per_epoch": (
            max_batches if max_batches is not None else "full"
        ),
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_modules": lora_modules,
        "qkv_lora_components": qkv_components,
        "reset_classifier": False,
        "injected_module_names": list(recovery_model.injected_module_names),
        "trainable_params": trainable_params,
        "total_params_during_recovery": total_params,
    }
    return merged_model, metadata
