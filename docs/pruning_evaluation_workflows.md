# Pruning Evaluation Workflows

This note summarizes the standard evaluation workflows for pretrained timm ViT
models, structured pruning, and LoRA recovery.

## Commands

Run from the repository root:

```bash
cd /home/hyeonho/projects/model_implementation
```

Training:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> bash scripts/run.sh <CONFIG_PATH>
```

Pruning:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> bash scripts/prune.sh <CONFIG_PATH> --inspect-groups
```

## Outputs

Training runs save checkpoints under:

```text
./runs/{model}_{dataset}_{method}/{timestamp}/best_cls_ckpt.pth
```

Pruning runs save artifacts under:

```text
./pruned/<experiment_name>/pruned_timm_classifier.pth
```

## Workflows

### 1. Frozen Backbone Linear Probe

Evaluates the pretrained backbone as a frozen feature extractor.

Configs:

```text
configs/timm_vit_linear_probe_cifar100.yaml
configs/timm_vit_linear_probe_flowers102.yaml
configs/timm_vit_linear_probe_cub200.yaml
```

Command:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_linear_probe_<dataset>.yaml
```

### 2. Direct-Pruned Backbone Linear Probe

Prunes the pretrained backbone first, then trains a fresh classifier on the
frozen pruned encoder.

Prune configs:

```text
configs/timm_vit_prune_from_pretrained_cifar100.yaml
configs/timm_vit_prune_from_pretrained_flowers102.yaml
configs/timm_vit_prune_from_pretrained_cub200.yaml
```

Probe configs:

```text
configs/timm_vit_pruned_linear_probe_cifar100.yaml
configs/timm_vit_pruned_linear_probe_flowers102.yaml
configs/timm_vit_pruned_linear_probe_cub200.yaml
```

Commands:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_prune_from_pretrained_<dataset>.yaml --inspect-groups
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_pruned_linear_probe_<dataset>.yaml
```

### 3. Direct-Pruned Backbone LoRA Recovery

Starts from the direct-pruned artifact, resets the classifier, and trains LoRA
adapters plus the new classifier.

Configs:

```text
configs/timm_vit_pruned_lora_recovery_reset_cls_cifar100.yaml
configs/timm_vit_pruned_lora_recovery_reset_cls_flowers102.yaml
configs/timm_vit_pruned_lora_recovery_reset_cls_cub200.yaml
```

Command:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_pruned_lora_recovery_reset_cls_<dataset>.yaml
```

### 4. LoRA Fine-Tuning Baseline

Fine-tunes LoRA adapters on the unpruned pretrained backbone.

Configs:

```text
configs/timm_vit_lora_cifar100.yaml
configs/timm_vit_lora_flowers102.yaml
configs/timm_vit_lora_cub200.yaml
```

Command:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_lora_<dataset>.yaml
```

### 5. Pruning After LoRA Fine-Tuning

Prunes the dense `merged_model` saved by a LoRA fine-tuning checkpoint.

Configs:

```text
configs/timm_vit_pruning_cifar100.yaml
configs/timm_vit_pruning.yaml          # flowers102
configs/timm_vit_pruning_cub200.yaml
```

Commands:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_pruning_cifar100.yaml --inspect-groups
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_pruning.yaml --inspect-groups
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_pruning_cub200.yaml --inspect-groups
```

### 6. LoRA Recovery After Fine-Tuned Pruning

Starts from a pruned artifact produced after LoRA fine-tuning and trains LoRA
recovery.

Configs:

```text
configs/timm_vit_pruned_lora_recovery_cifar100.yaml
configs/timm_vit_pruned_lora_recovery.yaml     # flowers102
configs/timm_vit_pruned_lora_recovery_cub200.yaml
```

Commands:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_pruned_lora_recovery_cifar100.yaml
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_pruned_lora_recovery.yaml
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_pruned_lora_recovery_cub200.yaml
```