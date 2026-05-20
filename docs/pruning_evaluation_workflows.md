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
By default these configs use magnitude importance, which ranks channels from
weights only.

Configs:

```text
configs/timm_vit_pruning_cifar100.yaml
configs/timm_vit_pruning_flowers102.yaml
configs/timm_vit_pruning_cub200.yaml
```

Commands:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_pruning_cifar100.yaml --inspect-groups
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_pruning_flowers102.yaml --inspect-groups
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_pruning_cub200.yaml --inspect-groups
```

Taylor importance can be used when a supervised calibration set is available.
It runs a few calibration forward/backward passes before pruning, then ranks
channels from `weight * gradient`.

Taylor config:

```text
configs/timm_vit_taylor_pruning_cifar100.yaml
configs/timm_vit_taylor_pruning_flowers102.yaml
configs/timm_vit_taylor_pruning_cub200.yaml
```

Commands:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_taylor_pruning_cifar100.yaml --inspect-groups
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_taylor_pruning_flowers102.yaml --inspect-groups
CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_vit_taylor_pruning_cub200.yaml --inspect-groups
```

The calibration size is controlled by `calibration_batch_size` and
`calibration_batches`; for example, `64` and `10` uses up to 640 training
examples to populate gradients.

### 5.1 Taylor Layer Sensitivity

Layer sensitivity evaluates each transformer block independently. The script
computes Taylor calibration gradients once, then deep-copies the dense source
model for each `(layer, ratio)` trial and restores the same gradient snapshot
before pruning. This keeps every trial independent while avoiding repeated
calibration passes.

For the compact experiment summary, read the top docstring and `main()` in:

```text
sensitivity_taylor.py
```

Configs:

```text
configs/timm_vit_taylor_sensitivity_cifar100.yaml
configs/timm_vit_taylor_sensitivity_flowers102.yaml
configs/timm_vit_taylor_sensitivity_cub200.yaml
```

Quick check:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/sensitivity_taylor.sh \
  configs/timm_vit_taylor_sensitivity_cifar100.yaml \
  --target-layers 0 \
  --ratios 0.0,0.1 \
  --max-batches 2
```

Full sweep:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/sensitivity_taylor.sh \
  configs/timm_vit_taylor_sensitivity_cifar100.yaml
```

All downstream datasets:

```bash
GPU_ID=7 bash scripts/experiments/run_taylor_sensitivity.sh
```

Plot generated results:

```bash
python analysis/plot_taylor_sensitivity.py
```

By default this records `12 layers x 10 ratios` to `results.jsonl`. All ratios,
including `0.0`, go through the pruning pipeline; `0.0` is a no-op prune used as
a pipeline sanity check. Add `--save-artifacts` when the trial models themselves
should be kept.

### 6. LoRA Recovery After Fine-Tuned Pruning

Starts from a pruned artifact produced after LoRA fine-tuning and trains LoRA
recovery.

Configs:

```text
configs/timm_vit_pruned_lora_recovery_cifar100.yaml
configs/timm_vit_pruned_lora_recovery_flowers102.yaml
configs/timm_vit_pruned_lora_recovery_cub200.yaml
```

Commands:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_pruned_lora_recovery_cifar100.yaml
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_pruned_lora_recovery_flowers102.yaml
CUDA_VISIBLE_DEVICES=7 bash scripts/run.sh configs/timm_vit_pruned_lora_recovery_cub200.yaml
```
