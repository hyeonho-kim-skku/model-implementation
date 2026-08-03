# Joint Pruning and Isomorphic Pruning: Experiment Snapshot

Updated: 2026-07-22

This document records the current method comparison and its intended scope. Exact artifacts, checkpoints, and run logs listed below remain the source of truth.

## Research Scope

There are two distinct experiments and they should not be interpreted as one table.

1. **Target-only comparison on CIFAR-100**: compare Joint MLP+entire-head pruning with a faithful implementation of Isomorphic Pruning, starting from the same ViT-B/16 CIFAR-100 LoRA checkpoint and using the same target-data calibration and LoRA recovery protocol. “Target-only” means that the pruning and recovery stages do not access the pretraining dataset; it does not mean that the model has no pretrained initialization.
2. **Conventional ImageNet validation**: test Joint MLP+entire-head pruning on pretrained DeiT-S and recover it with full fine-tuning. This establishes that the proposed method also runs in a standard ImageNet setting, but it is not the Isomorphic comparison.

The current CIFAR-100 result supports a method-level claim under a fixed target-only, LoRA-recovery protocol. It does not by itself establish that Joint pruning is universally better than Isomorphic Pruning under the latter's native full-fine-tuning or distillation recipe.

## CIFAR-100 Comparison Protocol

Both pruning methods use:

- Model: ViT-B/16 classifier from the same LoRA-fine-tuned checkpoint
- Source checkpoint: `runs/timm_lora_cifar100_supervised/0511-224807/best_cls_ckpt.pth`
- Calibration data: all 50,000 CIFAR-100 training images
- Calibration loader: batch 64, 782 batches, bilinear/default transform, seed 42
- Recovery: 20 epochs of LoRA fine-tuning on CIFAR-100
- Optimizer: AdamW, learning rate `5e-4`, weight decay `0.05`, cosine schedule
- Recovery batch size: 64
- LoRA targets: `qkv`, `proj`, and `mlp`, rank 4
- Comparison budget: MACs matched within 2% of the Joint artifact

Only the pruning method differs:

| Method | Scored and pruned structures | Importance and ranking |
| --- | --- | --- |
| Joint | MLP hidden channels and entire attention heads | Gate Taylor, `sum_square + samplewise`; separate global ranking for MLP and heads; requested ratios 40% and 40% |
| Isomorphic | Graph-discovered embedding/residual width, MLP width, attention head dimension, and entire heads | Upstream-style GroupTaylor over isomorphic scopes; structure ratio 0.22, head ratio 0.22, head-dimension ratio 0.10 |

## CIFAR-100 Results

| Model | MACs | Parameters | Pruning-only Top-1 | 20-epoch LoRA best | Last Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense LoRA baseline | 17.585G | 85.88M | 92.11% | — | — |
| Joint MLP 40% + head 40% | 10.295G | 51.75M | 44.54% | **91.08%** | 91.01% |
| Isomorphic, MAC-matched | 10.457G | 50.75M | 20.87% | 88.58% | 88.45% |

The Isomorphic artifact has 1.58% more MACs than the Joint artifact, which is within the predefined ±2% tolerance. Despite having slightly more compute, it is 23.67 percentage points lower immediately after pruning and 2.50 points lower at the best recovered checkpoint. Relative to the dense baseline, the best Joint result loses 1.03 points and the best Isomorphic result loses 3.53 points.

These results do not imply that Isomorphic Pruning is intrinsically weaker. Plausible reasons for the difference in this protocol are:

- Joint pruning preserves the residual embedding width and within-head dimension, whereas Isomorphic pruning changes a broader set of coupled dimensions.
- The available 20-epoch LoRA recovery can adapt `qkv`, `proj`, and MLP modules but has less freedom than full fine-tuning to repair widespread structural changes.
- Isomorphic Pruning was designed and reported together with its own downstream training recipe; the target-only LoRA constraint is deliberately different.
- Joint's pruning targets are aligned closely with the modules adapted by the LoRA recovery stage.

Therefore, the defensible conclusion is: **at approximately matched MACs, Joint pruning is more recoverable than faithful Isomorphic pruning under this CIFAR-100 target-only LoRA protocol.** Generalization requires repetition on the other target datasets.

## Why Matched GroupTaylor Is Excluded

A “matched GroupTaylor” baseline would keep GroupTaylor importance but restrict its pruning structures to Joint's MLP and entire-head targets. That can be useful as an ablation to separate the effects of importance scoring and structural search space, but it is neither the original Isomorphic method nor the complete Joint method.

The present objective is to compare our method with an external method under the same data and recovery constraints. For that objective, modifying Isomorphic's structural scopes would weaken method fidelity and is not required. It should only be added later if the research question changes to a component-level causal analysis.

Full fine-tuning and distillation are also optional follow-ups rather than required controls for the current target-only LoRA claim. They become necessary only if the claim is broadened to recovery methods beyond LoRA or to Isomorphic Pruning's native training protocol.

## DeiT-S ImageNet Validation

This experiment evaluates Joint pruning independently in a conventional pretrained ImageNet-1K setting.

- Model: `deit_small_patch16_224.fb_in1k`
- Calibration: 6,400 ImageNet train images, batch 64 × 100, FP32, seed 42
- Calibration transform: resize 256 with bicubic interpolation, center crop 224, ImageNet normalization
- Pruning: separate global MLP and entire-head ranking, both requested at 40%
- Actual pruning: 40.148% of MLP hidden channels and 40.278% of heads
- Recovery: 50-epoch full fine-tuning, BF16, AdamW, global batch 512, two-GPU DDP, no distillation or EMA

| Stage | MACs | Parameters | ImageNet Top-1 |
| --- | ---: | ---: | ---: |
| Dense pretrained DeiT-S | 4.610G | 22.05M | 79.846% |
| Joint, pruning only | 2.729G | 13.50M | 14.042% |
| Joint, 50-epoch full fine-tuning | 2.729G | 13.50M | 77.662% |

The 50-epoch model recovers 63.620 points from the pruning-only result and finishes 2.184 points below the dense baseline. This confirms substantial recoverability, while the very low pruning-only accuracy also indicates that the present unconstrained global allocation is aggressive.

## Reproducibility Sources

### CIFAR-100 Joint

- Pruning config: `configs/timm_vit_cifar100_joint_mlp40_head40.yaml`
- Recovery config: `configs/timm_vit_cifar100_joint_mlp40_head40_lora_recovery.yaml`
- Artifact: `pruned/vit_base_cifar100_lora50_joint_mlp40_head40/pruned_timm_classifier.pth`
- Pruning-only metrics: `pruned/vit_base_cifar100_lora50_joint_mlp40_head40/eval_metrics.json`
- Recovery run: `runs/timm_pruned_lora_cifar100_supervised/0722-011114`

### CIFAR-100 Isomorphic

- Pruning config: `configs/timm_vit_cifar100_isomorphic_target_macs.yaml`
- Recovery config: `configs/timm_vit_cifar100_isomorphic_target_macs_lora_recovery.yaml`
- Artifact: `pruned/vit_base_cifar100_lora50_isomorphic_target_macs/pruned_timm_classifier.pth`
- Pruning-only metrics: `pruned/vit_base_cifar100_lora50_isomorphic_target_macs/eval_metrics.json`
- Recovery run: `runs/timm_pruned_lora_cifar100_supervised/0722-112831`

### ImageNet Joint

- Artifact: `pruned/timm_deit_small_imagenet_joint_mlp40_head40/pruned_timm_classifier.pth`
- Pruning-only metrics: `runs/timm_deit_small_imagenet_joint_mlp40_head40/pruning_only_eval_metrics.json`
- Fine-tuning summary: `runs/timm_deit_small_imagenet_joint_mlp40_head40/finetune_50e_ddp/summary.json`

Relevant implementation commits are `9deab26` for the CIFAR-100 Joint experiment configuration and `0700768` for the Isomorphic baseline.

## Next Experiment

The next high-value step is to repeat the same faithful, MAC-matched Joint-versus-Isomorphic protocol on another target dataset, without adding matched GroupTaylor. This tests whether the CIFAR-100 ordering generalizes while keeping the scientific question and recovery constraint unchanged.
