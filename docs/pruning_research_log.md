# Pruning Research Log

This is a working research notebook for the ViT-Base LoRA MLP pruning study. It is meant to help future chats and future me understand the reasoning, not just the final numbers. Use CSVs, `results.jsonl`, `command.txt`, and `args.json` as the exact source of truth.

The completed Joint-versus-Isomorphic CIFAR-100 comparison and the separate DeiT-S/ImageNet validation are summarized in [`joint_isomorphic_comparison.md`](joint_isomorphic_comparison.md). That snapshot also defines the scope of the claims and explains why matched GroupTaylor is not part of the required comparison.

## Current Research Question

Can gate Taylor importance identify MLP pruning structures that are recoverable for ViT-Base LoRA fine-tuned models?

The current hypothesis is:

- Gate Taylor scores can identify useful MLP hidden-channel pruning candidates.
- Magnitude-based aggregation is more stable than signed aggregation.
- High-ratio global pruning may be acceptable if followed by LoRA recovery.
- If unconstrained global pruning over-concentrates pruning in a few layers, constrained or rebalanced global pruning may improve stability.

## Current Takeaway

- The current best working setting is `gate_taylor + fc2_in + sum_square`.
- Global pruning at 60% causes large pruning-only degradation.
- LoRA recovery restores 50% and 60% pruned models close to dense baselines.
- Global pruning concentrates heavily in late MLP blocks, especially blocks 9-10.
- This allocation is broadly aligned with layer-wise sensitivity, but not perfectly. Block 11 often appears more tolerant than its global allocation suggests.
- Recent aggregation ablations show that the axis used to aggregate element-wise gate Taylor contributions matters. In short: `samplewise` helps moderate pruning, `tokenwise` is more useful on fine-grained datasets than CIFAR100, and `channelwise` is cancellation-prone.
- Recovery follow-up shows aggregation effects mostly persist after LoRA recovery, but margins shrink. Pruning-only accuracy is useful but not a perfect proxy for recovered accuracy.
- The next research question is to explain aggregation behavior through cancellation-ratio and mask-overlap analyses. Constrained/rebalanced global pruning remains a later candidate.

## Terms Used In This Log

- **MLP hidden channel**: the intermediate MLP dimension between `fc1` and `fc2`. Pruning one hidden channel removes the corresponding `fc1` output channel and dependent `fc2` input channel.
- **`fc1_out` gate**: gate after `mlp.fc1` and before GELU.
- **`fc2_in` gate**: gate after GELU and before `mlp.fc2`.
- **Layer-wise sensitivity**: prune one MLP block at a time and measure accuracy drop.
- **Global pruning allocation**: how much of the global pruning budget is assigned to each MLP block by channel-level scores.
- **Pruning-only accuracy**: accuracy immediately after pruning without recovery.
- **Recovery accuracy**: accuracy after LoRA recovery fine-tuning on the pruned model.

## Experimental Setup

- Workspace: `/home/hyeonho/projects/model_implementation`
- Model family: ViT-Base LoRA fine-tuned classifiers.
- Datasets: CIFAR100, CUB200, FGVC-Aircraft, Stanford Cars.
- Current global pruning setting:
  - importance: `gate_taylor`
  - gate location: `fc2_in`
  - reduction: `sum_square`
  - normalizer: `None`
  - calibration split: full train split
  - calibration seed: `42`
  - calibration loss reduction: `sum`
- Full calibration examples:
  - CIFAR100: 50,000
  - CUB200: 5,994
  - FGVC-Aircraft: 6,667
  - Stanford Cars: 8,144

## Why Gate Taylor

The goal was to score structured MLP hidden channels directly. Instead of only using activation Taylor scores, an explicit channel-wise gate is inserted and the score is computed from the effect of removing that gate. The final score is still stored per MLP hidden channel, which matches the structured pruning unit used by Torch-Pruning.

Implementation notes:

- `MLPGateTaylorCollector` handles gate insertion and score accumulation.
- Scores are accumulated as element-wise gate-gradient contributions and then reduced to channel-wise scores.
- Gate score caches store block-indexed channel scores, not raw gradients.
- Cache helpers are in `pruning/gate_taylor_cache.py`.

## Step 1: Reduction Ablation

Question:

> Should gate Taylor contributions preserve sign, or should pruning importance use contribution magnitude?

Compared reductions:

- `sum_abs`
- `sum_square`
- `signed_damage`

Observation:

- `sum_square` was the most stable choice for subsequent experiments.
- `signed_damage` produced larger accuracy drops.

Interpretation:

- `signed_damage` keeps the direction of first-order Taylor contributions.
- A single MLP hidden channel can have different effects across samples and tokens.
- Mixed-sign contributions can cancel out.
- Important mixed-role channels can receive artificially low signed scores and be pruned too early.
- `sum_abs` and `sum_square` ignore direction and aggregate contribution magnitude, making pruning rankings more stable.

Important figures:

- `figures/gate_taylor_sensitivity/reduction_heatmaps/`
- `figures/gate_taylor_sensitivity/ppt_assets/gate_taylor_fc1_out_reduction_summary_table.png`

## Step 2: Gate Location Ablation

Question:

> Should the gate be placed before GELU (`fc1_out`) or after GELU (`fc2_in`)?

Observation:

- `fc2_in` was slightly better on average than `fc1_out`.
- The difference was small; the two locations are broadly comparable.
- `fc2_in + sum_square` was selected for global pruning.

Important files:

- `figures/gate_taylor_sensitivity/location_sum_square_comparison/`
- `figures/gate_taylor_sensitivity/fc2_in_sum_square/`

## Step 3: Global Pruning

Question:

> If channel scores are compared globally across all MLP blocks, how much can be pruned before accuracy collapses?

Global pruning was run at 40%, 50%, and 60% MLP hidden-channel pruning using cached `fc2_in + sum_square` Gate Taylor scores.

Pruning-only accuracy:

| Dataset | Dense | 40% | 50% | 60% |
|---|---:|---:|---:|---:|
| CIFAR100 | 92.11 | 87.66 | 79.72 | 58.96 |
| CUB200 | 87.73 | 83.33 | 81.27 | 73.78 |
| FGVC-Aircraft | 77.47 | 72.13 | 64.60 | 47.61 |
| Stanford Cars | 88.24 | 86.02 | 81.30 | 68.21 |

Interpretation:

- 40% pruning-only is relatively stable.
- 50% starts to show meaningful degradation.
- 60% pruning-only collapses on several datasets.
- High-ratio global pruning needs recovery or additional constraints.

Important files:

- `figures/gate_taylor_global_pruning/gate_taylor_global_pruning_only_table.png`
- `figures/gate_taylor_global_pruning/gate_taylor_global_summary.csv`
- `pruned/vit_base_<dataset>_lora50_gate_taylor_fc2_in_sum_square_global040/results.jsonl`
- `pruned/vit_base_<dataset>_lora50_gate_taylor_fc2_in_sum_square_global050/results.jsonl`
- `pruned/vit_base_<dataset>_lora50_gate_taylor_fc2_in_sum_square_global060/results.jsonl`

## Step 4: LoRA Recovery

Question:

> Are the aggressively pruned models structurally recoverable after LoRA fine-tuning?

Recovery accuracy:

| Dataset | Dense | 50% recovery | 60% recovery |
|---|---:|---:|---:|
| CIFAR100 | 92.11 | 91.57 | 90.80 |
| CUB200 | 87.73 | 86.88 | 85.81 |
| FGVC-Aircraft | 77.47 | 76.09 | 76.66 |
| Stanford Cars | 88.24 | 88.06 | 87.56 |

Interpretation:

- LoRA recovery restores both 50% and 60% pruned models close to dense baselines.
- This suggests that the pruned architectures remain recoverable even when pruning-only accuracy is poor.
- FGVC-Aircraft has a small anomaly where 60% recovery is higher than 50% recovery. Treat this cautiously as possible regularization or run-to-run noise unless repeated.

Important files:

- `figures/gate_taylor_global_pruning/gate_taylor_global_recovery_only_table.png`
- `figures/gate_taylor_global_pruning/gate_taylor_global_pruning_recovery_summary.csv`
- `runs/timm_pruned_lora_<dataset>_supervised/<timestamp>/command.txt`
- `runs/timm_pruned_lora_<dataset>_supervised/<timestamp>/args.json`

## Step 5: Sensitivity vs Global Allocation

Question:

> Is global pruning allocating budget to layers that are actually tolerant under layer-wise sensitivity?

Observation:

- Dataset-wise allocation heatmaps show global pruning does not prune each MLP block uniformly.
- Pruning concentrates heavily in late MLP blocks, especially blocks 9-10.
- Layer-wise sensitivity heatmaps show that late blocks are generally more tolerant.
- Therefore, the allocation is broadly sensitivity-aware.

Important nuance:

- The alignment is not perfect.
- Block 11 often appears tolerant in sensitivity heatmaps, but global pruning gives more budget to blocks 9-10 than block 11.
- This suggests channel-level Taylor ranking and layer-level sensitivity are related but not identical.
- The 60% pruning-only collapse is likely caused by simultaneous multi-layer structural change, not simply by ignoring sensitivity.

Important figures:

- `figures/gate_taylor_global_pruning/gate_taylor_global_layer_pruned_ratio_heatmap.png`
- `figures/gate_taylor_global_pruning/gate_taylor_global_layer_pruned_ratio_<dataset>.png`
- `figures/gate_taylor_sensitivity/fc2_in_sum_square/<dataset>_fc2_in_sum_square_sensitivity_heatmap.png`

## Step 6: Gate Taylor Aggregation Axis Ablation

Question:

> When gate Taylor contributions exist at sample-token-channel resolution, which axes should be summed before applying `sum_square`?

Aggregation variants:

- **Elementwise**: square each sample-token contribution, then sum over samples/tokens.
- **Samplewise**: sum over tokens per sample, square, then sum over samples.
- **Tokenwise**: sum over samples per token position, square, then sum over tokens.
- **Channelwise**: sum over all samples/tokens, then square.

Main observations:

- `samplewise` improves pruning-only accuracy at 40/50% across datasets, but loses that advantage at 60%.
- `tokenwise` is poor on CIFAR100 but tends to help fine-grained datasets.
- `channelwise` is consistently weak, likely because too much signed cancellation happens before squaring.
- Recovery follow-up (`samplewise50`, `tokenwise60`, seed 42) shows aggregation effects mostly persist but shrink after LoRA recovery.
- Early layer-allocation analysis suggests tokenwise gains are not mainly from changing how much each layer is pruned; within-layer channel selection is likely important.

Important files and exact numbers:

- `figures/gate_taylor_aggregation_global/gate_taylor_aggregation_accuracy_delta_table.png`
- `figures/gate_taylor_aggregation_global/gate_taylor_aggregation_accuracy_delta.csv`
- `figures/gate_taylor_aggregation_global/gate_taylor_samplewise50_recovery_comparison_table.png`
- `figures/gate_taylor_aggregation_global/gate_taylor_samplewise50_recovery_comparison.csv`
- `figures/gate_taylor_aggregation_global/gate_taylor_tokenwise60_recovery_comparison_table.png`
- `figures/gate_taylor_aggregation_global/gate_taylor_tokenwise60_recovery_comparison.csv`
- `figures/gate_taylor_aggregation_global/gate_taylor_aggregation_layer_pruned_ratio_delta.png`
- `analysis/plot_gate_taylor_aggregation_global.py`
- `analysis/plot_gate_taylor_aggregation_layer_distribution.py`
- `analysis/plot_gate_taylor_recovery_split_tables.py`

## Step 7: Feature-Dimension Masked CE Trial

Question:

> Can final-feature dimension selection provide a better neuron pruning signal?

Setup:

- CUB200, global MLP 50%, `fc2_in + sum_square + samplewise`.
- Feature dimensions were ranked by elementwise-square Taylor scores; low 10% dims were masked before classifier CE for gate Taylor calibration.

Result:

| Method | Pruning-only | Recovery |
|---|---:|---:|
| Standard CE Gate Taylor | 82.95 | 86.78 |
| Feature-dim masked CE | 82.43 | 86.50 |

Takeaway:

- Feature-dim masked CE was mathematically valid but did not outperform standard CE Gate Taylor.
- Interpretation: low-Taylor feature dims are not purely nuisance dimensions, and standard CE remains the stronger task-preserving neuron scoring signal.
- Figures: `figures/feature_dim_masking/`.

## Current Interpretation For Slides

Main message:

> Gate Taylor global pruning finds broadly sensitivity-aware MLP pruning allocations. However, high-ratio unconstrained pruning causes large pruning-only degradation, while LoRA recovery restores the pruned models close to dense baselines. At the score level, aggregation granularity is not a minor implementation detail: it changes pruning-only accuracy and leaves smaller but visible effects after recovery.

Supporting points:

- `sum_square` is preferred over `signed_damage` because magnitude aggregation avoids signed cancellation.
- `fc2_in` is the current gate location because it is slightly better than `fc1_out` and comparable in behavior.
- 60% pruning-only is too aggressive without recovery.
- Recovery results show the pruned structures are still trainable/recoverable.
- Layer allocation suggests the next step should test constraints or rebalancing rather than only increasing the global ratio.
- Aggregation-axis ablation should be framed as the current main observation: aggregation granularity changes pruning-only behavior and leaves smaller effects after recovery.
- Recovery follow-up is validation, not the main contribution.

## Step 8: Progressive Pruning Trials

Question:

> Does progressive re-scoring produce better pruning masks than one-shot scoring?

Protocol:

- Progressive pruning is implemented as a separate research pipeline under
  `progressive_pruning/`.
- CE-guided progressive pruning repeats score/prune to cumulative 10-60% MLP
  pruning targets.
- Fixed-prototype progressive pruning scores normalized CLS features against
  cached class prototypes from the source model.
- Adapted-source CE progressive starts from the LoRA-adapted checkpoint instead
  of the linear-probe source.
- A prune-recover variant adds 1 epoch of LoRA + classifier recovery before the
  next re-scoring step.
- Each saved ratio is recovered independently with a matched LoRA recovery
  recipe. Flowers102 is excluded from this round.

Main observations:

- CE progressive pruning did not clearly outperform matched one-shot pruning.
- Using an adapted source helped on fine-grained datasets, but progressive
  re-scoring itself still did not give a consistent gain over one-shot.
- Adding 1-epoch intermediate recovery before re-scoring changed the pruning
  path, but did not meaningfully improve final recovered accuracy.
- Recovery protocol is a major confound. Resetting and re-training the
  classifier during recovery substantially improved some recovered results,
  especially on FGVC-Aircraft and Stanford Cars.
- A 0% recovery control showed that fresh LoRA recovery with classifier reset
  can improve accuracy even without pruning, so pruning gains must be compared
  against matched recovery controls.

Interpretation:

- Simple CE-based progressive pruning is not the main source of gains under the
  current setup.
- Source adaptation and recovery design explain much of the observed difference.
- Classifier reset likely helps because pruning changes feature geometry; keeping
  the dense classifier can bias LoRA recovery toward the old feature-boundary
  alignment.
- Future comparisons should use matched recovery settings, include 0% controls,
  and test whether representation-aware scoring can select better masks than CE.

Current implementation:

- Main code lives in `progressive_pruning/objectives.py`,
  `progressive_pruning/pipeline.py`, `progressive_pruning/recovery.py`, and
  `progressive_pruning/representation.py`.
- New pruning configs live under `progressive_pruning/configs/pruning/`.
- New recovery configs live under `progressive_pruning/configs/recovery/`.
- Meeting figures and result tables are under `figures/progressive_pruning/`.

## Reproducibility Notes

- `main.py` saves `command.txt` and `args.json` in each run directory.
- Recovery checkpoints store the resolved args and command in `best_cls_ckpt.pth`.
- Global pruning score caches are under `pruned/cache/`.
- Generated CSV/PNG files are presentation artifacts; exact sources are `results.jsonl`, run `args.json`, and checkpoint metadata.

## Next Experiment Candidates

Primary next experiment:

- Improve iterative progressive pruning only if it is tested with stronger
  controls: longer intermediate recovery, smaller pruning steps, or classifier
  reset during intermediate recovery.
- Run adapted-source prototype progressive pruning to test whether
  representation-aware scoring gives a better mask than CE.
- Keep matched 0% controls for any recovery protocol that changes classifier
  reset, LoRA rank, or recovery budget.

Later analyses:

- Use cancellation-ratio and mask-overlap analyses to explain aggregation behavior.
- Revisit constrained or rebalanced global pruning if high-ratio layer allocation
  remains a bottleneck.
