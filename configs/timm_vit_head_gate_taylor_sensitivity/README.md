# ViT Head Gate Taylor Sensitivity Configs

Layer-wise attention-head pruning sensitivity configs for ViT-Base LoRA
checkpoints.

Common setting:

- `importance: head_gate_taylor`
- `pruning_modules: head`
- `head_gate_taylor_reduction: sum_abs`
- `head_gate_taylor_aggregation: samplewise`
- `calibration_batches: full`

The gate location and pruning root are intentionally not config options in the
CLI right now; both use the internal `proj_in` default.
