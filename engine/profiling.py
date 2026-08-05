"""Opt-in runtime profiling helpers kept separate from the training entrypoint."""

import torch
from torch_pruning.utils import op_counter

from models.kv_prompt import KVPromptedAttention
from pruning.structured_core import count_ops_and_params


def _kv_prompt_attention_counter(module, inputs, _output):
    """Match Torch-Pruning's timm attention count with the longer K/V axis."""
    x = inputs[0]
    batch_size, query_tokens, input_dim = x.shape
    key_value_tokens = query_tokens + module.num_prompt_tokens
    attention_dim = module.attn_dim

    macs = query_tokens * attention_dim  # Q scaling
    macs += 3 * query_tokens * input_dim * attention_dim
    if module.qkv.bias is not None:
        macs += 3 * query_tokens * attention_dim

    per_head = query_tokens * key_value_tokens * (2 * module.head_dim + 1)
    macs += module.num_heads * per_head

    # Keep the same projection convention as the existing timm counter so old
    # VPT and new KV-prompt profiles remain directly comparable.
    macs += query_tokens * attention_dim * (attention_dim + 1)
    module.__flops__ += int(batch_size * macs)


def profile_model_macs(model, device, image_size=None):
    """Profile one-image MACs while preserving the model's training state."""
    if image_size is None:
        image_size = getattr(model.encoder.patch_embed, "img_size", (224, 224))
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    example = torch.randn(1, 3, *image_size, device=device)
    was_training = model.training
    model.eval()
    op_counter.CUSTOM_MODULES_MAPPING[KVPromptedAttention] = (
        _kv_prompt_attention_counter
    )
    with torch.no_grad():
        macs, _ = count_ops_and_params(model, example)
    model.train(was_training)
    print(f"[ModelProfile] MACs: {macs:,}")
    return macs
