"""Opt-in runtime profiling helpers kept separate from the training entrypoint."""

import torch

from pruning.structured_core import count_ops_and_params


def profile_model_macs(model, device, image_size=None):
    """Profile one-image MACs while preserving the model's training state."""
    if image_size is None:
        image_size = getattr(model.encoder.patch_embed, "img_size", (224, 224))
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    example = torch.randn(1, 3, *image_size, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        macs, _ = count_ops_and_params(model, example)
    model.train(was_training)
    print(f"[ModelProfile] MACs: {macs:,}")
    return macs
