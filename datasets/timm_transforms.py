"""Model-specific evaluation transforms for timm pretrained weights."""

from timm.data import create_transform, resolve_model_data_config


def build_timm_eval_transform(model):
    """Build the exact evaluation transform advertised by a timm model."""

    backbone = getattr(model, "encoder", model)
    data_config = resolve_model_data_config(backbone)
    transform = create_transform(**data_config, is_training=False)
    return transform, data_config
