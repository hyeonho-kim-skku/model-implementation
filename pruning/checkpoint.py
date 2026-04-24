import torch

from models.timm_classifier import TIMMClassifier


def load_checkpoint(checkpoint_path, map_location="cpu"):
    return torch.load(checkpoint_path, map_location=map_location)


def build_dense_model_from_checkpoint(checkpoint_path, map_location="cpu"):
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)

    if "model_config" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_config.")
    if "merged_model" not in checkpoint:
        raise ValueError("Checkpoint does not contain merged_model.")

    model_config = checkpoint["model_config"]
    merged_model = checkpoint["merged_model"]

    if model_config.get("model") != "timm_lora":
        raise ValueError("Only timm_lora checkpoints are currently supported for dense reconstruction.")

    dense_model = TIMMClassifier(
        backbone_name=model_config["backbone_name"],
        num_classes=model_config["num_classes"],
        pretrained=False,
        img_size=model_config.get("img_size"),
    )
    dense_model.load_state_dict(
        {
            **{f"encoder.{name}": tensor for name, tensor in merged_model["encoder"].items()},
            **{f"classifier.{name}": tensor for name, tensor in merged_model["classifier"].items()},
        },
        strict=True,
    )
    return checkpoint, dense_model
