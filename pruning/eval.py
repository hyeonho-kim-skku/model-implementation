import torch

from datasets import build_timm_eval_transform, get_loader
from engine import evaluate_classifier


VALID_PRUNED_EVALUATION_TRANSFORMS = {"default", "timm_pretrained"}


def load_pruned_artifact(artifact_path, map_location="cpu"):
    artifact = torch.load(artifact_path, map_location=map_location)
    if "model" not in artifact:
        raise ValueError("Pruned artifact does not contain a serialized model.")
    return artifact


def build_pruned_evaluation_transform(model, evaluation_transform="default"):
    """Build an optional evaluation transform for a pruned artifact model.

    ``default`` intentionally returns no override so callers retain the legacy
    dataset ``test`` transform. ``timm_pretrained`` resolves the data config
    retained by the artifact's timm encoder, matching dense timm evaluation.
    """

    evaluation_transform = str(evaluation_transform).lower()
    if evaluation_transform not in VALID_PRUNED_EVALUATION_TRANSFORMS:
        raise ValueError(
            "evaluation_transform must be one of "
            f"{sorted(VALID_PRUNED_EVALUATION_TRANSFORMS)}, "
            f"got {evaluation_transform!r}."
        )
    if evaluation_transform == "default":
        return None, {
            "preset": "default",
            "mode": "test",
        }

    transform, data_config = build_timm_eval_transform(model)
    return transform, {
        "preset": "timm_pretrained",
        "mode": "timm_pretrained",
        "data_config": dict(data_config),
    }


@torch.no_grad()
def evaluate_pruned_model(
    artifact_path,
    dataset_name,
    batch_size=64,
    split="test",
    device="cpu",
    num_workers=4,
    max_batches=None,
    data_root="./data",
    evaluation_transform="default",
):
    artifact = load_pruned_artifact(artifact_path, map_location=device)
    model = artifact["model"].to(device)
    transform, _ = build_pruned_evaluation_transform(
        model,
        evaluation_transform=evaluation_transform,
    )

    # Reuse the project-level classifier evaluator so pruned and non-pruned
    # checkpoints use the same loss/accuracy calculation.
    loader = get_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        mode="test",
        train=(split == "train"),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        data_root=data_root,
        transform=transform,
    )
    return artifact, evaluate_classifier(model, loader, device, max_batches=max_batches)
