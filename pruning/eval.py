import torch

from datasets import get_loader
from engine import evaluate_classifier


def load_pruned_artifact(artifact_path, map_location="cpu"):
    artifact = torch.load(artifact_path, map_location=map_location)
    if "model" not in artifact:
        raise ValueError("Pruned artifact does not contain a serialized model.")
    return artifact


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
):
    artifact = load_pruned_artifact(artifact_path, map_location=device)
    model = artifact["model"].to(device)

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
    )
    return artifact, evaluate_classifier(model, loader, device, max_batches=max_batches)
