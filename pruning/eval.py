import torch
import torch.nn.functional as F

from datasets import get_loader
from utils import move_to_device


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
):
    artifact = load_pruned_artifact(artifact_path, map_location=device)
    model = artifact["model"].to(device)
    model.eval()

    loader = get_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        mode="test",
        train=(split == "train"),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, labels = move_to_device(batch, device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        num_batches += 1
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / num_batches
    acc = 100.0 * correct / total

    return artifact, {"loss": avg_loss, "acc": acc}
