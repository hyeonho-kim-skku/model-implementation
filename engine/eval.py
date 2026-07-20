import torch
import torch.nn.functional as F

from datasets import get_loader
from utils import move_to_device


@torch.no_grad()
def evaluate_classifier(model, dataloader, device, max_batches=None):
    """Evaluate a supervised classifier and return loss/accuracy metrics."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, labels = move_to_device(batch, device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        num_batches += 1

    if num_batches == 0:
        raise ValueError("No evaluation batches were processed.")

    return {
        "loss": total_loss / num_batches,
        "acc": 100.0 * correct / total,
    }


def evaluate_merged_checkpoint(
    checkpoint_path,
    device,
    dataset_name=None,
    batch_size=None,
    split="test",
    num_workers=4,
    max_batches=None,
):
    """Evaluate the dense model stored in checkpoint["merged_model"].

    LoRA checkpoints save both the wrapped LoRA state and a merged dense state.
    The pruning pipeline consumes the merged dense state, so this function
    provides the matching direct-evaluation path.
    """

    from pruning.checkpoint import build_dense_model_from_checkpoint

    checkpoint, model = build_dense_model_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    checkpoint_args = checkpoint.get("args", {})
    # Prefer explicit CLI values, but fall back to the training arguments saved
    # in the checkpoint so checkpoint eval does not require one config per run.
    dataset_name = dataset_name or checkpoint_args.get("dataset")
    batch_size = batch_size or checkpoint_args.get("batch_size")
    if dataset_name is None:
        raise ValueError("--dataset is required because checkpoint args do not contain dataset.")
    if batch_size is None:
        raise ValueError("--batch-size is required because checkpoint args do not contain batch_size.")

    dataloader = get_loader(
        dataset_name,
        batch_size,
        "test",
        train=(split == "train"),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    metrics = evaluate_classifier(model, dataloader, device, max_batches=max_batches)
    eval_config = {
        "dataset": dataset_name,
        "batch_size": batch_size,
        "split": split,
    }
    return checkpoint, metrics, eval_config


def evaluate_vpt_checkpoint(
    checkpoint_path,
    device,
    dataset_name=None,
    batch_size=None,
    split="test",
    num_workers=4,
    max_batches=None,
):
    """Rebuild and evaluate a prompt-tuned pruned model checkpoint."""

    from models import load_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("model_config")
    if not model_config or model_config.get("model") != "timm_pruned_vpt":
        raise ValueError("Checkpoint does not contain a timm_pruned_vpt model_config.")
    model = load_model(**model_config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    checkpoint_args = checkpoint.get("args", {})
    dataset_name = dataset_name or checkpoint_args.get("dataset")
    batch_size = batch_size or checkpoint_args.get("batch_size")
    if dataset_name is None:
        raise ValueError("--dataset is required because checkpoint args do not contain dataset.")
    if batch_size is None:
        raise ValueError("--batch-size is required because checkpoint args do not contain batch_size.")

    dataloader = get_loader(
        dataset_name,
        batch_size,
        "test",
        train=(split == "train"),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    metrics = evaluate_classifier(model, dataloader, device, max_batches=max_batches)
    return checkpoint, metrics, {
        "dataset": dataset_name,
        "batch_size": batch_size,
        "split": split,
        "checkpoint_type": "vpt",
    }


def evaluate_training_checkpoint(
    checkpoint_path,
    device,
    dataset_name=None,
    batch_size=None,
    split="test",
    num_workers=4,
    max_batches=None,
):
    """Dispatch checkpoint evaluation by the saved checkpoint format."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("model_config") or {}
    if model_config.get("model") == "timm_pruned_vpt":
        return evaluate_vpt_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            dataset_name=dataset_name,
            batch_size=batch_size,
            split=split,
            num_workers=num_workers,
            max_batches=max_batches,
        )
    return evaluate_merged_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        dataset_name=dataset_name,
        batch_size=batch_size,
        split=split,
        num_workers=num_workers,
        max_batches=max_batches,
    )


def run_checkpoint_eval(args, device):
    """CLI-facing wrapper for eval.py."""

    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required.")

    checkpoint, metrics, eval_config = evaluate_training_checkpoint(
        checkpoint_path=args.checkpoint_path,
        device=device,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        split=args.split,
        max_batches=args.max_batches,
    )

    print(f"[Eval] checkpoint: {args.checkpoint_path}")
    print(f"[Eval] dataset: {eval_config['dataset']} ({eval_config['split']})")
    print(f"[Eval] batch size: {eval_config['batch_size']}")
    checkpoint_type = eval_config.get("checkpoint_type", "merged")
    print(f"[Eval] checkpoint type: {checkpoint_type}")
    print(f"[Eval] loss: {metrics['loss']:.4f}")
    print(f"[Eval] acc: {metrics['acc']:.2f}%")
    print(f"[Eval] checkpoint acc meta: {checkpoint.get('acc')}")
