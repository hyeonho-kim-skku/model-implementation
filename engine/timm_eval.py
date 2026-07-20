"""Evaluation helpers for pretrained timm classifier baselines."""

from datasets import build_timm_eval_transform, get_loader
from models.timm_classifier import TIMMClassifier

from .eval import evaluate_classifier


def evaluate_timm_baseline(
    *,
    backbone_name,
    dataset_name,
    data_root,
    device,
    num_classes=1000,
    img_size=None,
    pretrained=True,
    classifier_init="pretrained",
    batch_size=128,
    num_workers=8,
    max_batches=None,
):
    """Evaluate a dense timm baseline with its pretrained data configuration."""

    model = TIMMClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        img_size=img_size,
        freeze_encoder=False,
        classifier_init=classifier_init,
    ).to(device)
    transform, data_config = build_timm_eval_transform(model)
    loader = get_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        mode="test",
        train=False,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        data_root=data_root,
        transform=transform,
    )
    raw_metrics = evaluate_classifier(
        model,
        loader,
        device,
        max_batches=max_batches,
    )
    return {
        "metrics": {
            "loss": raw_metrics["loss"],
            "top1": raw_metrics["acc"],
        },
        "model_config": model.export_config(),
        "data_config": dict(data_config),
        "evaluation": {
            "dataset": dataset_name,
            "split": "val",
            "data_root": data_root,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_batches": max_batches,
        },
    }
