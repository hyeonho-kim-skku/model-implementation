import timm
import torch
import torch.nn as nn


class TIMMClassifier(nn.Module):
    """Plain timm backbone + classifier head without LoRA wrappers."""

    def __init__(
        self,
        backbone_name,
        num_classes,
        pretrained=False,
        img_size=None,
        freeze_encoder=False,
    ):
        super().__init__()
        if backbone_name is None:
            raise ValueError("backbone_name must be provided for model='timm_classifier'.")
        if num_classes is None:
            raise ValueError("num_classes must be provided for model='timm_classifier'.")

        create_model_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
        }
        if img_size is not None:
            create_model_kwargs["img_size"] = img_size

        self.backbone_name = backbone_name
        self.img_size = img_size
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder
        self.encoder = timm.create_model(backbone_name, **create_model_kwargs)
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        feature_dim = getattr(self.encoder, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"{backbone_name} does not expose encoder.num_features.")

        self.classifier = nn.Linear(feature_dim, num_classes)
        self.trainable_params, self.total_params = self.count_parameters()

        print(f"[TIMMClassifier] backbone: {self.backbone_name}")
        print(f"[TIMMClassifier] freeze_encoder: {self.freeze_encoder}")
        print(
            f"[TIMMClassifier] trainable params: {self.trainable_params:,} / "
            f"{self.total_params:,} ({100.0 * self.trainable_params / self.total_params:.2f}%)"
        )

    def train(self, mode=True):
        # Keep the default PyTorch train/eval behavior for this module and all
        # children first. model.eval() also reaches this method as train(False).
        super().train(mode)
        # In linear probing, the encoder should behave as a fixed feature
        # extractor: no parameter updates and no train-mode Dropout/BatchNorm.
        if mode and self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward_features(self, x):
        features = self.encoder.forward_features(x)

        if hasattr(self.encoder, "forward_head"):
            pooled = self.encoder.forward_head(features, pre_logits=True)
            if pooled.ndim == 2:
                return pooled

        if isinstance(features, torch.Tensor) and features.ndim == 3:
            return features[:, 0]
        return features

    def forward(self, x):
        features = self.forward_features(x)
        return self.classifier(features)

    def count_parameters(self):
        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return trainable_params, total_params

    def export_config(self):
        return {
            "model": "timm_classifier",
            "backbone_name": self.backbone_name,
            "img_size": self.img_size,
            "num_classes": self.classifier.out_features,
            "pretrained": self.pretrained,
            "freeze_encoder": self.freeze_encoder,
        }
