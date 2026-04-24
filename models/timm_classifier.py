import timm
import torch
import torch.nn as nn


class TIMMClassifier(nn.Module):
    """Plain timm backbone + classifier head without LoRA wrappers."""

    def __init__(self, backbone_name, num_classes, pretrained=False, img_size=None):
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
        self.encoder = timm.create_model(backbone_name, **create_model_kwargs)

        feature_dim = getattr(self.encoder, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"{backbone_name} does not expose encoder.num_features.")

        self.classifier = nn.Linear(feature_dim, num_classes)

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
