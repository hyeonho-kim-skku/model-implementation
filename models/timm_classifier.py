import copy

import timm
import torch
import torch.nn as nn


class TIMMClassifier(nn.Module):
    """Plain timm backbone + classifier head without LoRA wrappers."""

    VALID_CLASSIFIER_INITS = {"random", "pretrained"}

    def __init__(
        self,
        backbone_name,
        num_classes,
        pretrained=False,
        img_size=None,
        freeze_encoder=False,
        classifier_init="random",
    ):
        super().__init__()
        if backbone_name is None:
            raise ValueError("backbone_name must be provided for model='timm_classifier'.")
        if num_classes is None:
            raise ValueError("num_classes must be provided for model='timm_classifier'.")
        classifier_init = str(classifier_init).lower()
        if classifier_init not in self.VALID_CLASSIFIER_INITS:
            raise ValueError(
                f"classifier_init must be one of {sorted(self.VALID_CLASSIFIER_INITS)}, "
                f"got {classifier_init!r}."
            )
        if classifier_init == "pretrained" and not pretrained:
            raise ValueError(
                "classifier_init='pretrained' requires pretrained=True."
            )

        create_model_kwargs = {
            "pretrained": pretrained,
            "num_classes": num_classes if classifier_init == "pretrained" else 0,
        }
        if img_size is not None:
            create_model_kwargs["img_size"] = img_size

        self.backbone_name = backbone_name
        self.img_size = img_size
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder
        self.classifier_init = classifier_init

        if classifier_init == "pretrained":
            pretrained_config = timm.get_pretrained_cfg(backbone_name)
            pretrained_num_classes = int(pretrained_config.num_classes)
            if int(num_classes) != pretrained_num_classes:
                raise ValueError(
                    "classifier_init='pretrained' requires num_classes to match "
                    f"the pretrained weight configuration: {num_classes} != "
                    f"{pretrained_num_classes}."
                )

        self.encoder = timm.create_model(backbone_name, **create_model_kwargs)

        if classifier_init == "pretrained":
            pretrained_classifier = self.encoder.get_classifier()
            if not isinstance(pretrained_classifier, nn.Linear):
                raise TypeError(
                    "TIMMClassifier currently supports pretrained nn.Linear "
                    f"classifiers, got {type(pretrained_classifier).__name__}."
                )
            self.classifier = copy.deepcopy(pretrained_classifier)
            self.encoder.reset_classifier(0)

        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        feature_dim = getattr(self.encoder, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"{backbone_name} does not expose encoder.num_features.")

        if classifier_init == "random":
            self.classifier = nn.Linear(feature_dim, num_classes)
        self.trainable_params, self.total_params = self.count_parameters()

        print(f"[TIMMClassifier] backbone: {self.backbone_name}")
        print(f"[TIMMClassifier] classifier init: {self.classifier_init}")
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
            "classifier_init": self.classifier_init,
        }
