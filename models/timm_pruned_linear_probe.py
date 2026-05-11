import torch.nn as nn

from pruning.eval import load_pruned_artifact


class TIMMPrunedLinearProbe(nn.Module):
    """Linear probing wrapper for a pruned TIMMClassifier artifact."""

    def __init__(
        self,
        artifact_path,
        reset_classifier=True,
        num_classes=None,
        freeze_encoder=True,
    ):
        super().__init__()
        if artifact_path is None:
            raise ValueError("artifact_path must be provided for model='timm_pruned_linear_probe'.")

        # The pruning artifact stores a full TIMMClassifier object whose encoder
        # has already been structurally pruned. This wrapper reuses that pruned
        # encoder and controls only the probing head/training policy.
        self.artifact_path = artifact_path
        self.artifact = load_pruned_artifact(artifact_path)
        self.model = self.artifact["model"]

        self.reset_classifier = reset_classifier
        self.freeze_encoder = freeze_encoder
        self.model_config = self.artifact.get("model_config")
        self.source_pruning_config = self.artifact.get("pruning_config")
        self.source_pruning_stats = self.artifact.get("pruning_stats")

        if self.reset_classifier:
            self._reset_classifier(num_classes)
        if self.freeze_encoder:
            self._freeze_encoder()

        trainable_params, total_params = self.count_parameters()
        print(f"[TIMMPrunedLinearProbe] artifact: {artifact_path}")
        print(f"[TIMMPrunedLinearProbe] reset_classifier: {self.reset_classifier}")
        print(f"[TIMMPrunedLinearProbe] freeze_encoder: {self.freeze_encoder}")
        print(
            f"[TIMMPrunedLinearProbe] trainable params: {trainable_params:,} / "
            f"{total_params:,} ({100.0 * trainable_params / total_params:.2f}%)"
        )

    def _reset_classifier(self, num_classes):
        # Linear probing should measure the pruned encoder with a fresh task
        # head. Reusing a classifier from another run would mix probe quality
        # with warm-start effects.
        if num_classes is None:
            num_classes = self.model.classifier.out_features
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def _freeze_encoder(self):
        # TIMMClassifier.train() checks freeze_encoder and keeps the encoder in
        # eval mode during training, which is the behavior expected for probing.
        self.model.freeze_encoder = True
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = False

    def forward_features(self, x):
        return self.model.forward_features(x)

    def forward(self, x):
        return self.model(x)

    def count_parameters(self):
        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return trainable_params, total_params

    def export_config(self):
        return {
            "model": "timm_pruned_linear_probe",
            "source_pruned_artifact_path": self.artifact_path,
            "source_pruning_config": self.source_pruning_config,
            "source_pruning_stats": self.source_pruning_stats,
            "model_config": self.model_config,
            "reset_classifier": self.reset_classifier,
            "freeze_encoder": self.freeze_encoder,
            "num_classes": self.model.classifier.out_features,
        }
