import torch
import torch.nn as nn

from pruning.eval import load_pruned_artifact


VALID_PROMPT_MODES = {"shallow", "deep"}


class TIMMPrunedVPT(nn.Module):
    """Visual prompt tuning recovery for a structurally pruned timm ViT."""

    def __init__(
        self,
        artifact_path,
        prompt_mode="shallow",
        num_prompt_tokens=1,
        reset_classifier=True,
        num_classes=None,
        prompt_init_std=0.02,
    ):
        super().__init__()
        if artifact_path is None:
            raise ValueError("artifact_path must be provided for model='timm_pruned_vpt'.")
        if prompt_mode not in VALID_PROMPT_MODES:
            raise ValueError(
                f"prompt_mode must be one of {sorted(VALID_PROMPT_MODES)}, "
                f"got {prompt_mode!r}."
            )
        if num_prompt_tokens is None or int(num_prompt_tokens) <= 0:
            raise ValueError("num_prompt_tokens must be greater than 0.")
        if float(prompt_init_std) <= 0:
            raise ValueError("prompt_init_std must be greater than 0.")

        self.artifact_path = artifact_path
        self.artifact = load_pruned_artifact(artifact_path)
        self.model = self.artifact["model"]
        self.prompt_mode = prompt_mode
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.prompt_init_std = float(prompt_init_std)
        self.reset_classifier = bool(reset_classifier)
        self.model_config = self.artifact.get("model_config")
        self.source_pruning_config = self.artifact.get("pruning_config")
        self.source_pruning_stats = self.artifact.get("pruning_stats")

        self._validate_encoder()
        self._freeze_source_model()
        if self.reset_classifier:
            self._reset_classifier(num_classes)
        else:
            self._unfreeze_classifier()
        self._create_prompt_parameters()

        trainable_params, total_params = self.count_parameters()
        print(f"[TIMMPrunedVPT] artifact: {artifact_path}")
        print(f"[TIMMPrunedVPT] prompt mode: {self.prompt_mode}")
        print(f"[TIMMPrunedVPT] prompt tokens: {self.num_prompt_tokens}")
        print(f"[TIMMPrunedVPT] reset classifier: {self.reset_classifier}")
        print(
            f"[TIMMPrunedVPT] trainable params: {trainable_params:,} / "
            f"{total_params:,} ({100.0 * trainable_params / total_params:.4f}%)"
        )

    @property
    def encoder(self):
        return self.model.encoder

    def _validate_encoder(self):
        if not hasattr(self.model, "encoder") or not hasattr(self.model, "classifier"):
            raise TypeError("The pruned artifact must contain a TIMMClassifier-like model.")
        required = (
            "patch_embed",
            "_pos_embed",
            "patch_drop",
            "norm_pre",
            "blocks",
            "norm",
            "forward_head",
            "num_prefix_tokens",
        )
        missing = [name for name in required if not hasattr(self.encoder, name)]
        if missing:
            raise TypeError(
                "TIMMPrunedVPT currently supports timm VisionTransformer encoders; "
                f"missing attributes: {missing}."
            )
        if len(self.encoder.blocks) == 0:
            raise ValueError("The encoder must contain at least one transformer block.")
        embed_dim = getattr(self.encoder, "embed_dim", None)
        if embed_dim is None:
            embed_dim = getattr(self.encoder, "num_features", None)
        if embed_dim is None:
            raise ValueError("The encoder does not expose embed_dim or num_features.")
        if int(self.encoder.num_prefix_tokens) < 1:
            raise ValueError("TIMMPrunedVPT requires at least one prefix token for CLS pooling.")
        self.embed_dim = int(embed_dim)
        self.num_blocks = len(self.encoder.blocks)
        self.num_prefix_tokens = int(self.encoder.num_prefix_tokens)

    def _freeze_source_model(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.freeze_encoder = True

    def _reset_classifier(self, num_classes):
        if num_classes is None:
            num_classes = self.model.classifier.out_features
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            int(num_classes),
        )

    def _unfreeze_classifier(self):
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

    def _create_prompt_parameters(self):
        if self.prompt_mode == "shallow":
            self.prompt_embeddings = nn.Parameter(
                torch.empty(1, self.num_prompt_tokens, self.embed_dim)
            )
        else:
            self.deep_prompt_embeddings = nn.Parameter(
                torch.empty(
                    self.num_blocks,
                    self.num_prompt_tokens,
                    self.embed_dim,
                )
            )
        nn.init.trunc_normal_(self.prompt_parameters, std=self.prompt_init_std)

    @property
    def prompt_parameters(self):
        if self.prompt_mode == "shallow":
            return self.prompt_embeddings
        return self.deep_prompt_embeddings

    def _embed_images(self, images):
        x = self.encoder.patch_embed(images)
        x = self.encoder._pos_embed(x)
        x = self.encoder.patch_drop(x)
        return self.encoder.norm_pre(x)

    def _insert_prompt(self, x, prompt):
        prefix = x[:, : self.num_prefix_tokens]
        tokens = x[:, self.num_prefix_tokens :]
        prompt = prompt.expand(x.shape[0], -1, -1)
        return torch.cat((prefix, prompt, tokens), dim=1)

    def _remove_prompt(self, x):
        prefix = x[:, : self.num_prefix_tokens]
        tokens = x[:, self.num_prefix_tokens + self.num_prompt_tokens :]
        return torch.cat((prefix, tokens), dim=1)

    def _forward_shallow(self, x):
        x = self._insert_prompt(x, self.prompt_embeddings)
        for block in self.encoder.blocks:
            x = block(x)
        return x

    def _forward_deep(self, x):
        for block_idx, block in enumerate(self.encoder.blocks):
            prompt = self.deep_prompt_embeddings[block_idx].unsqueeze(0)
            x = self._insert_prompt(x, prompt)
            x = block(x)
            x = self._remove_prompt(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen encoder deterministic while retaining autograd through
        # its operations so gradients can reach the prompt embeddings.
        self.encoder.eval()
        self.model.classifier.train(mode)
        return self

    def forward_features(self, images):
        x = self._embed_images(images)
        if self.prompt_mode == "shallow":
            x = self._forward_shallow(x)
        else:
            x = self._forward_deep(x)
        x = self.encoder.norm(x)
        return self.encoder.forward_head(x, pre_logits=True)

    def forward(self, images):
        return self.model.classifier(self.forward_features(images))

    def count_parameters(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return trainable, total

    def export_config(self):
        return {
            "model": "timm_pruned_vpt",
            "artifact_path": self.artifact_path,
            "prompt_mode": self.prompt_mode,
            "num_prompt_tokens": self.num_prompt_tokens,
            "prompt_init_std": self.prompt_init_std,
            "reset_classifier": self.reset_classifier,
            "num_classes": self.model.classifier.out_features,
            "embedding_dim": self.embed_dim,
            "num_blocks": self.num_blocks,
            "source_pruning_config": self.source_pruning_config,
            "source_pruning_stats": self.source_pruning_stats,
        }
