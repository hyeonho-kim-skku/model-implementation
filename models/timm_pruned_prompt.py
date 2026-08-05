"""Composable prompt recovery for structurally pruned timm ViTs."""

import torch
import torch.nn as nn

from .kv_prompt import KVPromptedAttention, inject_kv_prompts
from .layerwise_prompts import (
    LayerwisePromptTokens,
    normalize_prompt_tokens_per_layer,
)
from pruning.eval import load_pruned_artifact


VALID_PROMPT_MODES = {"shallow", "deep"}
VALID_PROMPT_COMPONENTS = {"vpt", "kv"}


def normalize_prompt_components(value):
    if isinstance(value, str):
        components = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        try:
            components = tuple(str(item).strip() for item in value)
        except TypeError as error:
            raise ValueError("prompt_components must be a CSV string or iterable.") from error
    components = tuple(dict.fromkeys(components))
    if not components:
        raise ValueError("prompt_components must include at least one component.")
    invalid = set(components) - VALID_PROMPT_COMPONENTS
    if invalid:
        raise ValueError(
            f"prompt_components must use {sorted(VALID_PROMPT_COMPONENTS)}, "
            f"got {sorted(invalid)}."
        )
    return tuple(name for name in ("vpt", "kv") if name in components)


class TIMMPrunedPromptRecovery(nn.Module):
    """Composable VPT and KV-prompt recovery for a pruned timm ViT."""

    def __init__(
        self,
        artifact_path,
        prompt_components="kv",
        prompt_mode="deep",
        num_prompt_tokens=1,
        reset_classifier=True,
        num_classes=None,
        prompt_init_std=0.02,
        prompt_tokens_per_layer=None,
        num_kv_prompt_tokens=5,
        kv_prompt_tokens_per_layer=None,
        prompt_allocation_label=None,
        model_name="timm_pruned_prompt",
    ):
        super().__init__()
        if artifact_path is None:
            raise ValueError("artifact_path must be provided for prompt recovery.")

        self.model_name = model_name
        self.prompt_components = normalize_prompt_components(prompt_components)
        self.has_vpt = "vpt" in self.prompt_components
        self.has_kv = "kv" in self.prompt_components
        self.prompt_mode = prompt_mode
        self.num_prompt_tokens = (
            int(num_prompt_tokens) if num_prompt_tokens is not None else 1
        )
        self.num_kv_prompt_tokens = (
            int(num_kv_prompt_tokens) if num_kv_prompt_tokens is not None else 1
        )
        self.prompt_init_std = float(prompt_init_std)
        self.prompt_allocation_label = prompt_allocation_label
        self.reset_classifier = bool(reset_classifier)

        self._validate_requested_prompts(
            prompt_tokens_per_layer=prompt_tokens_per_layer,
            kv_prompt_tokens_per_layer=kv_prompt_tokens_per_layer,
        )

        self.artifact_path = artifact_path
        self.artifact = load_pruned_artifact(artifact_path)
        self.model = self.artifact["model"]
        self.model_config = self.artifact.get("model_config")
        self.source_pruning_config = self.artifact.get("pruning_config")
        self.source_pruning_stats = self.artifact.get("pruning_stats")

        self._validate_encoder()
        self.prompt_tokens_per_layer = normalize_prompt_tokens_per_layer(
            prompt_tokens_per_layer,
            self.num_blocks,
        )
        self.kv_prompt_tokens_per_layer = normalize_prompt_tokens_per_layer(
            kv_prompt_tokens_per_layer,
            self.num_blocks,
        )
        if self.has_kv and sum(self.resolved_kv_prompt_tokens_per_layer) == 0:
            raise ValueError("KV recovery requires at least one KV prompt token.")

        self._freeze_source_model()
        if self.reset_classifier:
            self._reset_classifier(num_classes)
        else:
            self._unfreeze_classifier()
        self._inject_kv_prompt_modules()
        self._create_vpt_parameters()
        self._log_configuration()

    @property
    def encoder(self):
        return self.model.encoder

    def _validate_requested_prompts(
        self,
        *,
        prompt_tokens_per_layer,
        kv_prompt_tokens_per_layer,
    ):
        if self.has_vpt:
            if self.prompt_mode not in VALID_PROMPT_MODES:
                raise ValueError(
                    f"prompt_mode must be one of {sorted(VALID_PROMPT_MODES)}, "
                    f"got {self.prompt_mode!r}."
                )
            if prompt_tokens_per_layer is not None and self.prompt_mode != "deep":
                raise ValueError("prompt_tokens_per_layer is supported only in deep mode.")
            if prompt_tokens_per_layer is None and self.num_prompt_tokens <= 0:
                raise ValueError("num_prompt_tokens must be greater than 0.")
            if self.prompt_init_std <= 0:
                raise ValueError("prompt_init_std must be greater than 0.")
        elif prompt_tokens_per_layer is not None:
            raise ValueError("prompt_tokens_per_layer requires the vpt component.")

        if self.has_kv:
            if kv_prompt_tokens_per_layer is None and self.num_kv_prompt_tokens <= 0:
                raise ValueError("num_kv_prompt_tokens must be greater than 0.")
        elif kv_prompt_tokens_per_layer is not None:
            raise ValueError("kv_prompt_tokens_per_layer requires the kv component.")

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
                "Prompt recovery currently supports timm VisionTransformer encoders; "
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
            raise ValueError("Prompt recovery requires at least one prefix token.")
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

    def _inject_kv_prompt_modules(self):
        if not self.has_kv:
            self.kv_prompted_layer_indices = ()
            return
        self.kv_prompted_layer_indices = inject_kv_prompts(
            self.encoder.blocks,
            self.resolved_kv_prompt_tokens_per_layer,
        )

    def _create_vpt_parameters(self):
        if not self.has_vpt:
            return
        if self.prompt_mode == "shallow":
            self.prompt_embeddings = nn.Parameter(
                torch.empty(1, self.num_prompt_tokens, self.embed_dim)
            )
            nn.init.trunc_normal_(self.prompt_embeddings, std=self.prompt_init_std)
        elif self.prompt_tokens_per_layer is None:
            self.deep_prompt_embeddings = nn.Parameter(
                torch.empty(
                    self.num_blocks,
                    self.num_prompt_tokens,
                    self.embed_dim,
                )
            )
            nn.init.trunc_normal_(
                self.deep_prompt_embeddings, std=self.prompt_init_std
            )
        else:
            self.layerwise_prompts = LayerwisePromptTokens(
                token_counts=self.prompt_tokens_per_layer,
                embedding_dim=self.embed_dim,
                init_std=self.prompt_init_std,
            )

    @property
    def prompt_parameters(self):
        if not self.has_vpt:
            return ()
        if self.prompt_mode == "shallow":
            return self.prompt_embeddings
        if self.prompt_tokens_per_layer is not None:
            return self.layerwise_prompts.parameters()
        return self.deep_prompt_embeddings

    @property
    def resolved_prompt_tokens_per_layer(self):
        """Legacy alias for the visual-prompt allocation."""
        return self.resolved_vpt_prompt_tokens_per_layer

    @property
    def resolved_vpt_prompt_tokens_per_layer(self):
        if not self.has_vpt:
            return (0,) * self.num_blocks
        if self.prompt_mode == "shallow":
            return (self.num_prompt_tokens,)
        if self.prompt_tokens_per_layer is not None:
            return self.prompt_tokens_per_layer
        return (self.num_prompt_tokens,) * self.num_blocks

    @property
    def resolved_kv_prompt_tokens_per_layer(self):
        if not self.has_kv:
            return (0,) * self.num_blocks
        if self.kv_prompt_tokens_per_layer is not None:
            return self.kv_prompt_tokens_per_layer
        return (self.num_kv_prompt_tokens,) * self.num_blocks

    @property
    def total_prompt_tokens(self):
        """Legacy alias for total visual prompt tokens."""
        return self.total_vpt_prompt_tokens

    @property
    def total_vpt_prompt_tokens(self):
        return sum(self.resolved_vpt_prompt_tokens_per_layer)

    @property
    def total_kv_prompt_tokens(self):
        return sum(self.resolved_kv_prompt_tokens_per_layer)

    @property
    def vpt_prompt_parameter_count(self):
        if not self.has_vpt:
            return 0
        if self.prompt_mode == "shallow":
            return self.prompt_embeddings.numel()
        if self.prompt_tokens_per_layer is None:
            return self.deep_prompt_embeddings.numel()
        return sum(parameter.numel() for parameter in self.layerwise_prompts.parameters())

    @property
    def kv_prompt_parameter_count(self):
        if not self.has_kv:
            return 0
        return sum(
            block.attn.prompt_parameter_count
            for block in self.encoder.blocks
            if isinstance(block.attn, KVPromptedAttention)
        )

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

    def _remove_prompt(self, x, num_prompt_tokens=None):
        if num_prompt_tokens is None:
            num_prompt_tokens = self.num_prompt_tokens
        prefix = x[:, : self.num_prefix_tokens]
        tokens = x[:, self.num_prefix_tokens + num_prompt_tokens :]
        return torch.cat((prefix, tokens), dim=1)

    def _forward_blocks(self, x):
        for block in self.encoder.blocks:
            x = block(x)
        return x

    def _forward_shallow(self, x):
        x = self._insert_prompt(x, self.prompt_embeddings)
        return self._forward_blocks(x)

    def _forward_deep(self, x):
        for block_idx, block in enumerate(self.encoder.blocks):
            if self.prompt_tokens_per_layer is None:
                prompt = self.deep_prompt_embeddings[block_idx].unsqueeze(0)
            else:
                prompt = self.layerwise_prompts.prompt_for_layer(block_idx)
            num_prompt_tokens = prompt.shape[1]
            if num_prompt_tokens:
                x = self._insert_prompt(x, prompt)
            x = block(x)
            if num_prompt_tokens:
                x = self._remove_prompt(x, num_prompt_tokens)
        return x

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen encoder deterministic while retaining autograd through
        # it so gradients can reach both kinds of prompt.
        self.encoder.eval()
        self.model.classifier.train(mode)
        return self

    def forward_features(self, images):
        x = self._embed_images(images)
        if not self.has_vpt:
            x = self._forward_blocks(x)
        elif self.prompt_mode == "shallow":
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

    def _log_configuration(self):
        trainable_params, total_params = self.count_parameters()
        if self.model_name == "timm_pruned_vpt":
            print(f"[TIMMPrunedVPT] artifact: {self.artifact_path}")
            print(f"[TIMMPrunedVPT] prompt mode: {self.prompt_mode}")
            print(f"[TIMMPrunedVPT] prompt tokens: {self.num_prompt_tokens}")
            print(
                "[TIMMPrunedVPT] prompt tokens per layer: "
                f"{list(self.resolved_vpt_prompt_tokens_per_layer)}"
            )
            print(f"[TIMMPrunedVPT] total prompt tokens: {self.total_vpt_prompt_tokens}")
            prefix = "TIMMPrunedVPT"
        else:
            print(f"[TIMMPrunedPrompt] artifact: {self.artifact_path}")
            print(
                "[TIMMPrunedPrompt] components: "
                f"{','.join(self.prompt_components)}"
            )
            print(f"[TIMMPrunedPrompt] VPT mode: {self.prompt_mode if self.has_vpt else 'none'}")
            print(
                "[TIMMPrunedPrompt] VPT tokens per layer: "
                f"{list(self.resolved_vpt_prompt_tokens_per_layer)}"
            )
            print(
                "[TIMMPrunedPrompt] KV tokens per layer: "
                f"{list(self.resolved_kv_prompt_tokens_per_layer)}"
            )
            print(f"[TIMMPrunedPrompt] total VPT tokens: {self.total_vpt_prompt_tokens}")
            print(f"[TIMMPrunedPrompt] total KV tokens: {self.total_kv_prompt_tokens}")
            print(
                "[TIMMPrunedPrompt] VPT prompt params: "
                f"{self.vpt_prompt_parameter_count:,}"
            )
            print(
                "[TIMMPrunedPrompt] KV prompt params: "
                f"{self.kv_prompt_parameter_count:,}"
            )
            prefix = "TIMMPrunedPrompt"
        if self.prompt_allocation_label:
            print(f"[{prefix}] allocation label: {self.prompt_allocation_label}")
        print(f"[{prefix}] reset classifier: {self.reset_classifier}")
        print(
            f"[{prefix}] trainable params: {trainable_params:,} / "
            f"{total_params:,} ({100.0 * trainable_params / total_params:.4f}%)"
        )

    def export_config(self):
        config = {
            "model": self.model_name,
            "artifact_path": self.artifact_path,
            "prompt_components": list(self.prompt_components),
            "prompt_mode": self.prompt_mode,
            "num_prompt_tokens": self.num_prompt_tokens,
            "prompt_tokens_per_layer": (
                list(self.prompt_tokens_per_layer)
                if self.prompt_tokens_per_layer is not None
                else None
            ),
            "resolved_vpt_prompt_tokens_per_layer": list(
                self.resolved_vpt_prompt_tokens_per_layer
            ),
            "total_vpt_prompt_tokens": self.total_vpt_prompt_tokens,
            "num_kv_prompt_tokens": self.num_kv_prompt_tokens,
            "kv_prompt_tokens_per_layer": (
                list(self.kv_prompt_tokens_per_layer)
                if self.kv_prompt_tokens_per_layer is not None
                else None
            ),
            "resolved_kv_prompt_tokens_per_layer": list(
                self.resolved_kv_prompt_tokens_per_layer
            ),
            "total_kv_prompt_tokens": self.total_kv_prompt_tokens,
            "vpt_prompt_parameter_count": self.vpt_prompt_parameter_count,
            "kv_prompt_parameter_count": self.kv_prompt_parameter_count,
            "kv_share_key_value": True,
            "kv_prompt_init": "kaiming_uniform",
            "prompt_allocation_label": self.prompt_allocation_label,
            "prompt_init_std": self.prompt_init_std,
            "reset_classifier": self.reset_classifier,
            "num_classes": self.model.classifier.out_features,
            "embedding_dim": self.embed_dim,
            "num_blocks": self.num_blocks,
            "remaining_heads_per_layer": [
                int(block.attn.num_heads)
                for block in self.encoder.blocks
                if hasattr(block, "attn") and hasattr(block.attn, "num_heads")
            ],
            "source_pruning_config": self.source_pruning_config,
            "source_pruning_stats": self.source_pruning_stats,
        }
        if self.model_name == "timm_pruned_vpt":
            config["resolved_prompt_tokens_per_layer"] = list(
                self.resolved_vpt_prompt_tokens_per_layer
            )
            config["total_prompt_tokens"] = self.total_vpt_prompt_tokens
        return config


class TIMMPrunedVPT(TIMMPrunedPromptRecovery):
    """Backward-compatible VPT-only entrypoint and checkpoint format."""

    def __init__(
        self,
        artifact_path,
        prompt_mode="shallow",
        num_prompt_tokens=1,
        reset_classifier=True,
        num_classes=None,
        prompt_init_std=0.02,
        prompt_tokens_per_layer=None,
        prompt_allocation_label=None,
    ):
        super().__init__(
            artifact_path=artifact_path,
            prompt_components=("vpt",),
            prompt_mode=prompt_mode,
            num_prompt_tokens=num_prompt_tokens,
            reset_classifier=reset_classifier,
            num_classes=num_classes,
            prompt_init_std=prompt_init_std,
            prompt_tokens_per_layer=prompt_tokens_per_layer,
            prompt_allocation_label=prompt_allocation_label,
            model_name="timm_pruned_vpt",
        )
