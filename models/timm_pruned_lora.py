import copy

import torch.nn as nn

from pruning.eval import load_pruned_artifact

from .lora import FusedQKVLoRA, LoRAWrappedLinear
from .timm_lora import count_parameters, inject_lora_into_vit


class TIMMPrunedLoRA(nn.Module):
    """LoRA recovery wrapper for a pruned TIMMClassifier artifact."""

    def __init__(
        self,
        artifact_path,
        rank=4,
        lora_alpha=None,
        qkv_lora_components=None,
        lora_modules=None,
    ):
        super().__init__()

        # The pruning pipeline saves a dictionary artifact. Its "model" entry is
        # the already-pruned TIMMClassifier object, for example a ViT whose MLP
        # fc1/fc2 hidden dimension has been reduced. This wrapper does not build
        # a fresh timm model; it reuses that pruned model and attaches LoRA
        # adapters for recovery fine-tuning.
        self.artifact_path = artifact_path
        self.artifact = load_pruned_artifact(artifact_path)
        self.model = self.artifact["model"]

        self.lora_rank = rank
        self.lora_alpha = lora_alpha
        self.lora_modules = lora_modules
        self.qkv_lora_components = qkv_lora_components
        self.source_pruning_config = self.artifact.get("pruning_config")
        self.source_pruning_stats = self.artifact.get("pruning_stats")
        self.model_config = self.artifact.get("model_config")

        self._freeze_pruned_encoder()
        self.injected_module_names = self._inject_recovery_lora()
        self._unfreeze_classifier()

        trainable_params, total_params = count_parameters(self)
        print(f"[TIMMPrunedLoRA] artifact: {artifact_path}")
        print(f"[TIMMPrunedLoRA] injected modules ({len(self.injected_module_names)}):")
        for module_name in self.injected_module_names:
            print(f"  - model.encoder.{module_name}")
        print(
            f"[TIMMPrunedLoRA] trainable params: {trainable_params:,} / "
            f"{total_params:,} ({100.0 * trainable_params / total_params:.2f}%)"
        )

    def _freeze_pruned_encoder(self):
        # LoRA recovery treats the pruned dense encoder as the fixed base model.
        # TIMMClassifier.train() also checks freeze_encoder and keeps the encoder
        # in eval mode when the wrapper is trained.
        self.model.freeze_encoder = True
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = False

    def _inject_recovery_lora(self):
        # Reuse the same ViT LoRA injection helper used by TIMMLoRA. It reads
        # current Linear shapes, so pruned MLP layers such as 768->2457 and
        # 2457->768 work without special handling.
        injected_module_names = inject_lora_into_vit(
            self.model.encoder,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            qkv_lora_components=self.qkv_lora_components,
            lora_modules=self.lora_modules,
        )
        if not injected_module_names:
            raise ValueError("No LoRA modules were injected into the pruned encoder.")
        return injected_module_names

    def _unfreeze_classifier(self):
        # Keep the classifier trainable together with LoRA because it is small
        # and directly task-specific.
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

    def forward_features(self, x):
        return self.model.forward_features(x)

    def forward(self, x):
        return self.model(x)

    def export_config(self):
        # Trainer.save_checkpoint() stores this dictionary as model_config.
        # It records both the recovery setup and the source pruning metadata so
        # the checkpoint can be traced back to the pruned artifact it recovered.
        return {
            "model": "timm_pruned_lora",
            "source_pruned_artifact_path": self.artifact_path,
            "source_pruning_config": self.source_pruning_config,
            "source_pruning_stats": self.source_pruning_stats,
            "model_config": self.model_config,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_modules": self.lora_modules,
            "qkv_lora_components": self.qkv_lora_components,
        }

    def _build_merged_encoder(self):
        # Build a dense encoder copy for checkpoint export.
        #
        # During recovery training, the encoder contains LoRA wrapper modules
        # such as FusedQKVLoRA and LoRAWrappedLinear. For later evaluation or
        # another pruning pass, it is more convenient to have a plain dense
        # encoder where each LoRA update has been added into the corresponding
        # nn.Linear weight. The live training model is left unchanged.
        merged_encoder = copy.deepcopy(self.model.encoder)
        for block in merged_encoder.blocks:
            if isinstance(block.attn.qkv, FusedQKVLoRA):
                block.attn.qkv = block.attn.qkv.to_merged_linear()
            if isinstance(block.attn.proj, LoRAWrappedLinear):
                block.attn.proj = block.attn.proj.to_merged_linear()
            if isinstance(block.mlp.fc1, LoRAWrappedLinear):
                block.mlp.fc1 = block.mlp.fc1.to_merged_linear()
            if isinstance(block.mlp.fc2, LoRAWrappedLinear):
                block.mlp.fc2 = block.mlp.fc2.to_merged_linear()
        return merged_encoder

    def export_merged_state(self):
        # Trainer.save_checkpoint() stores this under checkpoint["merged_model"].
        # The format matches the merged_model saved by TIMMLoRA checkpoints:
        # separate dense state_dicts for encoder and classifier.
        merged_encoder = self._build_merged_encoder()
        return {
            "encoder": merged_encoder.state_dict(),
            "classifier": self.model.classifier.state_dict(),
        }
