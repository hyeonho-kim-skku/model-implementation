"""Custom importance criteria used by the structured pruning pipeline."""

from __future__ import annotations

import torch
import torch_pruning as tp


VALID_ACTIVATION_TAYLOR_REDUCTIONS = {"sum_abs", "abs_sum"}


class MLPActivationTaylorCollector:
    """Collect Taylor scores for MLP hidden activations during calibration.

    In timm ViT MLPs, fc2 receives the hidden neuron activation after
    fc1/GELU/drop/norm. Structurally we still prune fc1.out_features, because
    Torch-Pruning then propagates the same hidden channels to fc2.in_features.
    """

    def __init__(self, model, target_block_indices=None, reduction="sum_abs"):
        if reduction not in VALID_ACTIVATION_TAYLOR_REDUCTIONS:
            raise ValueError(
                "activation_taylor_reduction must be one of "
                f"{sorted(VALID_ACTIVATION_TAYLOR_REDUCTIONS)}, got {reduction!r}."
            )
        if not hasattr(model.encoder, "blocks"):
            raise ValueError("Activation Taylor pruning needs model.encoder.blocks.")

        self.reduction = reduction
        self.scores = {}
        self._pending_activations = []
        self._handles = []
        selected_block_indices = _normalize_target_block_indices(
            target_block_indices,
            num_blocks=len(model.encoder.blocks),
        )

        for block_idx, block in enumerate(model.encoder.blocks):
            if selected_block_indices is not None and block_idx not in selected_block_indices:
                continue
            fc1 = block.mlp.fc1
            fc2 = block.mlp.fc2
            self.scores[fc1] = torch.zeros(fc1.out_features, device=fc1.weight.device)
            self._handles.append(fc2.register_forward_hook(self._make_hook(fc1)))

    def _make_hook(self, fc1):
        def hook(_module, inputs, _output):
            activation = inputs[0]
            if not activation.requires_grad:
                return
            activation.retain_grad()
            self._pending_activations.append((fc1, activation))

        return hook

    def accumulate_batch(self):
        for fc1, activation in self._pending_activations:
            if activation.grad is None:
                continue
            # Channel is the last dim for ViT token tensors: [B, tokens, hidden].
            # sum_abs matches Torch-Pruning's current weight Taylor style;
            # abs_sum keeps the signed first-order delta until finalization.
            contribution = activation * activation.grad
            if self.reduction == "sum_abs":
                contribution = contribution.abs()
            channel_score = contribution.reshape(-1, contribution.shape[-1]).sum(dim=0)
            self.scores[fc1] = self.scores[fc1] + channel_score.detach()
        self._pending_activations.clear()

    def final_scores(self):
        if self.reduction == "abs_sum":
            return {module: score.abs() for module, score in self.scores.items()}
        return self.scores

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_activations.clear()


class MLPActivationTaylorImportance(tp.importance.MagnitudeImportance):
    """Torch-Pruning importance that reads precomputed MLP activation scores."""

    def __init__(self, scores, group_reduction="mean", normalizer="mean"):
        super().__init__(p=2, group_reduction=group_reduction, normalizer=normalizer)
        self.scores = scores

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []

        for i, (dep, idxs) in enumerate(group):
            layer = _dependency_module(dep)
            prune_fn = _dependency_handler(dep)
            if layer not in self.scores or prune_fn != tp.prune_linear_out_channels:
                continue

            idxs = sorted(idxs)
            root_idxs = group[i].root_idxs
            score = self.scores[layer].to(layer.weight.device)
            group_imp.append(score[idxs])
            group_idxs.append(root_idxs)

        if not group_imp:
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        return self._normalize(group_imp, self.normalizer)


def _normalize_target_block_indices(target_block_indices, num_blocks: int) -> tuple[int, ...] | None:
    """Local copy to keep activation collectors independent from pruning orchestration."""

    if target_block_indices is None:
        return None
    if isinstance(target_block_indices, str):
        if not target_block_indices.strip():
            return None
        indices = tuple(int(item.strip()) for item in target_block_indices.split(",") if item.strip())
    else:
        indices = tuple(int(item) for item in target_block_indices)

    invalid_indices = [idx for idx in indices if idx < 0 or idx >= num_blocks]
    if invalid_indices:
        raise ValueError(
            f"target_block_indices contains out-of-range indices {invalid_indices}; "
            f"valid range is 0..{num_blocks - 1}."
        )
    return tuple(dict.fromkeys(indices))


def _dependency_module(dep):
    """Handle Torch-Pruning 1.6 dependency attribute names in one place."""

    if hasattr(dep, "target") and hasattr(dep.target, "module"):
        return dep.target.module
    return dep.layer


def _dependency_handler(dep):
    if hasattr(dep, "handler"):
        return dep.handler
    return dep.pruning_fn
