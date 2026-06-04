"""Custom importance criteria used by the structured pruning pipeline."""

from __future__ import annotations

import torch
import torch_pruning as tp


VALID_ACTIVATION_TAYLOR_REDUCTIONS = {"sum_abs", "abs_sum"}
VALID_GATE_TAYLOR_REDUCTIONS = {"signed_damage", "sum_abs", "sum_square"}
VALID_GATE_TAYLOR_LOCATIONS = {"fc1_out", "fc2_in"}


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


class MLPGateTaylorCollector:
    """Collect Taylor scores from explicit element-wise gates in ViT MLPs.

    Gates can be inserted immediately after fc1 and before GELU (fc1_out), or
    on the activation entering fc2 after GELU/dropout/norm (fc2_in). A fresh
    output-shaped gate is created for each forward pass, so every
    image/token/channel activation has its own deletion Taylor contribution.
    Scores are aggregated by hidden channel and keyed by fc1 so the existing
    pruning importance can rank the structural fc1.out_features roots.
    """

    def __init__(
        self,
        model,
        target_block_indices=None,
        reduction="sum_abs",
        gate_location="fc1_out",
    ):
        if reduction not in VALID_GATE_TAYLOR_REDUCTIONS:
            raise ValueError(
                "gate_taylor reduction must be one of "
                f"{sorted(VALID_GATE_TAYLOR_REDUCTIONS)}, got {reduction!r}."
            )
        if gate_location not in VALID_GATE_TAYLOR_LOCATIONS:
            raise ValueError(
                "gate_taylor_location must be one of "
                f"{sorted(VALID_GATE_TAYLOR_LOCATIONS)}, got {gate_location!r}."
            )
        if not hasattr(model.encoder, "blocks"):
            raise ValueError("Gate Taylor pruning needs model.encoder.blocks.")

        self.reduction = reduction
        self.gate_location = gate_location
        self.score_mode = "elementwise_gate_grad"
        self.scores = {}
        self._pending_gates = []
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
            if gate_location == "fc1_out":
                self._register_fc1_out_gate(fc1)
            elif gate_location == "fc2_in":
                self._register_fc2_in_gate(fc1, fc2)

    def _register_fc1_out_gate(self, fc1):
        self._handles.append(fc1.register_forward_hook(self._make_fc1_out_hook()))

    def _make_fc1_out_hook(self):
        def hook(_module, _inputs, output):
            gate = torch.ones_like(output, requires_grad=True)
            gated_output = output * gate
            if gated_output.requires_grad:
                self._pending_gates.append((_module, gate))
            return gated_output

        return hook

    def _register_fc2_in_gate(self, fc1, fc2):
        self._handles.append(fc2.register_forward_pre_hook(self._make_fc2_in_hook(fc1)))

    def _make_fc2_in_hook(self, fc1):
        def hook(_module, inputs):
            activation = inputs[0]
            gate = torch.ones_like(activation, requires_grad=True)
            gated_activation = activation * gate
            if gated_activation.requires_grad:
                self._pending_gates.append((fc1, gate))
            return (gated_activation, *inputs[1:])

        return hook

    def accumulate_batch(self):
        for fc1, gate in self._pending_gates:
            if gate.grad is None:
                continue
            # Channel is the last dim for ViT token tensors: [B, tokens, hidden].
            contribution = gate * gate.grad
            if self.reduction == "signed_damage":
                # Removing a gate changes it by delta_g=-1, so predicted loss
                # change is -sum(gate * dL/dgate).
                channel_score = -contribution.reshape(-1, contribution.shape[-1]).sum(dim=0)
            elif self.reduction == "sum_abs":
                channel_score = contribution.abs().reshape(-1, contribution.shape[-1]).sum(dim=0)
            elif self.reduction == "sum_square":
                channel_score = contribution.square().reshape(-1, contribution.shape[-1]).sum(dim=0)
            else:
                raise ValueError(f"Unsupported gate_taylor reduction: {self.reduction!r}.")
            self.scores[fc1] = self.scores[fc1] + channel_score.detach()
            gate.grad = None
        self._pending_gates.clear()

    def final_scores(self):
        return self.scores

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for _fc1, gate in self._pending_gates:
            gate.grad = None
        self._pending_gates.clear()


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
