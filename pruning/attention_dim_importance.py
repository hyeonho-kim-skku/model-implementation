"""Gate-Taylor collectors for attention head-dimension pruning."""

from __future__ import annotations

import torch

from models.ragged_attention import RaggedFusedQKVAttention
from pruning.importance import VALID_GATE_TAYLOR_AGGREGATIONS, VALID_GATE_TAYLOR_REDUCTIONS


VALID_ATTENTION_DIM_GATE_LOCATIONS = {"proj_in", "qk_pair", "qkv_shared"}


class AttentionDimGateTaylorCollector:
    """Collect explicit gate Taylor scores for attention head dimensions."""

    def __init__(
        self,
        model,
        *,
        target,
        target_block_indices=None,
        reduction="sum_abs",
        gate_location=None,
        aggregation="samplewise",
    ):
        if reduction not in VALID_GATE_TAYLOR_REDUCTIONS:
            raise ValueError(
                "attention_dim reduction must be one of "
                f"{sorted(VALID_GATE_TAYLOR_REDUCTIONS)}, got {reduction!r}."
            )
        if aggregation not in VALID_GATE_TAYLOR_AGGREGATIONS:
            raise ValueError(
                "attention_dim aggregation must be one of "
                f"{sorted(VALID_GATE_TAYLOR_AGGREGATIONS)}, got {aggregation!r}."
            )
        if target not in {"v_proj", "qk_pair", "qkv_shared"}:
            raise ValueError(f"Unsupported attention_dim target: {target!r}.")
        if not hasattr(model, "encoder") or not hasattr(model.encoder, "blocks"):
            raise ValueError("Attention dim scoring needs model.encoder.blocks.")

        self.target = target
        self.reduction = reduction
        self.aggregation = aggregation
        self.gate_location = gate_location or _default_gate_location(target)
        if self.gate_location not in VALID_ATTENTION_DIM_GATE_LOCATIONS:
            raise ValueError(
                "attention_dim gate_location must be one of "
                f"{sorted(VALID_ATTENTION_DIM_GATE_LOCATIONS)}, got {self.gate_location!r}."
            )
        self.score_mode = "attention_dim_gate_grad"
        self.scores = {}
        self._pending_gates = []
        self._handles = []
        self._qk_hook_blocks = []
        self._qkv_hook_blocks = []
        selected_blocks = _normalize_target_block_indices(
            target_block_indices,
            num_blocks=len(model.encoder.blocks),
        )

        for block_idx, block in enumerate(model.encoder.blocks):
            if selected_blocks is not None and block_idx not in selected_blocks:
                continue
            attn = block.attn
            if not isinstance(attn, RaggedFusedQKVAttention):
                raise TypeError(
                    "AttentionDimGateTaylorCollector requires RaggedFusedQKVAttention."
                )
            self.scores[int(block_idx)] = torch.zeros(
                attn.num_heads,
                attn.original_head_dim,
                device=attn.qkv.weight.device,
            )
            if self.gate_location == "proj_in":
                self._handles.append(
                    attn.proj.register_forward_pre_hook(
                        self._make_proj_in_hook(block_idx, attn)
                    )
                )
            elif self.gate_location == "qk_pair":
                attn._qk_gate_hook = self._make_qk_gate_hook(block_idx, attn)
                self._qk_hook_blocks.append(attn)
            elif self.gate_location == "qkv_shared":
                attn._qkv_gate_hook = self._make_qkv_shared_gate_hook(block_idx, attn)
                self._qkv_hook_blocks.append(attn)

    def _make_proj_in_hook(self, block_idx, attn):
        def hook(_module, inputs):
            activation = inputs[0]
            gate = torch.ones_like(activation, requires_grad=True)
            gated_activation = activation * gate
            if gated_activation.requires_grad:
                self._pending_gates.append(("proj_in", int(block_idx), attn, gate))
            return (gated_activation, *inputs[1:])

        return hook

    def _make_qk_gate_hook(self, block_idx, attn):
        def hook(_attn, head_idx, q, k):
            gate = torch.ones_like(q, requires_grad=True)
            q = q * gate
            k = k * gate
            if q.requires_grad or k.requires_grad:
                self._pending_gates.append(
                    ("qk_pair", int(block_idx), attn, int(head_idx), gate)
                )
            return q, k

        return hook

    def _make_qkv_shared_gate_hook(self, block_idx, attn):
        def hook(_attn, head_idx, q, k, v):
            head_idx = int(head_idx)
            if list(attn.qk_dim_indices[head_idx]) != list(attn.v_dim_indices[head_idx]):
                raise ValueError(
                    "qkv_shared joint scoring requires matching active Q/K and V "
                    f"dim indices in block {block_idx}, head {head_idx}."
                )
            if q.shape[-1] != v.shape[-1]:
                raise ValueError(
                    "qkv_shared joint scoring requires matching Q/K and V widths "
                    f"in block {block_idx}, head {head_idx}: "
                    f"{q.shape[-1]} != {v.shape[-1]}."
                )
            gate = torch.ones_like(q, requires_grad=True)
            q = q * gate
            k = k * gate
            v = v * gate
            if q.requires_grad or k.requires_grad or v.requires_grad:
                self._pending_gates.append(
                    ("qkv_shared", int(block_idx), attn, head_idx, gate)
                )
            return q, k, v

        return hook

    def accumulate_batch(self):
        for item in self._pending_gates:
            if item[0] == "proj_in":
                _kind, block_idx, attn, gate = item
                if gate.grad is None:
                    continue
                contribution = gate * gate.grad
                self._accumulate_proj_in(block_idx, attn, contribution)
                gate.grad = None
            elif item[0] == "qk_pair":
                _kind, block_idx, attn, head_idx, gate = item
                if gate.grad is None:
                    continue
                contribution = gate * gate.grad
                self._accumulate_qk(block_idx, attn, head_idx, contribution)
                gate.grad = None
            elif item[0] == "qkv_shared":
                _kind, block_idx, attn, head_idx, gate = item
                if gate.grad is None:
                    continue
                contribution = gate * gate.grad
                self._accumulate_qkv_shared(block_idx, attn, head_idx, contribution)
                gate.grad = None
        self._pending_gates.clear()

    def _accumulate_proj_in(self, block_idx, attn, contribution):
        # contribution: [B, tokens, sum(v_head_dims)]
        cursor = 0
        for head_idx, active_dims in enumerate(attn.v_dim_indices):
            width = len(active_dims)
            head_contribution = contribution[..., cursor:cursor + width]
            score = self._reduce_contribution(head_contribution)
            for local_pos, dim_idx in enumerate(active_dims):
                self.scores[block_idx][head_idx, int(dim_idx)] += score[local_pos].detach()
            cursor += width

    def _accumulate_qk(self, block_idx, attn, head_idx, contribution):
        # contribution: [B, 1, tokens, qk_head_dim] from a shared Q/K gate.
        score = self._reduce_contribution(contribution.squeeze(1))
        for local_pos, dim_idx in enumerate(attn.qk_dim_indices[int(head_idx)]):
            self.scores[block_idx][int(head_idx), int(dim_idx)] += score[local_pos].detach()

    def _accumulate_qkv_shared(self, block_idx, attn, head_idx, contribution):
        # contribution: [B, 1, tokens, dim] from a shared Q/K/V gate.
        score = self._reduce_contribution(contribution.squeeze(1))
        for local_pos, dim_idx in enumerate(attn.qk_dim_indices[int(head_idx)]):
            self.scores[block_idx][int(head_idx), int(dim_idx)] += score[local_pos].detach()

    def _reduce_contribution(self, contribution):
        if self.aggregation == "elementwise":
            return self._reduce_elementwise(contribution)
        if self.aggregation == "samplewise":
            return self._reduce_samplewise(contribution)
        raise ValueError(
            "attention_dim_gate_taylor v1 supports elementwise/samplewise "
            f"aggregation, got {self.aggregation!r}."
        )

    def _reduce_elementwise(self, contribution):
        if self.reduction == "signed_damage":
            return -contribution.reshape(-1, contribution.shape[-1]).sum(dim=0)
        if self.reduction == "sum_abs":
            return contribution.abs().reshape(-1, contribution.shape[-1]).sum(dim=0)
        if self.reduction == "sum_square":
            return contribution.square().reshape(-1, contribution.shape[-1]).sum(dim=0)
        raise ValueError(f"Unsupported reduction: {self.reduction!r}.")

    def _reduce_samplewise(self, contribution):
        per_sample = contribution.sum(dim=1)
        if self.reduction == "signed_damage":
            return -per_sample.sum(dim=0)
        if self.reduction == "sum_abs":
            return per_sample.abs().sum(dim=0)
        if self.reduction == "sum_square":
            return per_sample.square().sum(dim=0)
        raise ValueError(f"Unsupported reduction: {self.reduction!r}.")

    def final_scores(self):
        return {
            int(block_idx): score.detach().cpu().clone()
            for block_idx, score in self.scores.items()
        }

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for attn in self._qk_hook_blocks:
            attn._qk_gate_hook = None
        self._qk_hook_blocks.clear()
        for attn in self._qkv_hook_blocks:
            attn._qkv_gate_hook = None
        self._qkv_hook_blocks.clear()
        for item in self._pending_gates:
            gate = item[-1]
            if hasattr(gate, "grad"):
                gate.grad = None
        self._pending_gates.clear()


def _default_gate_location(target):
    if target == "qk_pair":
        return "qk_pair"
    if target == "qkv_shared":
        return "qkv_shared"
    return "proj_in"


def _normalize_target_block_indices(target_block_indices, num_blocks):
    if target_block_indices is None:
        return None
    if isinstance(target_block_indices, str):
        if not target_block_indices.strip():
            return None
        indices = tuple(int(item.strip()) for item in target_block_indices.split(",") if item.strip())
    else:
        indices = tuple(int(item) for item in target_block_indices)
    invalid = [idx for idx in indices if idx < 0 or idx >= int(num_blocks)]
    if invalid:
        raise ValueError(f"target_block_indices out of range: {invalid}.")
    return tuple(dict.fromkeys(indices))
