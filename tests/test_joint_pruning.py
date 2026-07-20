from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from prune import build_parser
from pruning.calibration import compute_taylor_gradients as moved_compute_taylor_gradients
from pruning.head_pruning import select_attention_heads_by_score
from pruning.joint import (
    JOINT_GATE_TAYLOR_HEAD_CONFIG as moved_head_config,
    JOINT_GATE_TAYLOR_MLP_CONFIG as moved_mlp_config,
)
from pruning.structured import (
    JOINT_GATE_TAYLOR_HEAD_CONFIG,
    JOINT_GATE_TAYLOR_MLP_CONFIG,
    _validate_joint_gate_taylor_scores,
    compute_taylor_gradients,
)


def _score_validation_model(num_blocks=12, hidden_dim=1536, num_heads=6):
    blocks = []
    for _ in range(num_blocks):
        blocks.append(
            SimpleNamespace(
                mlp=SimpleNamespace(fc1=nn.Linear(384, hidden_dim)),
                attn=SimpleNamespace(
                    qkv=nn.Linear(384, 3 * 384),
                    proj=nn.Linear(384, 384),
                    num_heads=num_heads,
                ),
            )
        )
    return SimpleNamespace(encoder=SimpleNamespace(blocks=blocks))


class JointPruningTest(unittest.TestCase):
    def test_compatibility_reexports(self):
        self.assertIs(compute_taylor_gradients, moved_compute_taylor_gradients)
        self.assertIs(JOINT_GATE_TAYLOR_MLP_CONFIG, moved_mlp_config)
        self.assertIs(JOINT_GATE_TAYLOR_HEAD_CONFIG, moved_head_config)

    def test_joint_defaults_and_cli_ratios(self):
        self.assertEqual(
            JOINT_GATE_TAYLOR_MLP_CONFIG,
            {
                "gate_location": "fc2_in",
                "reduction": "sum_square",
                "aggregation": "samplewise",
            },
        )
        self.assertEqual(
            JOINT_GATE_TAYLOR_HEAD_CONFIG,
            {
                "gate_location": "proj_in",
                "reduction": "sum_square",
                "aggregation": "samplewise",
            },
        )

        args = build_parser().parse_args(["--importance", "joint_gate_taylor"])
        self.assertAlmostEqual(args.mlp_pruning_ratio, 0.4)
        self.assertAlmostEqual(args.head_pruning_ratio, 0.4)

    def test_deit_small_joint_score_shapes(self):
        model = _score_validation_model()
        mlp_scores = {
            block.mlp.fc1: torch.ones(1536)
            for block in model.encoder.blocks
        }
        head_scores = {
            block.attn.qkv: torch.ones(6)
            for block in model.encoder.blocks
        }

        _validate_joint_gate_taylor_scores(model, mlp_scores, head_scores)

        mlp_scores[model.encoder.blocks[0].mlp.fc1] = torch.ones(1535)
        with self.assertRaisesRegex(ValueError, "MLP score shape"):
            _validate_joint_gate_taylor_scores(model, mlp_scores, head_scores)

    def test_global_head_budget_preserves_one_head_per_block(self):
        scores = {
            block_idx: torch.arange(6, dtype=torch.float32) + block_idx / 100
            for block_idx in range(12)
        }
        selected = select_attention_heads_by_score(
            scores,
            pruning_ratio=0.4,
            global_pruning=True,
            min_heads_per_block=1,
        )

        self.assertEqual(sum(len(head_ids) for head_ids in selected.values()), 29)
        self.assertTrue(
            all(len(selected.get(block_idx, ())) <= 5 for block_idx in scores)
        )


if __name__ == "__main__":
    unittest.main()
