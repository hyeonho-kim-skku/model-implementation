import copy
import io
import unittest
import warnings

import torch

from models.timm_classifier import TIMMClassifier
from pruning.structured import prune_model


class StructuredPruningRegressionTest(unittest.TestCase):
    """Characterize the structural pruning behavior before refactoring it."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.base_model = TIMMClassifier(
            backbone_name="deit_tiny_patch16_224.fb_in1k",
            num_classes=10,
            pretrained=False,
            img_size=32,
        ).eval()
        cls.example_inputs = torch.randn(1, 3, 32, 32)

    def _common_kwargs(self, model):
        return {
            "model": model,
            "model_config": {"img_size": 32},
            "source_info": {"type": "regression_fixture"},
            "output_dir": "/tmp/unused-structured-pruning-regression",
            "target_block_indices": "0",
            "use_existing_taylor_gradients": True,
            "existing_calibration_config": {"type": "fixed_scores"},
            "save_artifact": False,
            "verbose": False,
            "device": "cpu",
        }

    def test_mlp_gate_taylor_prunes_expected_hidden_channels(self):
        model = copy.deepcopy(self.base_model)
        block = model.encoder.blocks[0]
        original_fc1_weight = block.mlp.fc1.weight.detach().clone()
        original_fc2_weight = block.mlp.fc2.weight.detach().clone()
        scores = torch.arange(block.mlp.fc1.out_features, dtype=torch.float32)

        artifact = prune_model(
            **self._common_kwargs(model),
            importance="gate_taylor",
            pruning_ratio=0.25,
            pruning_modules="mlp",
            global_pruning=False,
            existing_gate_taylor_scores={block.mlp.fc1: scores},
        )

        pruned_model = artifact["model"]
        pruned_mlp = pruned_model.encoder.blocks[0].mlp
        self.assertEqual(pruned_mlp.fc1.out_features, 576)
        self.assertEqual(pruned_mlp.fc2.in_features, 576)
        self.assertTrue(torch.equal(pruned_mlp.fc1.weight, original_fc1_weight[192:]))
        self.assertTrue(torch.equal(pruned_mlp.fc2.weight, original_fc2_weight[:, 192:]))
        self.assertEqual(
            artifact["pruning_stats"]["target_pruning_summary"]["overall"]["mlp"][
                "pruned_hidden"
            ],
            192,
        )
        self.assertEqual(tuple(pruned_model(self.example_inputs).shape), (1, 10))

    def test_whole_head_pruning_preserves_attention_invariants(self):
        model = copy.deepcopy(self.base_model)
        attention = model.encoder.blocks[0].attn

        artifact = prune_model(
            **self._common_kwargs(model),
            importance="head_gate_taylor",
            pruning_ratio=0.2,
            pruning_modules="head",
            global_pruning=False,
            existing_head_gate_taylor_scores={
                attention.qkv: torch.tensor([0.0, 2.0, 1.0])
            },
        )

        pruned_model = artifact["model"]
        pruned_attention = pruned_model.encoder.blocks[0].attn
        self.assertEqual(artifact["pruning_stats"]["selected_attention_heads"], {0: [0]})
        self.assertEqual(pruned_attention.num_heads, 2)
        self.assertEqual(pruned_attention.head_dim, 64)
        self.assertEqual(pruned_attention.attn_dim, 128)
        self.assertEqual(pruned_attention.qkv.out_features, 384)
        self.assertEqual(pruned_attention.proj.in_features, 128)
        self.assertEqual(tuple(pruned_model(self.example_inputs).shape), (1, 10))

    def test_joint_pruning_and_artifact_round_trip(self):
        model = copy.deepcopy(self.base_model)
        block = model.encoder.blocks[0]

        artifact = prune_model(
            **self._common_kwargs(model),
            importance="joint_gate_taylor",
            pruning_modules="mlp,head",
            mlp_pruning_ratio=0.25,
            head_pruning_ratio=0.2,
            global_pruning=True,
            existing_gate_taylor_scores={
                block.mlp.fc1: torch.arange(
                    block.mlp.fc1.out_features,
                    dtype=torch.float32,
                )
            },
            existing_head_gate_taylor_scores={
                block.attn.qkv: torch.tensor([0.0, 2.0, 1.0])
            },
        )

        pruned_model = artifact["model"]
        self.assertEqual(pruned_model.encoder.blocks[0].mlp.fc1.out_features, 576)
        self.assertEqual(pruned_model.encoder.blocks[0].attn.num_heads, 2)
        self.assertEqual(artifact["pruning_stats"]["selected_attention_heads"], {0: [0]})
        self.assertEqual(artifact["pruning_stats"]["num_pruned_mlp_groups"], 1)
        self.assertEqual(artifact["pruning_stats"]["num_pruned_heads"], 1)

        expected_output = pruned_model(self.example_inputs)
        buffer = io.BytesIO()
        torch.save(artifact, buffer)
        buffer.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            restored_artifact = torch.load(
                buffer,
                map_location="cpu",
                weights_only=False,
            )
        restored_output = restored_artifact["model"](self.example_inputs)
        self.assertTrue(torch.equal(expected_output, restored_output))
        self.assertEqual(
            artifact["pruning_config"],
            restored_artifact["pruning_config"],
        )


if __name__ == "__main__":
    unittest.main()
