import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from models import load_model
from models.layerwise_prompts import (
    LayerwisePromptTokens,
    normalize_prompt_tokens_per_layer,
)
from models.timm_pruned_vpt import TIMMPrunedVPT


class FakePatchEmbed(nn.Module):
    img_size = (4, 4)

    def __init__(self, embedding_dim):
        super().__init__()
        self.projection = nn.Conv2d(3, embedding_dim, kernel_size=2, stride=2)

    def forward(self, images):
        return self.projection(images).flatten(2).transpose(1, 2)


class RecordingBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.last_sequence_length = None

    def forward(self, tokens):
        self.last_sequence_length = tokens.shape[1]
        return tokens + self.projection(tokens)


class FakeEncoder(nn.Module):
    def __init__(self, embedding_dim=8, num_blocks=3):
        super().__init__()
        self.embed_dim = embedding_dim
        self.num_features = embedding_dim
        self.num_prefix_tokens = 1
        self.patch_embed = FakePatchEmbed(embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList(
            [RecordingBlock(embedding_dim) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def _pos_embed(self, tokens):
        return torch.cat((self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)

    def forward_head(self, tokens, pre_logits=False):
        return tokens[:, 0]


class FakePrunedClassifier(nn.Module):
    def __init__(self, embedding_dim=8, num_blocks=3, num_classes=5):
        super().__init__()
        self.encoder = FakeEncoder(embedding_dim, num_blocks)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.freeze_encoder = False


def fake_artifact(num_blocks=3):
    return {
        "model": FakePrunedClassifier(num_blocks=num_blocks),
        "model_config": {"model": "fake"},
        "pruning_config": {"pruning_ratio": 0.4},
        "pruning_stats": {"pruned_heads_per_layer": [2, 0, 1]},
    }


class LayerwisePromptTokensTest(unittest.TestCase):
    def test_normalizes_string_and_integer_list(self):
        self.assertEqual(
            normalize_prompt_tokens_per_layer("2,0,1", 3),
            (2, 0, 1),
        )
        self.assertEqual(
            normalize_prompt_tokens_per_layer([2, 0, 1], 3),
            (2, 0, 1),
        )

    def test_rejects_invalid_schedules(self):
        invalid_values = ([1, 2], [1, -1, 2], [1, 1.5, 2], "1,x,2")
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_prompt_tokens_per_layer(value, 3)

    def test_zero_token_layers_and_parameter_count(self):
        prompts = LayerwisePromptTokens([2, 0, 1], embedding_dim=8)
        self.assertEqual(prompts.total_tokens, 3)
        self.assertEqual(tuple(prompts.prompt_for_layer(0).shape), (1, 2, 8))
        self.assertEqual(tuple(prompts.prompt_for_layer(1).shape), (1, 0, 8))
        self.assertEqual(sum(parameter.numel() for parameter in prompts.parameters()), 24)


class TIMMPrunedVPTTest(unittest.TestCase):
    def build_model(self, num_blocks=3, **kwargs):
        artifact = fake_artifact(num_blocks=num_blocks)
        with patch(
            "models.timm_pruned_vpt.load_pruned_artifact",
            return_value=artifact,
        ):
            return TIMMPrunedVPT(
                artifact_path="fake.pth",
                num_classes=5,
                **kwargs,
            )

    def test_legacy_shallow_and_deep_shapes_and_insertion(self):
        images = torch.randn(2, 3, 4, 4)
        shallow = self.build_model(prompt_mode="shallow", num_prompt_tokens=1)
        deep = self.build_model(prompt_mode="deep", num_prompt_tokens=1)

        self.assertEqual(tuple(shallow.prompt_embeddings.shape), (1, 1, 8))
        self.assertEqual(tuple(deep.deep_prompt_embeddings.shape), (3, 1, 8))
        self.assertEqual(tuple(shallow(images).shape), (2, 5))
        self.assertEqual(tuple(deep(images).shape), (2, 5))
        self.assertEqual(
            [block.last_sequence_length for block in shallow.encoder.blocks],
            [6, 6, 6],
        )
        self.assertEqual(
            [block.last_sequence_length for block in deep.encoder.blocks],
            [6, 6, 6],
        )

    def test_uniform_scalar_matches_explicit_uniform_parameter_count(self):
        scalar = self.build_model(
            num_blocks=12,
            prompt_mode="deep",
            num_prompt_tokens=5,
        )
        explicit = self.build_model(
            num_blocks=12,
            prompt_mode="deep",
            num_prompt_tokens=1,
            prompt_tokens_per_layer=[5] * 12,
        )
        scalar_count = scalar.deep_prompt_embeddings.numel()
        explicit_count = sum(
            parameter.numel() for parameter in explicit.layerwise_prompts.parameters()
        )
        self.assertEqual(scalar_count, explicit_count)
        self.assertEqual(tuple(explicit(torch.randn(2, 3, 4, 4)).shape), (2, 5))

    def test_explicit_schedule_takes_priority_over_scalar(self):
        model = self.build_model(
            prompt_mode="deep",
            num_prompt_tokens=0,
            prompt_tokens_per_layer=[2, 0, 1],
        )
        self.assertEqual(model.prompt_tokens_per_layer, (2, 0, 1))
        self.assertEqual(model.total_prompt_tokens, 3)

    def test_explicit_schedule_controls_each_layer_and_total(self):
        model = self.build_model(
            prompt_mode="deep",
            prompt_tokens_per_layer="2,0,1",
        )
        output = model(torch.randn(2, 3, 4, 4))
        self.assertEqual(tuple(output.shape), (2, 5))
        self.assertEqual(model.total_prompt_tokens, 3)
        self.assertEqual(
            [block.last_sequence_length for block in model.encoder.blocks],
            [7, 5, 6],
        )

    def test_cifar_schedule_has_expected_parameter_shapes(self):
        schedule = [11, 5, 4, 7, 5, 2, 2, 1, 3, 4, 7, 7]
        artifact = fake_artifact(num_blocks=12)
        artifact["model"] = FakePrunedClassifier(embedding_dim=768, num_blocks=12)
        with patch(
            "models.timm_pruned_vpt.load_pruned_artifact",
            return_value=artifact,
        ):
            model = TIMMPrunedVPT(
                artifact_path="fake.pth",
                prompt_mode="deep",
                prompt_tokens_per_layer=schedule,
                num_classes=5,
            )
        self.assertEqual(model.total_prompt_tokens, 58)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.layerwise_prompts.parameters()),
            58 * 768,
        )
        self.assertEqual(
            [parameter.shape[1] for parameter in model.layerwise_prompts.prompts],
            schedule,
        )

    def test_rejects_shallow_layerwise_and_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "only in deep mode"):
            self.build_model(
                prompt_mode="shallow",
                prompt_tokens_per_layer=[1, 1, 1],
            )
        with self.assertRaisesRegex(ValueError, "length must match"):
            self.build_model(
                prompt_mode="deep",
                prompt_tokens_per_layer=[1, 1],
            )

    def test_only_prompt_and_classifier_receive_gradients(self):
        model = self.build_model(
            prompt_mode="deep",
            prompt_tokens_per_layer=[2, 0, 1],
        )
        model.train()
        model(torch.randn(2, 3, 4, 4)).sum().backward()

        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.model.classifier.parameters()))
        nonempty_prompts = [
            parameter
            for parameter in model.layerwise_prompts.parameters()
            if parameter.numel() > 0
        ]
        self.assertTrue(all(parameter.grad is not None for parameter in nonempty_prompts))

    def test_checkpoint_config_and_state_dict_round_trip(self):
        original = self.build_model(
            prompt_mode="deep",
            prompt_tokens_per_layer=[2, 0, 1],
            prompt_allocation_label="test-allocation",
        )
        config = original.export_config()
        self.assertEqual(config["prompt_tokens_per_layer"], [2, 0, 1])
        self.assertEqual(config["total_prompt_tokens"], 3)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pth"
            torch.save(
                {"model_config": config, "model": original.state_dict()},
                checkpoint_path,
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        with patch(
            "models.timm_pruned_vpt.load_pruned_artifact",
            return_value=fake_artifact(),
        ):
            reconstructed = load_model(**checkpoint["model_config"])
        reconstructed.load_state_dict(checkpoint["model"])
        self.assertEqual(reconstructed.prompt_tokens_per_layer, (2, 0, 1))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, reconstructed.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
