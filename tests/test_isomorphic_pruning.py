import unittest
from pathlib import Path

import yaml

from prune import build_parser
from pruning.isomorphic.adapter import build_structure_summary


class IsomorphicPruningTest(unittest.TestCase):
    def test_cifar_comparison_config_matches_joint_calibration_inputs(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "timm_vit_cifar100_isomorphic_target_macs.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        self.assertEqual(config["calibration_dataset"], "cifar100")
        self.assertEqual(config["calibration_split"], "train")
        self.assertEqual(config["calibration_transform"], "default")
        self.assertEqual(config["calibration_batches"], "full")
        self.assertEqual(config["calibration_seed"], 42)

    def test_cli_exposes_independent_reference_ratios(self):
        args = build_parser().parse_args([
            "--importance", "isomorphic_taylor",
            "--isomorphic-pruning-ratio", "0.22",
            "--isomorphic-head-pruning-ratio", "0.22",
            "--isomorphic-head-dim-pruning-ratio", "0.10",
        ])
        self.assertEqual(args.importance, "isomorphic_taylor")
        self.assertAlmostEqual(args.isomorphic_pruning_ratio, 0.22)
        self.assertAlmostEqual(args.isomorphic_head_pruning_ratio, 0.22)
        self.assertAlmostEqual(args.isomorphic_head_dim_pruning_ratio, 0.10)

    def test_structure_summary_keeps_full_method_dimensions_separate(self):
        before = {
            "classifier_in_features": 768,
            "blocks": {
                "blocks.0": {
                    "embed_dim": 768,
                    "mlp_hidden_dim": 3072,
                    "num_heads": 12,
                    "head_dim": 64,
                    "attn_dim": 768,
                }
            },
        }
        after = {
            "classifier_in_features": 598,
            "blocks": {
                "blocks.0": {
                    "embed_dim": 598,
                    "mlp_hidden_dim": 2514,
                    "num_heads": 6,
                    "head_dim": 38,
                    "attn_dim": 228,
                }
            },
        }
        summary = build_structure_summary(before, after)
        block = summary["blocks"]["blocks.0"]
        self.assertEqual(summary["classifier"]["in_features_after"], 598)
        self.assertEqual(block["embed_dim"]["pruned"], 170)
        self.assertEqual(block["mlp_hidden_dim"]["pruned"], 558)
        self.assertEqual(block["num_heads"]["pruned"], 6)
        self.assertEqual(block["head_dim"]["pruned"], 26)


if __name__ == "__main__":
    unittest.main()
