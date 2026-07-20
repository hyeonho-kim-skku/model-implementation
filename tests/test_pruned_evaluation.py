import unittest

from models.timm_classifier import TIMMClassifier
from pruning.eval import build_pruned_evaluation_transform


class PrunedEvaluationTransformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TIMMClassifier(
            backbone_name="deit_small_patch16_224.fb_in1k",
            num_classes=1000,
            pretrained=False,
            classifier_init="random",
        )

    def test_default_transform_preserves_legacy_loader_path(self):
        transform, metadata = build_pruned_evaluation_transform(
            self.model,
            evaluation_transform="default",
        )

        self.assertIsNone(transform)
        self.assertEqual(metadata, {"preset": "default", "mode": "test"})

    def test_timm_pretrained_transform_uses_encoder_data_config(self):
        transform, metadata = build_pruned_evaluation_transform(
            self.model,
            evaluation_transform="timm_pretrained",
        )

        self.assertIsNotNone(transform)
        self.assertEqual(metadata["preset"], "timm_pretrained")
        self.assertEqual(metadata["data_config"]["input_size"], (3, 224, 224))
        self.assertEqual(metadata["data_config"]["interpolation"], "bicubic")


if __name__ == "__main__":
    unittest.main()
