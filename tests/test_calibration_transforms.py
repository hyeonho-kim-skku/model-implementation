import unittest

from torchvision import transforms

from datasets.build import get_transform
from pruning.calibration import _resolve_calibration_transform


class CalibrationTransformTest(unittest.TestCase):
    def test_isomorphic_eval_matches_documented_preprocessing(self):
        transform = get_transform("imagenet", "isomorphic_eval")
        resize, crop, to_tensor, normalize = transform.transforms

        self.assertIsInstance(resize, transforms.Resize)
        self.assertEqual(resize.size, 256)
        self.assertEqual(resize.interpolation, transforms.InterpolationMode.BICUBIC)
        self.assertIsInstance(crop, transforms.CenterCrop)
        self.assertEqual(crop.size, (224, 224))
        self.assertIsInstance(to_tensor, transforms.ToTensor)
        self.assertIsInstance(normalize, transforms.Normalize)
        self.assertEqual(normalize.mean, (0.485, 0.456, 0.406))
        self.assertEqual(normalize.std, (0.229, 0.224, 0.225))

    def test_default_calibration_transform_preserves_legacy_test_mode(self):
        mode, metadata = _resolve_calibration_transform("default")

        self.assertEqual(mode, "test")
        self.assertEqual(metadata["preset"], "default")
        self.assertEqual(metadata["interpolation"], "bilinear")


if __name__ == "__main__":
    unittest.main()
