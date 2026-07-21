import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from models.timm_classifier import TIMMClassifier
from pruning.finetune import (
    build_finetune_train_transform,
    build_mixup,
    load_resume_checkpoint,
    save_checkpoint,
    validate_resume_config,
)


class PrunedFineTuneTests(unittest.TestCase):
    def test_train_transform_and_mixup(self):
        model = TIMMClassifier(
            backbone_name="deit_small_patch16_224.fb_in1k",
            num_classes=1000,
            pretrained=False,
        )
        transform, data_config = build_finetune_train_transform(model)
        self.assertEqual(data_config["input_size"], (3, 224, 224))
        self.assertIn("bicubic", repr(transform).lower())
        self.assertIn("randaugment", repr(transform).lower())
        mixup = build_mixup({"num_classes": 1000})
        images, targets = mixup(torch.randn(2, 3, 224, 224), torch.tensor([1, 2]))
        self.assertEqual(images.shape, (2, 3, 224, 224))
        self.assertEqual(targets.shape, (2, 1000))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "artifact.pth"
            artifact_model = nn.Linear(4, 2)
            torch.save({"model": artifact_model}, artifact_path)
            model = nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
            checkpoint_path = Path(directory) / "latest.pth"
            save_checkpoint(
                checkpoint_path,
                artifact_path=artifact_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                best_top1=12.5,
                config={"epochs": 3},
                history=[{"epoch": 1, "top1": 12.5}],
                transform_metadata={},
            )
            checkpoint, _, resumed_model = load_resume_checkpoint(checkpoint_path, "cpu")
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["best_top1"], 12.5)
            for original, restored in zip(model.parameters(), resumed_model.parameters()):
                self.assertTrue(torch.equal(original, restored))
            validate_resume_config({"epochs": 3}, checkpoint)
            with self.assertRaisesRegex(ValueError, "different epochs"):
                validate_resume_config({"epochs": 4}, checkpoint)


if __name__ == "__main__":
    unittest.main()
