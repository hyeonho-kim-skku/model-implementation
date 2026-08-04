import tempfile
import unittest
from pathlib import Path

from analysis.summarize_layerwise_vpt_recovery import discover_runs, parse_log


class LayerwiseVPTSummaryTest(unittest.TestCase):
    def test_discovers_multiple_datasets_from_shared_log_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "cub200__uniform5.log").touch()
            (directory / "cub200__head_proportional.log").touch()
            (directory / "stanford_cars__uniform5.log").touch()
            runs = discover_runs(
                directory,
                datasets="cub200,stanford_cars",
            )

        self.assertEqual(
            [(dataset, path.name) for dataset, path in runs],
            [
                ("cub200", "cub200__head_proportional.log"),
                ("cub200", "cub200__uniform5.log"),
                ("stanford_cars", "stanford_cars__uniform5.log"),
            ],
        )

    def test_parses_allocation_metrics_without_scalar_token_assumption(self):
        log_text = """
[TIMMPrunedVPT] prompt mode: deep
[TIMMPrunedVPT] prompt tokens per layer: [11, 5, 0]
[TIMMPrunedVPT] total prompt tokens: 16
[TIMMPrunedVPT] allocation label: head-proportional-1to1
[TIMMPrunedVPT] trainable params: 123,456 / 1,000,000 (12.0%)
[ModelProfile] MACs: 9,876,543
[Epoch 0] - Test Loss: 1.0, Test Acc: 70.00%, Best Acc: 70.00
[Epoch 1] - Test Loss: 0.8, Test Acc: 72.00%, Best Acc: 72.00
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(log_text)
            metrics = parse_log(path)

        self.assertEqual(metrics["allocation_label"], "head-proportional-1to1")
        self.assertEqual(metrics["prompt_tokens_per_layer"], "11,5,0")
        self.assertEqual(metrics["total_prompt_tokens"], 16)
        self.assertEqual(metrics["trainable_params"], 123456)
        self.assertEqual(metrics["macs"], 9876543)
        self.assertEqual(metrics["best_acc"], 72.0)
        self.assertEqual(metrics["final_acc"], 72.0)


if __name__ == "__main__":
    unittest.main()
