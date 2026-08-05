import tempfile
import unittest
from pathlib import Path

from analysis.summarize_prompt_recovery import discover_runs, parse_log


class PromptRecoverySummaryTest(unittest.TestCase):
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
        self.assertEqual(metrics["vpt_prompt_tokens_per_layer"], "11,5,0")
        self.assertEqual(metrics["total_vpt_prompt_tokens"], 16)
        self.assertEqual(metrics["trainable_params"], 123456)
        self.assertEqual(metrics["macs"], 9876543)
        self.assertEqual(metrics["best_acc"], 72.0)
        self.assertEqual(metrics["final_acc"], 72.0)

    def test_parses_composable_vpt_and_kv_prompt_metrics(self):
        log_text = """
[TIMMPrunedPrompt] components: vpt,kv
[TIMMPrunedPrompt] VPT mode: deep
[TIMMPrunedPrompt] VPT tokens per layer: [5, 5, 5]
[TIMMPrunedPrompt] KV tokens per layer: [8, 8, 8]
[TIMMPrunedPrompt] KV prompt sharing: separate
[TIMMPrunedPrompt] total VPT tokens: 15
[TIMMPrunedPrompt] total KV tokens: 24
[TIMMPrunedPrompt] VPT prompt params: 120
[TIMMPrunedPrompt] KV prompt params: 128
[TIMMPrunedPrompt] LoRA enabled: true
[TIMMPrunedPrompt] LoRA rank: 4
[TIMMPrunedPrompt] LoRA params: 456
[TIMMPrunedPrompt] staged recovery: true
[TIMMPrunedPrompt] initial recovery checkpoint: runs/lora/best_cls_ckpt.pth
[TIMMPrunedPrompt] allocation label: vpt5-kv8
[TIMMPrunedPrompt] trainable params: 1,234 / 10,000 (12.34%)
[ModelProfile] MACs: 9,999
[Epoch 0] - Test Loss: 1.0, Test Acc: 72.00%, Best Acc: 72.00
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(log_text)
            metrics = parse_log(path)
            path.write_text(
                log_text.replace(
                    "[TIMMPrunedPrompt] KV prompt sharing: separate\n",
                    "",
                )
            )
            legacy_metrics = parse_log(path)

        self.assertEqual(metrics["prompt_components"], "vpt,kv")
        self.assertEqual(metrics["vpt_prompt_tokens_per_layer"], "5,5,5")
        self.assertEqual(metrics["kv_prompt_tokens_per_layer"], "8,8,8")
        self.assertEqual(metrics["kv_prompt_sharing"], "separate")
        self.assertEqual(metrics["total_vpt_prompt_tokens"], 15)
        self.assertEqual(metrics["total_kv_prompt_tokens"], 24)
        self.assertEqual(metrics["vpt_prompt_params"], 120)
        self.assertEqual(metrics["kv_prompt_params"], 128)
        self.assertTrue(metrics["lora_enabled"])
        self.assertEqual(metrics["lora_rank"], 4)
        self.assertEqual(metrics["lora_params"], 456)
        self.assertTrue(metrics["staged_recovery"])
        self.assertEqual(
            metrics["initial_recovery_checkpoint"], "runs/lora/best_cls_ckpt.pth"
        )
        self.assertEqual(legacy_metrics["kv_prompt_sharing"], "shared")

    def test_old_prompt_logs_default_to_no_lora(self):
        log_text = """
[TIMMPrunedVPT] prompt mode: deep
[TIMMPrunedVPT] prompt tokens per layer: [5, 5]
[TIMMPrunedVPT] total prompt tokens: 10
[TIMMPrunedVPT] trainable params: 100 / 1,000 (10.0%)
[Epoch 0] - Test Loss: 1.0, Test Acc: 70.00%, Best Acc: 70.00
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.log"
            path.write_text(log_text)
            metrics = parse_log(path)

        self.assertFalse(metrics["lora_enabled"])
        self.assertEqual(metrics["lora_rank"], "")
        self.assertEqual(metrics["lora_params"], 0)
        self.assertFalse(metrics["staged_recovery"])
        self.assertEqual(metrics["initial_recovery_checkpoint"], "")


if __name__ == "__main__":
    unittest.main()
