"""Create the ViT linear-probe and LoRA fine-tuning baseline table.

Run:
  python analysis/plot_adaptation_baseline_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch


BASELINES = {
    "CIFAR100": {
        "linear_probe": (
            "runs/timm_classifier_cifar100_supervised/"
            "0511-151239/best_cls_ckpt.pth"
        ),
        "lora": (
            "runs/timm_lora_cifar100_supervised/"
            "0511-224807/best_cls_ckpt.pth"
        ),
    },
    "CUB200": {
        "linear_probe": (
            "runs/timm_classifier_cub200_supervised/"
            "0511-163642/best_cls_ckpt.pth"
        ),
        "lora": (
            "runs/timm_lora_cub200_supervised/"
            "0512-022506/best_cls_ckpt.pth"
        ),
    },
    "FGVC-Aircraft": {
        "linear_probe": (
            "runs/timm_classifier_fgvc_aircraft_supervised/"
            "0523-174841/best_cls_ckpt.pth"
        ),
        "lora": (
            "runs/timm_lora_fgvc_aircraft_supervised/"
            "0523-182219/best_cls_ckpt.pth"
        ),
    },
    "Stanford Cars": {
        "linear_probe": (
            "runs/timm_classifier_stanford_cars_supervised/"
            "0523-202255/best_cls_ckpt.pth"
        ),
        "lora": (
            "runs/timm_lora_stanford_cars_supervised/"
            "0523-205731/best_cls_ckpt.pth"
        ),
    },
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="figures/baselines")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "accuracy": float(checkpoint["acc"]),
        "best_epoch": int(checkpoint["epoch"]),
    }


def build_results(repo_root):
    records = []
    for dataset, paths in BASELINES.items():
        linear_path = repo_root / paths["linear_probe"]
        lora_path = repo_root / paths["lora"]
        linear = load_checkpoint(linear_path)
        lora = load_checkpoint(lora_path)
        records.append(
            {
                "Dataset": dataset,
                "Linear Probe": linear["accuracy"],
                "LoRA Fine-tuning": lora["accuracy"],
                "linear_probe_best_epoch": linear["best_epoch"],
                "lora_best_epoch": lora["best_epoch"],
                "linear_probe_checkpoint": paths["linear_probe"],
                "lora_checkpoint": paths["lora"],
            }
        )
    return pd.DataFrame(records)


def plot_table(results, output_path, dpi):
    display = results[["Dataset", "Linear Probe", "LoRA Fine-tuning"]].copy()
    formatted = display.copy()
    for column in ("Linear Probe", "LoRA Fine-tuning"):
        formatted[column] = display[column].map(lambda value: f"{value:.2f}")

    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.axis("off")
    table = ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.28, 0.34],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.75)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#344054")
            cell.set_text_props(color="white", weight="bold")
            continue

        cell.set_facecolor("#F8FAFC")
        if col_idx == 0:
            cell.set_text_props(weight="bold", color="#1D2939", ha="left")
        elif col_idx == 2:
            cell.set_facecolor("#EAF2FF")
            cell.set_text_props(weight="bold", color="#101828")

    ax.set_title(
        "ViT Adaptation Baselines",
        fontsize=16,
        weight="bold",
        color="#101828",
        pad=12,
    )
    fig.text(
        0.5,
        0.055,
        "Test accuracy (%)  |  Linear Probe: frozen backbone  |  "
        "LoRA: backbone adapters + classifier",
        ha="center",
        fontsize=9,
        color="#475467",
    )
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.95))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_lora_only_table(results, output_path, dpi):
    display = results[["Dataset", "LoRA Fine-tuning"]].copy()
    formatted = display.copy()
    formatted["LoRA Fine-tuning"] = display["LoRA Fine-tuning"].map(
        lambda value: f"{value:.2f}"
    )

    fig, ax = plt.subplots(figsize=(5.5, 2.35))
    ax.axis("off")
    table = ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.53, 0.43],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.28)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#344054")
            cell.set_text_props(color="white", weight="bold")
            continue

        cell.set_facecolor("#F8FAFC")
        if col_idx == 0:
            cell.set_text_props(weight="bold", color="#1D2939", ha="left")
        else:
            cell.set_facecolor("#EAF2FF")
            cell.set_text_props(weight="bold", color="#101828")

    ax.set_title(
        "ViT LoRA Fine-tuning Baseline",
        fontsize=13,
        weight="bold",
        color="#101828",
        pad=4,
    )
    fig.text(
        0.5,
        0.025,
        "Test accuracy (%) | Rank-4 LoRA + classifier",
        ha="center",
        fontsize=8,
        color="#475467",
    )
    fig.tight_layout(rect=(0.015, 0.08, 0.985, 0.94))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(args):
    repo_root = Path(args.repo_root).resolve()
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = build_results(repo_root)
    csv_path = output_dir / "linear_probe_vs_lora.csv"
    image_path = output_dir / "linear_probe_vs_lora.png"
    lora_csv_path = output_dir / "lora_finetuning.csv"
    lora_image_path = output_dir / "lora_finetuning.png"
    results.to_csv(csv_path, index=False, float_format="%.4f")
    results[
        [
            "Dataset",
            "LoRA Fine-tuning",
            "lora_best_epoch",
            "lora_checkpoint",
        ]
    ].to_csv(lora_csv_path, index=False, float_format="%.4f")
    plot_table(results, image_path, args.dpi)
    plot_lora_only_table(results, lora_image_path, args.dpi)

    print(f"[AdaptationBaselines] saved {csv_path}")
    print(f"[AdaptationBaselines] saved {image_path}")
    print(f"[AdaptationBaselines] saved {lora_csv_path}")
    print(f"[AdaptationBaselines] saved {lora_image_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
