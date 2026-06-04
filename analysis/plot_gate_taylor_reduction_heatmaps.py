"""Plot gate-Taylor reduction comparison heatmaps.

Each figure shows one dataset with three panels:
sum_abs, sum_square, and signed_damage. All panels in a dataset share the same
color scale so reduction differences are visually comparable.

Run:
  python analysis/plot_gate_taylor_reduction_heatmaps.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REDUCTIONS = ("sum_abs", "sum_square", "signed_damage")
DATASET_LABELS = {
    "cifar100": "CIFAR100",
    "cub200": "CUB200",
    "fgvc_aircraft": "FGVC-Aircraft",
    "stanford_cars": "Stanford Cars",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="figures/gate_taylor_sensitivity/gate_taylor_trials.csv",
        help="CSV produced by build_gate_taylor_trials_csv.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/gate_taylor_sensitivity/reduction_heatmaps",
        help="Directory for dataset-wise 3-panel heatmaps.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Saved figure DPI.")
    return parser


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def pivot_heatmap(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.pivot(index="layer_idx", columns="ratio", values="acc_drop")
    return table.sort_index().reindex(sorted(table.columns), axis=1)


def plot_dataset(dataset_frame: pd.DataFrame, dataset: str, output_dir: Path, dpi: int) -> Path:
    vmax = max(0.1, float(dataset_frame["acc_drop"].max()))
    vmin = min(0.0, float(dataset_frame["acc_drop"].min()))

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8), sharey=True)
    for ax, reduction in zip(axes, REDUCTIONS):
        reduction_frame = dataset_frame[dataset_frame["reduction"] == reduction]
        if reduction_frame.empty:
            raise ValueError(f"No rows for {dataset}/{reduction}.")
        table = pivot_heatmap(reduction_frame)
        sns.heatmap(
            table,
            ax=ax,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            linewidths=0.35,
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "Accuracy drop (%p)"},
        )
        ax.set_title(reduction)
        ax.set_xlabel("Pruning ratio")
        ax.set_ylabel("Transformer block" if ax is axes[0] else "")

    fig.suptitle(f"{dataset_label(dataset)} Gate Taylor MLP Sensitivity", y=1.02)
    fig.tight_layout()
    output_path = output_dir / f"{dataset}_gate_taylor_reduction_heatmaps.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(args: argparse.Namespace) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    frame = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(frame["dataset"].unique()):
        dataset_frame = frame[frame["dataset"] == dataset]
        output_path = plot_dataset(dataset_frame, dataset, output_dir, args.dpi)
        print(f"[Plot] saved {output_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
