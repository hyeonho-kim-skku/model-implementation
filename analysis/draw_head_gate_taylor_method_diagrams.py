"""Draw slide-ready diagrams for the head-gate Taylor pruning method.

Outputs:
  figures/head_gate_taylor_method/gate_location.{png,svg}
  figures/head_gate_taylor_method/samplewise_aggregation.{png,svg}
  figures/head_gate_taylor_method/head_block_contribution.{png,svg}

Run:
  python analysis/draw_head_gate_taylor_method_diagrams.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patches import Rectangle


BLUE = "#2F6BFF"
GREEN = "#2E9D68"
ORANGE = "#E68A2E"
PURPLE = "#7A5CFF"
GRAY = "#5E6470"
LIGHT_BLUE = "#EAF0FF"
LIGHT_GREEN = "#EAF7F0"
LIGHT_ORANGE = "#FFF3E5"
LIGHT_PURPLE = "#F0EDFF"
LIGHT_GRAY = "#F5F6F8"
DARK = "#1F2937"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="figures/head_gate_taylor_method",
        help="Directory for generated PNG/SVG diagrams.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser


def rounded_box(ax, xy, width, height, text, *, fc, ec, fontsize=11, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=DARK,
        weight=weight,
    )
    return box


def arrow(ax, start, end, *, color=GRAY, lw=1.8, mutation_scale=14):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        linewidth=lw,
        mutation_scale=mutation_scale,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(patch)
    return patch


def save(fig, output_dir, stem, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def draw_gate_location(output_dir, dpi):
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        0.25,
        4.78,
        "Attention head gate placement",
        fontsize=17,
        weight="bold",
        color=DARK,
    )
    ax.text(
        0.25,
        4.42,
        "A scalar gate masks each head's context vector before the output projection.",
        fontsize=11.5,
        color=GRAY,
    )

    y = 2.72
    x_positions = [0.35, 1.95, 3.55, 5.25, 7.05, 9.05, 11.0]
    widths = [1.2, 1.2, 1.25, 1.35, 1.55, 1.45, 1.35]
    labels = [
        "Input\nTokens",
        "QKV\nProjection",
        "Q, K, V",
        "softmax\n(QK^T)",
        "Per-head\nContext",
        "Head Outputs\n[B,T,H,D]",
        "Flatten\n[B,T,H*D]",
    ]
    colors = [
        (LIGHT_GRAY, GRAY),
        (LIGHT_BLUE, BLUE),
        (LIGHT_BLUE, BLUE),
        (LIGHT_ORANGE, ORANGE),
        (LIGHT_GREEN, GREEN),
        (LIGHT_GREEN, GREEN),
        (LIGHT_GRAY, GRAY),
    ]
    height = 0.78
    for x, w, label, (fc, ec) in zip(x_positions, widths, labels, colors):
        rounded_box(ax, (x, y), w, height, label, fc=fc, ec=ec)
    for idx in range(len(x_positions) - 1):
        arrow(
            ax,
            (x_positions[idx] + widths[idx], y + height / 2),
            (x_positions[idx + 1], y + height / 2),
        )

    gate_x, gate_y = 9.0, 1.25
    rounded_box(
        ax,
        (gate_x, gate_y),
        1.55,
        0.78,
        "Head Gate\n[B, T, H, 1]",
        fc=LIGHT_PURPLE,
        ec=PURPLE,
        fontsize=10.5,
        weight="bold",
    )
    ax.text(
        9.77,
        2.22,
        "broadcast over\nhead_dim",
        ha="center",
        va="center",
        fontsize=10,
        color=PURPLE,
    )
    arrow(ax, (9.77, gate_y + 0.78), (9.77, y), color=PURPLE, lw=2.2)

    proj_x, proj_y = 11.0, 0.05
    rounded_box(
        ax,
        (proj_x, proj_y),
        1.95,
        0.78,
        "attn.proj\nOutput Projection",
        fc=LIGHT_BLUE,
        ec=BLUE,
        fontsize=10.5,
    )
    arrow(
        ax,
        (11.0 + 1.35 / 2, y),
        (proj_x + 1.95 / 2, proj_y + 0.78),
        color=GRAY,
        lw=1.8,
    )

    rounded_box(
        ax,
        (12.65, proj_y),
        0.45,
        0.78,
        "y",
        fc=LIGHT_GRAY,
        ec=GRAY,
        fontsize=12,
        weight="bold",
    )
    arrow(ax, (proj_x + 1.95, proj_y + 0.39), (12.65, proj_y + 0.39))

    ax.text(
        7.05,
        3.78,
        "Head-wise masking before flattening:\ncontext [B,T,H,D] * gate [B,T,H,1]",
        ha="left",
        va="center",
        fontsize=11,
        color=GREEN,
        weight="bold",
    )
    ax.text(
        0.65,
        0.58,
        "Implementation note:\nthe projection input [B,T,H*D]\nis reshaped to [B,T,H,D] for scoring.",
        ha="left",
        va="center",
        fontsize=10.6,
        color=DARK,
    )

    return save(fig, output_dir, "gate_location", dpi)


def draw_samplewise_aggregation(output_dir, dpi):
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        0.25,
        4.78,
        "Samplewise head importance aggregation",
        fontsize=17,
        weight="bold",
        color=DARK,
    )
    ax.text(
        0.25,
        4.42,
        "Token-level gate Taylor contributions are aggregated into one score per attention head.",
        fontsize=11.5,
        color=GRAY,
    )

    boxes = [
        (
            0.45,
            "Gate Taylor\nContribution\n[B, T, H]",
            LIGHT_PURPLE,
            PURPLE,
        ),
        (
            3.0,
            "Sum over\nTokens\n[B, H]",
            LIGHT_GREEN,
            GREEN,
        ),
        (
            5.55,
            "Absolute\nValue\n[B, H]",
            LIGHT_ORANGE,
            ORANGE,
        ),
        (
            8.1,
            "Sum over\nCalibration Set\n[H]",
            LIGHT_BLUE,
            BLUE,
        ),
        (
            10.95,
            "Head Scores\nper block [H]\nmodel-wide [L,H]",
            LIGHT_GRAY,
            GRAY,
        ),
    ]
    y = 2.55
    width = 1.75
    height = 0.95
    for x, label, fc, ec in boxes:
        rounded_box(ax, (x, y), width, height, label, fc=fc, ec=ec, fontsize=11)
    for idx in range(len(boxes) - 1):
        x = boxes[idx][0]
        nx = boxes[idx + 1][0]
        arrow(ax, (x + width, y + height / 2), (nx, y + height / 2))

    ax.text(
        0.45,
        1.72,
        "c[b,t,h] = gate[b,t,h] * dL/dgate[b,t,h]",
        fontsize=11,
        color=PURPLE,
        weight="bold",
    )
    ax.text(
        3.0,
        1.18,
        "per-sample contribution:\nC[n,h] = sum_t c[n,t,h]",
        fontsize=10.5,
        color=GREEN,
    )
    ax.text(
        5.55,
        1.18,
        "avoid sign cancellation\nbetween samples",
        fontsize=10.5,
        color=ORANGE,
    )
    ax.text(
        8.1,
        1.18,
        "score[h] = sum_{n in D_cal} |C[n,h]|",
        fontsize=10.5,
        color=BLUE,
        weight="bold",
    )

    rounded_box(
        ax,
        (0.45, 0.15),
        12.25,
        0.58,
        "Final structural pruning: remove low-score heads, then delete matching qkv/proj slices with Torch-Pruning.",
        fc="#FAFAFB",
        ec="#D6DAE2",
        fontsize=11.2,
    )

    return save(fig, output_dir, "samplewise_aggregation", dpi)


def rect(ax, xy, width, height, *, fc, ec, lw=1.4, alpha=1.0):
    patch = Rectangle(
        xy,
        width,
        height,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def draw_head_block_contribution(output_dir, dpi):
    fig, ax = plt.subplots(figsize=(15.4, 5.8))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    ax.text(
        0.25,
        5.42,
        "Head pruning as output-projection block removal",
        fontsize=17,
        weight="bold",
        color=DARK,
    )
    ax.text(
        0.25,
        5.05,
        "Multi-head attention can be viewed as a sum of per-head context blocks passed through matching projection blocks.",
        fontsize=11.5,
        color=GRAY,
    )

    # Left matrix: concatenated head outputs O = [O_1 | ... | O_H].
    ox, oy = 0.65, 2.42
    block_w, block_h = 0.82, 1.55
    head_colors = [LIGHT_GREEN, "#DFF2EA", LIGHT_PURPLE, "#E9E5FF", LIGHT_GREEN]
    edge_colors = [GREEN, GREEN, PURPLE, PURPLE, GREEN]
    labels = ["O1", "O2", "Oh", "...", "OH"]
    for idx, label in enumerate(labels):
        fc = head_colors[idx]
        ec = edge_colors[idx]
        lw = 2.4 if label == "Oh" else 1.2
        rect(ax, (ox + idx * block_w, oy), block_w, block_h, fc=fc, ec=ec, lw=lw)
        ax.text(
            ox + idx * block_w + block_w / 2,
            oy + block_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold" if label == "Oh" else "normal",
            color=PURPLE if label == "Oh" else DARK,
        )
    ax.text(
        ox + 2.5 * block_w,
        oy + block_h + 0.35,
        r"Per-head context blocks $O=[O_1\,|\,O_2\,|\,\cdots\,|\,O_H]$",
        ha="center",
        va="center",
        fontsize=11,
        color=GREEN,
        weight="bold",
    )
    ax.text(
        ox + 2.5 * block_w,
        oy - 0.38,
        r"$O_h=\mathrm{softmax}(Q_hK_h^\top)V_h$     shape: $[T,D_{head}]$",
        ha="center",
        va="center",
        fontsize=10.3,
        color=GRAY,
    )

    ax.text(5.25, 3.18, "x", fontsize=20, weight="bold", color=DARK)

    # Middle matrix: output projection split into matching row blocks.
    wx, wy = 5.85, 1.42
    w_w, w_h = 1.15, 0.62
    w_labels = ["W1", "W2", "Wh", "...", "WH"]
    for idx, label in enumerate(w_labels):
        y = wy + (len(w_labels) - 1 - idx) * w_h
        fc = LIGHT_PURPLE if label == "Wh" else LIGHT_BLUE
        ec = PURPLE if label == "Wh" else BLUE
        lw = 2.4 if label == "Wh" else 1.2
        rect(ax, (wx, y), w_w, w_h, fc=fc, ec=ec, lw=lw)
        ax.text(
            wx + w_w / 2,
            y + w_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=11.5,
            weight="bold" if label == "Wh" else "normal",
            color=PURPLE if label == "Wh" else DARK,
        )
    ax.text(
        wx + w_w / 2,
        wy + len(w_labels) * w_h + 0.18,
        r"$W_O$ row blocks",
        ha="center",
        va="center",
        fontsize=11.0,
        color=BLUE,
        weight="bold",
    )

    ax.text(7.48, 3.18, "=", fontsize=20, weight="bold", color=DARK)

    # Right side: sum of per-head contributions.
    sx, sy = 8.25, 2.33
    terms = [
        (r"$O_1W_1$", LIGHT_GREEN, GREEN),
        ("+", "white", "white"),
        (r"$O_2W_2$", LIGHT_GREEN, GREEN),
        ("+", "white", "white"),
        (r"$O_hW_h$", LIGHT_PURPLE, PURPLE),
        ("+", "white", "white"),
        ("...", "white", "white"),
        ("+", "white", "white"),
        (r"$O_HW_H$", LIGHT_GREEN, GREEN),
    ]
    x = sx
    for text, fc, ec in terms:
        if text == "+":
            ax.text(x, sy + 0.42, "+", fontsize=17, weight="bold", color=GRAY)
            x += 0.32
            continue
        if text == "...":
            ax.text(x, sy + 0.42, "...", fontsize=15, weight="bold", color=GRAY)
            x += 0.56
            continue
        width = 1.12 if text != r"$O_hW_h$" else 1.18
        rect(ax, (x, sy), width, 0.86, fc=fc, ec=ec, lw=2.4 if text == r"$O_hW_h$" else 1.2)
        ax.text(
            x + width / 2,
            sy + 0.43,
            text,
            ha="center",
            va="center",
            fontsize=11.0,
            weight="bold" if text == r"$O_hW_h$" else "normal",
            color=PURPLE if text == r"$O_hW_h$" else DARK,
        )
        x += width + 0.1

    ax.text(
        8.25,
        3.75,
        r"$Y=OW_O=\sum_h O_hW_h$",
        fontsize=14,
        color=DARK,
        weight="bold",
    )

    # Masking annotation.
    mask_x = ox + 2 * block_w
    rect(ax, (mask_x, oy), block_w, block_h, fc="#FFFFFF", ec="#B42318", lw=2.0, alpha=0.72)
    ax.plot(
        [mask_x + 0.12, mask_x + block_w - 0.12],
        [oy + 0.15, oy + block_h - 0.15],
        color="#B42318",
        linewidth=2.2,
    )
    ax.plot(
        [mask_x + 0.12, mask_x + block_w - 0.12],
        [oy + block_h - 0.15, oy + 0.15],
        color="#B42318",
        linewidth=2.2,
    )
    ax.text(
        mask_x + block_w / 2,
        oy - 0.78,
        r"set gate $g_h \rightarrow 0$" "\n(mask head h)",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#B42318",
        weight="bold",
    )
    arrow(ax, (mask_x + block_w / 2, oy - 0.45), (mask_x + block_w / 2, oy), color="#B42318", lw=2.0)

    ax.text(
        8.25,
        1.22,
        r"Masking head $h$ removes its full contribution $O_hW_h$.",
        fontsize=12.0,
        color="#B42318",
        weight="bold",
    )
    ax.text(
        8.25,
        0.72,
        "Importance is estimated by the first-order loss change of this masked block;\ntoken-level contributions are summed into one score per head.",
        fontsize=10.8,
        color=GRAY,
    )

    return save(fig, output_dir, "head_block_contribution", dpi)


def main(args):
    output_dir = Path(args.output_dir)
    for path in draw_gate_location(output_dir, args.dpi):
        print(f"[HeadGateTaylorDiagrams] saved {path}")
    for path in draw_samplewise_aggregation(output_dir, args.dpi):
        print(f"[HeadGateTaylorDiagrams] saved {path}")
    for path in draw_head_block_contribution(output_dir, args.dpi):
        print(f"[HeadGateTaylorDiagrams] saved {path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
