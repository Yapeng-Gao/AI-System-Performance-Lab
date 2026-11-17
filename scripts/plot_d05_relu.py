#!/usr/bin/env python3
"""D-05 ReLU short-chapter figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "D-05_sweep.csv"
OUT_DIR = ROOT / "article" / "04_operators" / "assets"
OUT_COVER = OUT_DIR / "D-05-relu-cover.png"
OUT_BW = OUT_DIR / "D-05-bandwidth-vs-n.png"

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
CYAN, AMBER, MUTED = "#39c5cf", "#f0a202", "#8b949e"


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(True, ls="--", color=GRID, alpha=0.7)


def draw_cover() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (1.0, 3.0), 6.0, 3.0, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=CYAN, facecolor=CYAN, alpha=0.9,
        )
    )
    ax.text(4.0, 4.9, "ReLU", color=BG, fontsize=28, ha="center",
            va="center", fontweight="bold")
    ax.text(4.0, 3.7, "max(0,x) → cuDNN", color=BG, fontsize=14, ha="center", va="center")
    ax.annotate(
        "", xy=(10.2, 4.5), xytext=(7.4, 4.5),
        arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.5, mutation_scale=18),
    )
    ax.add_patch(
        FancyBboxPatch(
            (10.4, 3.0), 4.6, 3.0, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=AMBER, facecolor=AMBER, alpha=0.9,
        )
    )
    ax.text(12.7, 4.9, "~1.50 TB/s", color=BG, fontsize=24, ha="center",
            va="center", fontweight="bold")
    ax.text(12.7, 3.7, "same BW wall", color=BG, fontsize=15, ha="center", va="center")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT_COVER, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_COVER}")


def draw_bw() -> None:
    ns, gb, cudnn = [], [], []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ns.append(int(row["n"]))
            gb.append(float(row["vec4_GBps"]))
            cudnn.append(float(row["cudnn_GBps"]))
    x = np.asarray(ns, dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, gb, "s-", color=AMBER, lw=2, ms=7, label="hand vec4 GB/s")
    ax.plot(x, cudnn, "o--", color=CYAN, lw=1.6, ms=6, label="cuDNN GB/s")
    ax.axhline(1550, color=MUTED, ls=":", lw=1.0, alpha=0.7, label="D-03 ~1.55 TB/s")
    ax.axvline(2**26, color=MUTED, ls="--", lw=1.0, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (float elements)")
    ax.set_ylabel("GB/s (2*n*4, incl. L2)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    fig.savefig(OUT_BW, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_BW}")


if __name__ == "__main__":
    draw_cover()
    draw_bw()
