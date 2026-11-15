#!/usr/bin/env python3
"""D-03 operator-card figures.

Cover  = a+b → useful bandwidth
Fig1   = traffic shape (2R1W)
Speed  = scalar/vec4 vs n
BW     = vec4 useful GB/s vs n
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "D-03_sweep.csv"
OUT_DIR = ROOT / "article" / "04_operators" / "assets"
OUT_SPEED = OUT_DIR / "D-03-speedup-vs-n.png"
OUT_COVER = OUT_DIR / "D-03-elementwise-cover.png"
OUT_BW = OUT_DIR / "D-03-bandwidth-vs-n.png"
OUT_FIG1 = OUT_DIR / "D-03-fig1-two-read-one-write.png"

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
CYAN, AMBER, GREEN, MUTED = "#39c5cf", "#f0a202", "#3fb950", "#8b949e"


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(True, ls="--", color=GRID, alpha=0.7)


def _box(ax, xy, w, h, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy, w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.9,
        )
    )


def _canvas():
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return fig, ax


def draw_cover() -> None:
    fig, ax = _canvas()
    _box(ax, (1.0, 3.0), 6.0, 3.0, CYAN)
    ax.text(4.0, 4.9, "Elementwise", color=BG, fontsize=26, ha="center",
            va="center", fontweight="bold")
    ax.text(4.0, 3.7, "hand-written a+b", color=BG, fontsize=15, ha="center", va="center", alpha=0.85)

    ax.annotate(
        "", xy=(10.2, 4.5), xytext=(7.4, 4.5),
        arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.5, mutation_scale=18),
    )

    _box(ax, (10.4, 3.0), 4.6, 3.0, AMBER)
    ax.text(12.7, 4.9, "~1.55 TB/s", color=BG, fontsize=24, ha="center",
            va="center", fontweight="bold")
    ax.text(12.7, 3.7, "reach BW ceiling", color=BG, fontsize=14, ha="center", va="center")
    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COVER, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_COVER}")


def draw_fig1() -> None:
    """Traffic: 2 read + 1 write, no reduce."""
    fig, ax = _canvas()
    _box(ax, (1.0, 6.0), 4.4, 1.6, CYAN)
    _box(ax, (1.0, 3.8), 4.4, 1.6, AMBER)
    _box(ax, (1.0, 1.6), 4.4, 1.6, GREEN)
    ax.text(3.2, 6.8, "read a", color=BG, fontsize=22, ha="center",
            va="center", fontweight="bold")
    ax.text(3.2, 4.6, "read b", color=BG, fontsize=22, ha="center",
            va="center", fontweight="bold")
    ax.text(3.2, 2.4, "write c", color=BG, fontsize=22, ha="center",
            va="center", fontweight="bold")

    for y in (6.8, 4.6, 2.4):
        ax.annotate(
            "", xy=(7.2, 4.6), xytext=(5.6, y),
            arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.6, mutation_scale=12),
        )

    _box(ax, (7.4, 3.2), 7.4, 2.8, MUTED)
    ax.text(11.1, 5.1, "3 · n · 4 bytes", color=BG, fontsize=22, ha="center",
            va="center", fontweight="bold")
    ax.text(11.1, 3.9, "no reduce tree", color=BG, fontsize=16, ha="center", va="center")
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT_FIG1, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_FIG1}")


def draw_plots() -> None:
    if not SWEEP_CSV.is_file():
        print(f"skip plot: missing {SWEEP_CSV}")
        return
    ns, sp, gb = [], [], []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ns.append(int(row["n"]))
            sp.append(float(row["scalar_over_vec4"]))
            gb.append(float(row["vec4_GBps"]))
    x = np.asarray(ns, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, sp, "o-", color=CYAN, lw=2, ms=8, label="scalar/vec4")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.7)
    ax.axvline(2**26, color=MUTED, ls="--", lw=1.0, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.7, 1.4)
    ax.set_xlabel("n (float elements)")
    ax.set_ylabel("scalar/vec4")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    fig.savefig(OUT_SPEED, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_SPEED}")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, gb, "s-", color=AMBER, lw=2, ms=7, label="vec4 useful GB/s")
    ax.axhline(1550, color=GREEN, ls="--", lw=1.2, alpha=0.8, label="~1.55 TB/s")
    ax.axvline(2**26, color=MUTED, ls="--", lw=1.0, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (float elements)")
    ax.set_ylabel("GB/s (incl. L2)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    fig.savefig(OUT_BW, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_BW}")


if __name__ == "__main__":
    draw_cover()
    draw_fig1()
    draw_plots()
