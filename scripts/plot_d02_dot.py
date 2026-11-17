#!/usr/bin/env python3
"""D-02 operator-card figures.

Cover  = Dot → cublasSdot
Fig1   = traffic shape (read x + read y)
Speed  = scalar/vec4 + vec4/cublas vs n
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "D-02_sweep.csv"
OUT_DIR = ROOT / "article" / "04_operators" / "assets"
OUT_SPEED = OUT_DIR / "D-02-speedup-vs-n.png"
OUT_COVER = OUT_DIR / "D-02-dot-cover.png"
OUT_FIG1 = OUT_DIR / "D-02-fig1-two-streams.png"

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
    ax.text(4.0, 4.9, "Dot", color=BG, fontsize=30, ha="center",
            va="center", fontweight="bold")
    ax.text(4.0, 3.7, "dot → cublasSdot", color=BG, fontsize=15, ha="center", va="center", alpha=0.85)

    ax.annotate(
        "", xy=(10.2, 4.5), xytext=(7.4, 4.5),
        arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.5, mutation_scale=18),
    )

    _box(ax, (10.4, 3.0), 4.6, 3.0, GREEN)
    ax.text(12.7, 4.9, "cublasSdot", color=BG, fontsize=22, ha="center",
            va="center", fontweight="bold")
    ax.text(12.7, 3.7, "reach ≈ 1.00", color=BG, fontsize=15, ha="center", va="center")
    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COVER, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_COVER}")


def draw_fig1() -> None:
    """Traffic shape: two read streams → mul-add → reduce."""
    fig, ax = _canvas()
    _box(ax, (1.0, 5.6), 5.2, 1.8, CYAN)
    _box(ax, (1.0, 2.8), 5.2, 1.8, AMBER)
    ax.text(3.6, 6.5, "read x", color=BG, fontsize=24, ha="center",
            va="center", fontweight="bold")
    ax.text(3.6, 3.7, "read y", color=BG, fontsize=24, ha="center",
            va="center", fontweight="bold")

    ax.annotate(
        "", xy=(8.2, 5.5), xytext=(6.4, 6.5),
        arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.0, mutation_scale=14),
    )
    ax.annotate(
        "", xy=(8.2, 4.5), xytext=(6.4, 3.7),
        arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.0, mutation_scale=14),
    )

    _box(ax, (8.4, 3.6), 6.4, 2.8, GREEN)
    ax.text(11.6, 5.5, "mul-add", color=BG, fontsize=22, ha="center",
            va="center", fontweight="bold")
    ax.text(11.6, 4.3, "→ block reduce → 1", color=BG, fontsize=16, ha="center", va="center")
    ax.text(8.0, 1.4, "traffic ~2× D-01; widen-load gain shrinks to ~1.02×",
            color=FG, fontsize=15, ha="center")
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT_FIG1, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_FIG1}")


def draw_speedup() -> None:
    if not SWEEP_CSV.is_file():
        print(f"skip plot: missing {SWEEP_CSV}")
        return
    ns, sv, vc = [], [], []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ns.append(int(row["n"]))
            sv.append(float(row["scalar_over_vec4"]))
            vc.append(float(row["vec4_over_cublas"]))
    x = np.asarray(ns, dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, sv, "o-", color=CYAN, lw=2, ms=8, label="scalar/vec4")
    ax.plot(x, vc, "^-", color=GREEN, lw=2, ms=7, label="vec4/cublas")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.7)
    ax.axvline(2**26, color=MUTED, ls="--", lw=1.0, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.4, 1.4)
    ax.set_xlabel("n (float elements)")
    ax.set_ylabel("ratio")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SPEED, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_SPEED}")


if __name__ == "__main__":
    draw_cover()
    draw_fig1()
    draw_speedup()
