#!/usr/bin/env python3
"""Plot D-01 device-reduce sweep and mechanism figures.

Reads docs/results/D-01_sweep.csv.
Writes:
  article/04_operators/assets/D-01-device-reduce-cover.png
  article/04_operators/assets/D-01-fig1-scalar-vs-float4.png
  article/04_operators/assets/D-01-speedup-vs-n.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "D-01_sweep.csv"
OUT_DIR = ROOT / "article" / "04_operators" / "assets"
OUT_SPEED = OUT_DIR / "D-01-speedup-vs-n.png"
OUT_COVER = OUT_DIR / "D-01-device-reduce-cover.png"
OUT_FIG1 = OUT_DIR / "D-01-fig1-scalar-vs-float4.png"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"
GREEN = "#3fb950"


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, ls="--", color=GRID, alpha=0.7)


def _box(ax, xy, w, h, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=color,
            facecolor=color,
            alpha=0.9,
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
    _box(ax, (1.2, 3.2), 5.6, 2.4, CYAN)
    _box(ax, (9.0, 3.2), 5.6, 2.4, AMBER)
    ax.text(4.0, 4.4, "scalar", color=BG, fontsize=28, ha="center", va="center", fontweight="bold")
    ax.text(11.8, 4.4, "float4", color=BG, fontsize=28, ha="center", va="center", fontweight="bold")
    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COVER, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_COVER}")


def draw_fig1() -> None:
    fig, ax = _canvas()
    _box(ax, (1.0, 5.6), 6.2, 1.6, CYAN)
    _box(ax, (1.0, 2.4), 6.2, 1.6, AMBER)
    ax.text(4.1, 6.4, "ld.32", color=BG, fontsize=22, ha="center", va="center", fontweight="bold")
    ax.text(4.1, 3.2, "ld.128", color=BG, fontsize=22, ha="center", va="center", fontweight="bold")
    ax.text(9.2, 4.8, "one atomic", color=FG, fontsize=22)
    ax.text(9.2, 3.4, "or 2nd kernel", color=FG, fontsize=18, alpha=0.85)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT_FIG1, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_FIG1}")


def draw_speedup() -> None:
    if not SWEEP_CSV.is_file():
        print(f"skip plot: missing {SWEEP_CSV}")
        return
    ns: list[int] = []
    vec: list[float] = []
    cub: list[float] = []
    vc: list[float] = []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ns.append(int(row["n"]))
            vec.append(float(row["scalar_over_vec4"]))
            cub.append(float(row["scalar_over_cub"]))
            vc.append(float(row.get("vec4_over_cub", float(row["vec4_ms"]) / float(row["cub_ms"]))))
    x = np.asarray(ns, dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, vec, "o-", color=CYAN, lw=2, ms=8, label="scalar/vec4")
    ax.plot(x, cub, "s-", color=AMBER, lw=2, ms=7, label="scalar/cub")
    ax.plot(x, vc, "^-", color=GREEN, lw=2, ms=7, label="vec4/cub")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.5, 1.7)
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
