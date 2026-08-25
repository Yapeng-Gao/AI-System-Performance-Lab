#!/usr/bin/env python3
"""Plot C-08 PDL sweeps and mechanism figures.

Reads:
  docs/results/C-08_sweep.csv       -> serial/pdl vs work
  docs/results/C-08_sweep_tail.csv  -> serial/pdl vs tail

Writes:
  article/03_compute_primitives/assets/C-08-speedup-vs-work.png
  article/03_compute_primitives/assets/C-08-speedup-vs-tail.png
  article/03_compute_primitives/assets/C-08-pdl-cover.png
  article/03_compute_primitives/assets/C-08-fig1-serial-vs-pdl.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "C-08_sweep.csv"
TAIL_CSV = ROOT / "docs" / "results" / "C-08_sweep_tail.csv"
OUT_DIR = ROOT / "article" / "03_compute_primitives" / "assets"
OUT_WORK = OUT_DIR / "C-08-speedup-vs-work.png"
OUT_TAIL = OUT_DIR / "C-08-speedup-vs-tail.png"
OUT_COVER = OUT_DIR / "C-08-pdl-cover.png"
OUT_FIG1 = OUT_DIR / "C-08-fig1-serial-vs-pdl.png"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, ls="--", color=GRID, alpha=0.7)


def _plot_xy(csv_path: Path, xkey: str, xlabel: str, title: str, out: Path) -> None:
    if not csv_path.is_file():
        print(f"skip plot: missing {csv_path}")
        return
    xs: list[int] = []
    ys: list[float] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            xs.append(int(row[xkey]))
            ys.append(float(row["speedup"]))
    if not xs:
        print(f"skip plot: empty {csv_path}")
        return
    x = np.asarray(xs)
    y = np.asarray(ys)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "o-", color=CYAN, lw=2, ms=8, label="serial/pdl")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.6, label="1.0x")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("speedup (serial wall / pdl wall)")
    ax.set_title(title)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


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
            alpha=0.88,
        )
    )


def draw_cover() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(1.0, 7.5, "serial", color=CYAN, fontsize=28, fontweight="bold")
    _box(ax, (1.0, 5.6), 6.2, 1.3, CYAN)
    _box(ax, (8.0, 5.6), 6.2, 1.3, CYAN)
    ax.text(4.1, 6.15, "K1", color=BG, fontsize=20, ha="center", va="center")
    ax.text(11.1, 6.15, "K2", color=BG, fontsize=20, ha="center", va="center")
    ax.text(1.0, 4.2, "PDL", color=AMBER, fontsize=28, fontweight="bold")
    _box(ax, (1.0, 1.7), 7.4, 1.5, AMBER)
    _box(ax, (5.6, 1.7), 8.6, 1.5, CYAN)
    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COVER, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_COVER}")


def draw_fig1() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.6, 8.15, "same stream, wait for retire", color=CYAN, fontsize=20, fontweight="bold")
    _box(ax, (1.0, 5.9), 5.5, 1.4, CYAN)
    _box(ax, (7.2, 5.9), 5.5, 1.4, CYAN)
    ax.text(3.75, 6.6, "K1 store+tail", color=BG, fontsize=16, ha="center", va="center")
    ax.text(9.95, 6.6, "K2 wait+body", color=BG, fontsize=16, ha="center", va="center")
    ax.text(0.6, 4.15, "PDL: trigger then overlap", color=AMBER, fontsize=20, fontweight="bold")
    _box(ax, (1.0, 1.55), 4.2, 1.5, AMBER)
    _box(ax, (5.4, 1.55), 4.4, 1.5, AMBER)
    _box(ax, (7.2, 1.55), 6.6, 1.5, CYAN)
    ax.text(3.1, 2.3, "store+trig", color=BG, fontsize=14, ha="center", va="center")
    ax.text(7.5, 2.3, "tail || work", color=BG, fontsize=14, ha="center", va="center")
    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG1, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_FIG1}")


def main() -> None:
    _plot_xy(
        SWEEP_CSV,
        "work",
        "work (K2 independent FMA iters)",
        "C-08: PDL vs serial (sweep work)",
        OUT_WORK,
    )
    _plot_xy(
        TAIL_CSV,
        "tail",
        "tail (K1 FMA after trigger)",
        "C-08: PDL vs serial (sweep tail)",
        OUT_TAIL,
    )
    draw_cover()
    draw_fig1()


if __name__ == "__main__":
    main()
