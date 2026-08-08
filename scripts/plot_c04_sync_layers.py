#!/usr/bin/env python3
"""Plot C-04 sync-layer sweeps.

Reads:
  docs/results/C-04_sweep.csv       -> block/warp ratio vs nwarps
  docs/results/C-04_sweep_grid.csv  -> grid sync median_ms vs nblocks

Writes:
  article/03_compute_primitives/assets/C-04-ratio-vs-nwarps.png
  article/03_compute_primitives/assets/C-04-grid-vs-nblocks.png

Curated CSV formats:

  # C-04_sweep.csv
  nwarps,warp_ms,block_ms,ratio_block_warp
  1,...

  # C-04_sweep_grid.csv
  nblocks,grid_ms
  1,...

Usage (repo root):
  python scripts/plot_c04_sync_layers.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "C-04_sweep.csv"
GRID_CSV = ROOT / "docs" / "results" / "C-04_sweep_grid.csv"
OUT_RATIO = ROOT / "article" / "03_compute_primitives" / "assets" / "C-04-ratio-vs-nwarps.png"
OUT_GRID = ROOT / "article" / "03_compute_primitives" / "assets" / "C-04-grid-vs-nblocks.png"

BG = "#0d1117"
FG = "#e6edf3"
GRID_C = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID_C)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, ls="--", color=GRID_C, alpha=0.7)


def plot_ratio() -> None:
    if not SWEEP_CSV.is_file():
        print(f"skip ratio plot: missing {SWEEP_CSV}")
        return
    nwarps: list[int] = []
    ratios: list[float] = []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            nwarps.append(int(row["nwarps"]))
            ratios.append(float(row["ratio_block_warp"]))
    x = np.asarray(nwarps)
    y = np.asarray(ratios)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "o-", color=CYAN, lw=2, ms=8, label="block/warp")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.6, label="1.0x")
    ax.set_xlabel("nwarps (blockDim = nwarps × 32)")
    ax.set_ylabel("median time ratio (block / warp)")
    ax.set_title("C-04: empty __syncthreads vs __syncwarp vs nwarps")
    ax.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=FG)
    ax.set_xticks(x)
    fig.tight_layout()
    OUT_RATIO.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_RATIO, dpi=160, facecolor=BG)
    print(f"wrote {OUT_RATIO}")


def plot_grid() -> None:
    if not GRID_CSV.is_file():
        print(f"skip grid plot: missing {GRID_CSV}")
        return
    nblocks: list[int] = []
    ms: list[float] = []
    with GRID_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            nblocks.append(int(row["nblocks"]))
            ms.append(float(row["grid_ms"]))
    x = np.asarray(nblocks)
    y = np.asarray(ms)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "s-", color=AMBER, lw=2, ms=8, label="grid.sync")
    ax.set_xlabel("nblocks (cooperative launch, clamped)")
    ax.set_ylabel("median time (ms)")
    ax.set_title("C-04: empty this_grid().sync() vs nblocks")
    ax.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=FG)
    fig.tight_layout()
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GRID, dpi=160, facecolor=BG)
    print(f"wrote {OUT_GRID}")


def main() -> None:
    plot_ratio()
    plot_grid()


if __name__ == "__main__":
    main()
