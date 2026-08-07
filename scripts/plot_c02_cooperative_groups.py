#!/usr/bin/env python3
"""Plot C-02 Cooperative Groups tile-size sweep (cg::reduce vs tile size).

Reads:  docs/results/C-02_sweep.csv
Writes: article/03_compute_primitives/assets/C-02-cliff-vs-tilesize.png

Curated CSV format (fill from `--mode sweep --csv-only` after cleanup):

  tile,median_ms,norm
  8,<ms>,<ms/ms@32>
  16,...
  32,...,1.000
  64,...
  128,...

`norm` = median_ms / median_ms@tile=32 (abstraction-tax-free baseline). The
>32 points rising above 1.0 is the multi-warp CG software-sync cliff.

Usage (repo root):
  python scripts/plot_c02_cooperative_groups.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "C-02_sweep.csv"
OUT = ROOT / "article" / "03_compute_primitives" / "assets" / "C-02-cliff-vs-tilesize.png"

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


def main() -> None:
    tiles: list[int] = []
    norm: list[float] = []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tiles.append(int(row["tile"]))
            norm.append(float(row["norm"]))

    x = np.asarray(tiles)
    y = np.asarray(norm)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "o-", color=CYAN, lw=2, ms=8, label="cg::reduce (norm to tile=32)")
    ax.axhline(1.0, color=AMBER, ls=":", lw=1.5, label="tile=32 baseline")
    ax.axvline(32, color=FG, ls="--", lw=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in tiles])
    ax.set_xlabel("tile size N (tiled_partition<N>)")
    ax.set_ylabel("normalized time (÷ tile=32)")
    ax.set_title("C-02 RTX 5090: CG tile-size cliff (>32 = software multi-warp sync)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.set_ylim(0.9, max(1.5, float(y.max()) + 0.1))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
