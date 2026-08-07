#!/usr/bin/env python3
"""Plot C-01 warp primitives sweep (shfl/smem vs nwarps).

Reads:  docs/results/C-01_sweep.csv
Writes: article/03_compute_primitives/assets/C-01-speedup-vs-nwarps.png

Usage (repo root):
  python scripts/plot_c01_warp_primitives.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "C-01_sweep.csv"
OUT = ROOT / "article" / "03_compute_primitives" / "assets" / "C-01-speedup-vs-nwarps.png"

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
    nwarps: list[int] = []
    speedup: list[float] = []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            nwarps.append(int(row["nwarps"]))
            speedup.append(float(row["speedup"]))

    x = np.asarray(nwarps)
    y = np.asarray(speedup)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "o-", color=CYAN, lw=2, ms=8, label="shfl/smem")
    ax.axhline(1.0, color=AMBER, ls=":", lw=1.5, label="1.0x (parity)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in nwarps])
    ax.set_xlabel("nwarps (blockDim = nwarps × 32)")
    ax.set_ylabel("speedup (shfl / smem)")
    ax.set_title("C-01 RTX 5090: shfl/smem vs nwarps")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.set_ylim(0.9, max(1.35, float(y.max()) + 0.05))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
