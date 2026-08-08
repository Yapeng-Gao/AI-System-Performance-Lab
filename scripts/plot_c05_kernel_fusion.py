#!/usr/bin/env python3
"""Plot C-05 kernel fusion sweep (fused/naive vs chain length k).

Reads:  docs/results/C-05_sweep.csv
Writes: article/03_compute_primitives/assets/C-05-speedup-vs-k.png

CSV:
  k,naive_ms,fused_ms,fused_speedup
  2,...

Usage:
  python scripts/plot_c05_kernel_fusion.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "C-05_sweep.csv"
OUT = ROOT / "article" / "03_compute_primitives" / "assets" / "C-05-speedup-vs-k.png"

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
    if not SWEEP_CSV.is_file():
        print(f"missing {SWEEP_CSV}")
        return
    ks: list[int] = []
    sp: list[float] = []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ks.append(int(row["k"]))
            sp.append(float(row["fused_speedup"]))
    x = np.asarray(ks)
    y = np.asarray(sp)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "o-", color=CYAN, lw=2, ms=8, label="fused/naive")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.6, label="1.0x")
    ax.set_xlabel("chain length k (elementwise stages)")
    ax.set_ylabel("speedup (naive wall / fused wall)")
    ax.set_title("C-05: vertical fusion vs multi-kernel chain length")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.set_xticks(x)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
