#!/usr/bin/env python3
"""Plot C-03 atomics contention sweep (speedup vs hit_rate).

Reads:  docs/results/C-03_sweep.csv
Writes: article/03_compute_primitives/assets/C-03-speedup-vs-hitrate.png

Curated CSV format (fill after bare --mode sweep):

  hit_rate,naive_ms,smem_ms,agg_ms,agg_speedup,smem_speedup
  0.05,...
  ...
  1.0,...

Usage (repo root):
  python scripts/plot_c03_atomics_contention.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "C-03_sweep.csv"
OUT = ROOT / "article" / "03_compute_primitives" / "assets" / "C-03-speedup-vs-hitrate.png"

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
    hit: list[float] = []
    agg_sp: list[float] = []
    smem_sp: list[float] = []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            hit.append(float(row["hit_rate"]))
            agg_sp.append(float(row["agg_speedup"]))
            smem_sp.append(float(row["smem_speedup"]))

    x = np.asarray(hit)
    y_a = np.asarray(agg_sp)
    y_s = np.asarray(smem_sp)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y_a, "o-", color=CYAN, lw=2, ms=8, label="agg/naive")
    ax.plot(x, y_s, "s--", color=AMBER, lw=2, ms=7, label="smem/naive")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.6, label="1.0x")
    ax.set_xlabel("hit_rate (filter pass fraction → contention)")
    ax.set_ylabel("speedup vs naive global atomic")
    ax.set_title("C-03 RTX 5090: warp-agg / smem staging vs hit_rate")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ymin = min(0.8, float(min(y_a.min(), y_s.min())) - 0.1)
    ymax = max(1.5, float(max(y_a.max(), y_s.max())) + 0.2)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
