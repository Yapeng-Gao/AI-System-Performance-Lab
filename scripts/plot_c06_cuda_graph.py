#!/usr/bin/env python3
"""Plot C-06 CUDA Graph sweeps.

Reads:
  docs/results/C-06_sweep.csv       -> stream/graph speedup vs n_nodes
  docs/results/C-06_sweep_work.csv  -> speedup vs work (optional)

Writes:
  article/03_compute_primitives/assets/C-06-speedup-vs-nnodes.png
  article/03_compute_primitives/assets/C-06-speedup-vs-work.png (if CSV exists)

CSV formats:
  n_nodes,stream_ms,graph_ms,speedup
  work,stream_ms,graph_ms,speedup

Usage:
  python scripts/plot_c06_cuda_graph.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "C-06_sweep.csv"
WORK_CSV = ROOT / "docs" / "results" / "C-06_sweep_work.csv"
OUT_NODES = ROOT / "article" / "03_compute_primitives" / "assets" / "C-06-speedup-vs-nnodes.png"
OUT_WORK = ROOT / "article" / "03_compute_primitives" / "assets" / "C-06-speedup-vs-work.png"

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


def plot_nodes() -> None:
    if not SWEEP_CSV.is_file():
        print(f"skip nodes plot: missing {SWEEP_CSV}")
        return
    xs: list[int] = []
    ys: list[float] = []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            xs.append(int(row["n_nodes"]))
            ys.append(float(row["speedup"]))
    x = np.asarray(xs)
    y = np.asarray(ys)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "o-", color=CYAN, lw=2, ms=8, label="stream/graph")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.6, label="1.0x")
    ax.set_xlabel("n_nodes (short kernels per chain)")
    ax.set_ylabel("speedup (stream wall / graph wall)")
    ax.set_title("C-06: CUDA Graph vs stream launch (short-kernel chain)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.set_xticks(x)
    fig.tight_layout()
    OUT_NODES.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_NODES, dpi=160, facecolor=BG)
    print(f"wrote {OUT_NODES}")


def plot_work() -> None:
    if not WORK_CSV.is_file():
        print(f"skip work plot: missing {WORK_CSV}")
        return
    xs: list[int] = []
    ys: list[float] = []
    with WORK_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            xs.append(int(row["work"]))
            ys.append(float(row["speedup"]))
    x = np.asarray(xs)
    y = np.asarray(ys)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "s-", color=AMBER, lw=2, ms=8, label="stream/graph")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.6, label="1.0x")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("work (FMA iters per element)")
    ax.set_ylabel("speedup (stream wall / graph wall)")
    ax.set_title("C-06: Graph benefit vs kernel work (should shrink)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    OUT_WORK.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_WORK, dpi=160, facecolor=BG)
    print(f"wrote {OUT_WORK}")


def main() -> None:
    plot_nodes()
    plot_work()


if __name__ == "__main__":
    main()
