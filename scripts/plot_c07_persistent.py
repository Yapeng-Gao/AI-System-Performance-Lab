#!/usr/bin/env python3
"""Plot C-07 Persistent Kernel sweeps.

Reads:
  docs/results/C-07_sweep.csv       -> launch/persistent speedup vs n_tasks
  docs/results/C-07_sweep_work.csv  -> speedup vs work (optional)

Writes:
  article/03_compute_primitives/assets/C-07-speedup-vs-ntasks.png
  article/03_compute_primitives/assets/C-07-speedup-vs-work.png (if CSV exists)
  article/03_compute_primitives/assets/C-07-persistent-cover.png
  article/03_compute_primitives/assets/C-07-fig1-launch-vs-persist.png

CSV formats:
  n_tasks,launch_ms,persist_ms,speedup
  work,launch_ms,persist_ms,speedup

Usage:
  python scripts/plot_c07_persistent.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "C-07_sweep.csv"
WORK_CSV = ROOT / "docs" / "results" / "C-07_sweep_work.csv"
OUT_DIR = ROOT / "article" / "03_compute_primitives" / "assets"
OUT_TASKS = OUT_DIR / "C-07-speedup-vs-ntasks.png"
OUT_WORK = OUT_DIR / "C-07-speedup-vs-work.png"
OUT_COVER = OUT_DIR / "C-07-persistent-cover.png"
OUT_FIG1 = OUT_DIR / "C-07-fig1-launch-vs-persist.png"

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


def plot_tasks() -> None:
    if not SWEEP_CSV.is_file():
        print(f"skip n_tasks plot: missing {SWEEP_CSV}")
        return
    xs: list[int] = []
    ys: list[float] = []
    with SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            xs.append(int(row["n_tasks"]))
            ys.append(float(row["speedup"]))
    if not xs:
        print(f"skip n_tasks plot: empty {SWEEP_CSV}")
        return
    x = np.asarray(xs)
    y = np.asarray(ys)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "o-", color=CYAN, lw=2, ms=8, label="launch/persistent")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.6, label="1.0x")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_tasks (1 launch per task)")
    ax.set_ylabel("speedup (launch wall / persistent wall)")
    ax.set_title("C-07: persistent pull vs one-launch-per-task")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in xs])
    fig.tight_layout()
    OUT_TASKS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_TASKS, dpi=160, facecolor=BG)
    print(f"wrote {OUT_TASKS}")


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
    if not xs:
        print(f"skip work plot: empty {WORK_CSV}")
        return
    x = np.asarray(xs)
    y = np.asarray(ys)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    ax.plot(x, y, "s-", color=AMBER, lw=2, ms=8, label="launch/persistent")
    ax.axhline(1.0, color=FG, ls=":", lw=1.2, alpha=0.6, label="1.0x")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("work (FMA iters per task)")
    ax.set_ylabel("speedup (launch wall / persistent wall)")
    ax.set_title("C-07: persistent benefit vs task work (should shrink)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    OUT_WORK.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_WORK, dpi=160, facecolor=BG)
    print(f"wrote {OUT_WORK}")


def _rounded_box(ax, xy, w, h, color: str, lw: float = 1.4) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=lw,
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

    ax.text(1.0, 7.6, "launch", color=CYAN, fontsize=28, fontweight="bold")
    for i in range(8):
        _rounded_box(ax, (1.0 + i * 1.55, 5.7), 1.15, 1.15, CYAN)

    ax.text(1.0, 4.15, "persistent", color=AMBER, fontsize=28, fontweight="bold")
    for i in range(4):
        _rounded_box(ax, (1.0 + i * 1.85, 2.15), 1.55, 1.45, AMBER)
    ax.add_patch(
        plt.Circle((10.6, 2.85), 0.72, fill=False, edgecolor=FG, lw=2.2)
    )
    ax.text(10.6, 2.85, "queue", color=FG, fontsize=16, ha="center", va="center")
    ax.text(13.15, 2.85, "occupancy", color=FG, fontsize=20, va="center")

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

    ax.text(0.6, 8.15, "1 task = 1 launch", color=CYAN, fontsize=22, fontweight="bold")
    for i in range(6):
        x = 1.1 + i * 2.35
        _rounded_box(ax, (x, 5.85), 1.55, 1.35, CYAN)
        ax.annotate(
            "",
            xy=(x + 0.75, 5.85),
            xytext=(x + 0.75, 5.15),
            arrowprops={"arrowstyle": "->", "color": FG, "lw": 1.4},
        )
    ax.text(8.0, 4.85, "host  x N", color=FG, fontsize=14, ha="center")

    ax.text(0.6, 3.85, "occupancy grid pulls", color=AMBER, fontsize=22, fontweight="bold")
    for i in range(4):
        _rounded_box(ax, (1.1 + i * 2.05, 1.45), 1.7, 1.45, AMBER)
    ax.add_patch(plt.Circle((11.35, 2.15), 0.85, fill=False, edgecolor=FG, lw=2.2))
    ax.text(11.35, 2.15, "atomicAdd", color=FG, fontsize=12, ha="center", va="center")
    ax.annotate(
        "",
        xy=(9.15, 2.15),
        xytext=(10.5, 2.15),
        arrowprops={"arrowstyle": "<->", "color": FG, "lw": 1.5},
    )

    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG1, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_FIG1}")


def main() -> None:
    plot_tasks()
    plot_work()
    draw_cover()
    draw_fig1()


if __name__ == "__main__":
    main()
