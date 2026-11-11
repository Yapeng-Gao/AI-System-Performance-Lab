#!/usr/bin/env python3
"""Draw C-10 checklist figures. No CSV: this chapter adds no micro-bench."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "article" / "03_compute_primitives" / "assets"
OUT_COVER = OUT_DIR / "C-10-checklist-cover.png"
OUT_FIG1 = OUT_DIR / "C-10-triage-layers.png"

BG = "#0d1117"
FG = "#e6edf3"
CYAN = "#39c5cf"
AMBER = "#f0a202"


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
    boxes = [
        (1.0, "symptom", CYAN),
        (6.2, "evidence", AMBER),
        (11.2, "fix", CYAN),
    ]
    for x, label, color in boxes:
        _box(ax, (x, 3.4), 3.6, 2.2, color)
        ax.text(x + 1.8, 4.5, label, color=BG, fontsize=20, ha="center", va="center", fontweight="bold")
    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COVER, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_COVER}")


def draw_fig1() -> None:
    fig, ax = _canvas()
    rows = [
        (7.35, "memory", CYAN),
        (5.75, "comm", CYAN),
        (4.15, "sync", AMBER),
        (2.55, "launch", AMBER),
        (0.95, "overlap", CYAN),
    ]
    for y, label, color in rows:
        _box(ax, (1.1, y), 5.4, 1.15, color)
        ax.text(3.8, y + 0.58, label, color=BG, fontsize=22, ha="center", va="center", fontweight="bold")
    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG1, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_FIG1}")


if __name__ == "__main__":
    draw_cover()
    draw_fig1()
