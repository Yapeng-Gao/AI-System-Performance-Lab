#!/usr/bin/env python3
"""Plot B-03 register-pressure results.

Reads:
  docs/results/B-03_modes.csv

Writes:
  article/02_memory_optim/assets/B-03-mode-ms-bars.png
  article/02_memory_optim/assets/B-03-speedup-vs-baseline.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODES_CSV = ROOT / "docs" / "results" / "B-03_modes.csv"
OUT_DIR = ROOT / "article" / "02_memory_optim" / "assets"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"
GREEN = "#3fb950"
MUTED = "#8b949e"

MODE_ORDER = ["baseline", "highreg", "launch_bounds"]
MODE_COLOR = {"baseline": GREEN, "highreg": AMBER, "launch_bounds": CYAN}


def _style_axes(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, which="both", ls="--", color=GRID, alpha=0.7)


def load_modes(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mode = (row.get("mode") or "").strip()
            if not mode or mode.startswith("#"):
                continue
            try:
                rows.append(
                    {
                        "mode": mode,
                        "median_ms": float(row["median_ms"]),
                        "speedup_vs_baseline": float(row["speedup_vs_baseline"]),
                    }
                )
            except (KeyError, ValueError) as e:
                print(f"skip row {row!r}: {e}", file=sys.stderr)
    return rows


def main() -> int:
    if not MODES_CSV.is_file():
        print(f"missing {MODES_CSV}", file=sys.stderr)
        return 1
    rank = {m: i for i, m in enumerate(MODE_ORDER)}
    rows = sorted(load_modes(MODES_CSV), key=lambda r: rank.get(str(r["mode"]), 100))
    if not rows:
        print(f"no data rows in {MODES_CSV}", file=sys.stderr)
        return 2

    labels = [str(r["mode"]) for r in rows]
    ms = [float(r["median_ms"]) for r in rows]
    sp = [float(r["speedup_vs_baseline"]) for r in rows]
    colors = [MODE_COLOR.get(m, CYAN) for m in labels]

    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=BG)
    _style_axes(ax)
    ax.bar(labels, ms, color=colors, edgecolor=GRID)
    ax.set_ylabel("median (ms)")
    ax.set_title("B-03: register-pressure modes (median)")
    for i, v in enumerate(ms):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", color=FG, fontsize=9)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out1 = OUT_DIR / "B-03-mode-ms-bars.png"
    fig.savefig(out1, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out1}")

    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=BG)
    _style_axes(ax)
    ax.bar(labels, sp, color=colors, edgecolor=GRID)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.2)
    ax.set_ylabel("speedup (baseline_ms / mode_ms)")
    ax.set_title("B-03: speedup vs baseline")
    for i, v in enumerate(sp):
        ax.text(i, v, f"{v:.2f}x", ha="center", va="bottom", color=FG, fontsize=9)
    fig.tight_layout()
    out2 = OUT_DIR / "B-03-speedup-vs-baseline.png"
    fig.savefig(out2, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
