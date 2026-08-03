#!/usr/bin/env python3
"""Generate C-01 cover: SMEM path vs SHFL path (article thesis)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "article" / "03_compute_primitives" / "assets" / "C-01-warp-primitives-cover.png"

BG = "#0b1220"
CYAN = "#39c5cf"
AMBER = "#f0a202"
ORANGE = "#ff8c42"
MUTED = "#8b949e"
PANEL = "#121a2b"
GRID = "#1e2a3d"
FG = "#e6edf3"


def _cjk_font() -> font_manager.FontProperties:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ]
    for p in candidates:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            return font_manager.FontProperties(fname=str(p))
    raise SystemExit("No CJK font found for cover text")


def main() -> None:
    prop = _cjk_font()
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_axis_off()
    ax.set_facecolor(BG)

    for x in np.arange(0, 16.1, 0.5):
        ax.plot([x, x], [0, 9], color=GRID, lw=0.35, alpha=0.3, zorder=0)
    for y in np.arange(0, 9.1, 0.5):
        ax.plot([0, 16], [y, y], color=GRID, lw=0.35, alpha=0.3, zorder=0)

    def text(x, y, s, **kw):
        ax.text(x, y, s, fontproperties=prop, **kw)

    text(
        0.45,
        8.35,
        "C-01  Warp Primitives",
        color=AMBER,
        fontsize=28,
        fontweight="bold",
        va="center",
    )
    text(
        0.45,
        7.72,
        "同 warp 交换：SMEM 树  vs  SHFL 寄存器直连    |    *_sync + 逻辑 mask",
        color=CYAN,
        fontsize=13,
        va="center",
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.4, 1.55),
            7.2,
            5.7,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=PANEL,
            edgecolor=CYAN,
            lw=1.6,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (8.4, 1.55),
            7.2,
            5.7,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=PANEL,
            edgecolor=AMBER,
            lw=1.6,
            zorder=1,
        )
    )
    text(4.0, 6.85, "SMEM 路径", color=CYAN, fontsize=16, fontweight="bold", ha="center")
    text(4.0, 6.42, "STS  →  __syncthreads  →  LDS", color=MUTED, fontsize=10, ha="center")
    text(12.0, 6.85, "SHFL 路径", color=AMBER, fontsize=16, fontweight="bold", ha="center")
    text(
        12.0,
        6.42,
        "__shfl_down_sync（寄存器直连）",
        color=MUTED,
        fontsize=10,
        ha="center",
    )

    def draw_threads(x0, y, n=8, color=ORANGE):
        w, h, gap = 0.55, 0.35, 0.12
        xs = []
        for i in range(n):
            x = x0 + i * (w + gap)
            ax.add_patch(
                Rectangle(
                    (x, y), w, h, facecolor=color, edgecolor="none", zorder=3, alpha=0.92
                )
            )
            xs.append(x + w / 2)
        return xs

    tx = draw_threads(1.0, 5.7)
    for x in tx:
        ax.annotate(
            "",
            xy=(x, 4.85),
            xytext=(x, 5.65),
            arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.2),
            zorder=2,
        )
    text(4.0, 5.32, "STS", color=CYAN, fontsize=9, ha="center")
    ax.add_patch(
        FancyBboxPatch(
            (1.0, 4.15),
            6.0,
            0.55,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            facecolor="#1a3a4a",
            edgecolor=CYAN,
            lw=1.2,
            zorder=3,
        )
    )
    text(4.0, 4.42, "Shared Memory", color=FG, fontsize=11, ha="center", va="center")
    ax.add_patch(
        FancyBboxPatch(
            (1.3, 3.35),
            5.4,
            0.45,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor="#2a2030",
            edgecolor=AMBER,
            lw=1.3,
            zorder=3,
        )
    )
    text(
        4.0,
        3.57,
        "__syncthreads",
        color=AMBER,
        fontsize=11,
        ha="center",
        va="center",
        fontweight="bold",
    )
    for x in tx:
        ax.annotate(
            "",
            xy=(x, 2.55),
            xytext=(x, 3.3),
            arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.2),
            zorder=2,
        )
    text(4.0, 2.92, "LDS", color=CYAN, fontsize=9, ha="center")
    draw_threads(1.0, 1.95)
    text(4.0, 1.70, "指令多 · 吃 SMEM/bank · 跨 warp 仍需要", color=MUTED, fontsize=9, ha="center")

    n = 8
    w, h, gap = 0.55, 0.35, 0.12
    x0 = 9.15
    tops = []
    for i in range(n):
        x = x0 + i * (w + gap)
        ax.add_patch(
            Rectangle((x, 5.55), w, h, facecolor=ORANGE, edgecolor="none", zorder=3, alpha=0.92)
        )
        tops.append(x + w / 2)
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        ax.annotate(
            "",
            xy=(tops[a], 4.85),
            xytext=(tops[b], 5.5),
            arrowprops=dict(
                arrowstyle="->", color=CYAN, lw=1.4, connectionstyle="arc3,rad=0.25"
            ),
            zorder=2,
        )
    text(12.0, 5.12, "shfl_down  offset=1", color=MUTED, fontsize=8, ha="center")
    ax.annotate(
        "",
        xy=(tops[0], 4.15),
        xytext=(tops[2], 4.75),
        arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.4, connectionstyle="arc3,rad=0.2"),
        zorder=2,
    )
    ax.annotate(
        "",
        xy=(tops[4], 4.15),
        xytext=(tops[6], 4.75),
        arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.4, connectionstyle="arc3,rad=0.2"),
        zorder=2,
    )
    text(12.0, 4.38, "shfl_down  offset=2 …", color=MUTED, fontsize=8, ha="center")
    ax.annotate(
        "",
        xy=(tops[0], 3.35),
        xytext=(tops[4], 4.05),
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.8, connectionstyle="arc3,rad=0.15"),
        zorder=2,
    )
    text(
        12.0,
        3.52,
        "结果落在 lane0（block 级再用少量 SMEM 收每 warp）",
        color=MUTED,
        fontsize=8,
        ha="center",
    )
    ax.add_patch(Rectangle((x0, 2.55), w, h, facecolor=AMBER, edgecolor="none", zorder=3))
    text(x0 + w / 2, 2.28, "lane0", color=AMBER, fontsize=9, ha="center")
    for i in range(1, n):
        x = x0 + i * (w + gap)
        ax.add_patch(
            Rectangle((x, 2.55), w, h, facecolor="#3a4558", edgecolor="none", zorder=3, alpha=0.75)
        )
    text(12.0, 1.70, "少指令 · 少共享往返 · 本机大 block ~1.27×", color=MUTED, fontsize=9, ha="center")

    ax.add_patch(Rectangle((0, 0), 16, 1.25, facecolor="#091018", zorder=4))
    text(
        8.0,
        0.78,
        "主结论：GMEM 主导时 ≈持平；nwarps↑ 后 SHFL 少指令/少 SMEM（NCU inst≈0.58×）",
        color=FG,
        fontsize=12,
        ha="center",
        va="center",
        zorder=5,
    )
    text(
        8.0,
        0.32,
        "mask 用程序逻辑，勿滥用 __activemask()   ·   float 无硬件 reduce，int 可试 __reduce_*_sync",
        color=MUTED,
        fontsize=9,
        ha="center",
        va="center",
        zorder=5,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
