"""Chart for drift_comparison.py: what the analysis looks like with drift left in."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drift_comparison import DRIFT_AXIS, miller, mm1887

SURFACE, TEXT_PRI, TEXT_SEC, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc"
RAW, COR, ART = "#eb6834", "#2a78d6", "#898781"
DOWN, UP = "#2a78d6", "#d64550"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0, colors=TEXT_SEC, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dial(ax, title):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], color=TEXT_SEC, fontsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(GRID)
    ax.set_title(title, color=TEXT_PRI, fontsize=11, fontweight="bold", pad=14)
    for a in DRIFT_AXIS.values():
        t = np.radians([a, a + 180])
        ax.plot(t, [1, 1], color=ART, linewidth=0)
        for tt in t:
            ax.plot([tt, tt], [0, 1], color=ART, linewidth=1.4, linestyle=(0, (4, 3)))


def needle(ax, az, length, color, lw=1.6, alpha=0.8):
    for tt in np.radians([az, az + 180]):
        ax.plot([tt, tt], [0, length], color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")


mm = mm1887()
sh = miller()

fig = plt.figure(figsize=(12.5, 11.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.3)

# (1) MM 1887 k=2 per turn
ax = fig.add_subplot(gs[0, 0])
style(ax)
x = np.arange(len(mm))
w = 0.27
ax.bar(x - w, [r[2][1] for r in mm], w - 0.03, color=RAW, label="drift left in", zorder=3)
ax.bar(x, [0.163 * abs(r[1]) for r in mm], w - 0.03, color=ART, label="drift alone would give (0.163 × drift)", zorder=3)
ax.bar(x + w, [r[4][1] for r in mm], w - 0.03, color=COR, label="drift removed", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([r[0].replace(" ", "\n", 2) for r in mm], fontsize=8.5)
ax.set_ylabel("k=2 amplitude (fringes)", color=TEXT_SEC, fontsize=10)
ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SEC, loc="upper right")
ax.set_title("1 · Michelson–Morley 1887: two-per-turn term, each turn",
             color=TEXT_PRI, fontsize=11.5, fontweight="bold", loc="left", pad=10)

# (2) MM 1887 axes
ax = fig.add_subplot(gs[0, 1], projection="polar")
dial(ax, "2 · 1887 axes, drift left in")
for r in mm:
    needle(ax, r[3], 0.9, DOWN if r[1] < 0 else UP, lw=2.2)
ax.text(0.5, -0.16, "blue = fringes slid down during the turn, red = slid up\n"
        "dashed = where a pure drift puts the axis (34° / 124°)",
        transform=ax.transAxes, ha="center", va="top", color=TEXT_SEC, fontsize=8.5)

# (3) Miller spectrum
ax = fig.add_subplot(gs[1, 0])
style(ax)
k = np.arange(1, 8)
sr = np.array([s["raw"] for s in sh]).mean(0)
sc = np.array([s["cor"] for s in sh]).mean(0)
ax.bar(k - 0.2, sr, 0.37, color=RAW, label="drift left in", zorder=3)
ax.bar(k + 0.2, sc, 0.37, color=COR, label="drift removed", zorder=3)
ax.set_xticks(k)
ax.set_xticklabels([f"k={i}" for i in k])
ax.set_ylabel("Mean amplitude, sheet averages (fringes)", color=TEXT_SEC, fontsize=10)
ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SEC, loc="upper right")
ax.set_title(f"3 · Miller 1925–26: harmonics of the {len(sh)} sheet averages",
             color=TEXT_PRI, fontsize=11.5, fontweight="bold", loc="left", pad=10)

# (4) Miller raw axes
ax = fig.add_subplot(gs[1, 1], projection="polar")
dial(ax, "4 · Miller axes, drift left in")
for s in sh:
    needle(ax, s["ax_raw"], min(s["raw"][1] / 0.12, 1), DOWN if s["drift"] < 0 else UP, lw=1.0, alpha=0.5)
ax.text(0.5, -0.16, "one needle per sheet, length = k=2 amplitude (rim = 0.12 fringe)\n"
        "blue = sheet drifted down, red = drifted up",
        transform=ax.transAxes, ha="center", va="top", color=TEXT_SEC, fontsize=8.5)

fig.suptitle("What the analysis looks like with the drift left in",
             x=0.06, ha="left", color=TEXT_PRI, fontsize=15, fontweight="bold", y=0.985)
fig.text(0.06, 0.955,
         "A steady slide of D fringes over one turn looks, to the Fourier fit, like a two-per-turn signal of 0.163·D "
         "pointing at 34° (sliding down) or 124° (sliding up).",
         color=TEXT_SEC, fontsize=9, style="italic", va="top")
out = Path(__file__).parent / "drift_comparison.png"
plt.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(out)
