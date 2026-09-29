"""Phase analysis of the 1887 data with the drift LEFT IN.

Top: compass dials for noon and evening -- each turn's k=2 axis (thin,
coloured by which way the fringes slid), the session-average axis (bold), the
CMB-dipole prediction (dashed) and the same prediction with the paper's
unknown sign convention flipped (dotted, i.e. rotated 90 degrees).
Bottom: the paper's folded "Final mean" curves, centred but NOT drift-corrected,
against the prediction.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drift_comparison import DRIFT_AXIS, fit
from mm1887_table import EVENING, EVENING_FINAL, EVENING_MEAN, NOON, NOON_FINAL, NOON_MEAN

HERE = Path(__file__).parent
pred = json.loads((HERE / "cmb_pred.json").read_text())
SURFACE, TEXT_PRI, TEXT_SEC, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc"
DOWN, UP, PRED = "#1baf7a", "#c98500", "#52514e"
NOON_C, NOON_CMB_C = "#d64550", "#e2828a"
EVE_C, EVE_CMB_C = "#2a78d6", "#5b9bdb"


def pred_axis(values):
    v = np.array(values[:8])
    return fit(np.r_[v, v, v[0]], False)


def dial(ax, title):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], color=TEXT_SEC, fontsize=9.5)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(GRID)
    ax.set_title(title, color=TEXT_PRI, fontsize=12, fontweight="bold", pad=14)


def needle(ax, az, length, color, lw, ls="-", alpha=1.0, label=None):
    for i, tt in enumerate(np.radians([az, az + 180])):
        ax.plot([tt, tt], [0, length], color=color, linewidth=lw, linestyle=ls, alpha=alpha,
                solid_capstyle="round", label=label if i == 0 else None)


fig = plt.figure(figsize=(12, 12.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.38, wspace=0.35)

lines = []
for col, (name, table, mean, key) in enumerate((("Noon", NOON, NOON_MEAN, "noon"),
                                                ("Evening", EVENING, EVENING_MEAN, "evening"))):
    ax = fig.add_subplot(gs[0, col], projection="polar")
    dial(ax, f"{name} · drift left in")
    _, pax = pred_axis(pred[key])
    needle(ax, pax, 1.0, PRED, 2, ls=(0, (6, 3)), label="CMB prediction")
    needle(ax, (pax + 90) % 180, 1.0, PRED, 1.6, ls=(0, (1, 2.5)), label="CMB prediction, sign flipped")
    for day, row in table.items():
        y = np.array(row) / 50
        a, axis = fit(y, False)
        needle(ax, axis, min(a[1] / 0.11, 1), DOWN if y[16] < y[0] else UP, 2.2, alpha=0.75)
        ax.text(np.radians(axis), min(a[1] / 0.11, 1) + 0.08, day.replace("July ", "Jul "),
                color=TEXT_SEC, fontsize=8, ha="center", va="center")
    m = np.array(mean) / 50
    a, axis = fit(m, False)
    needle(ax, axis, 0.75, TEXT_PRI, 3.2, label="session average")
    diff = ((axis - pax + 90) % 180) - 90
    ax.text(0.5, -0.13, f"session average {axis:.0f}° vs prediction {pax:.0f}° ({diff:+.0f}°)\n"
            f"drift over the averaged turn: {m[16] - m[0]:+.2f} fringe",
            transform=ax.transAxes, ha="center", va="top", color=TEXT_SEC, fontsize=9)
    if col == 0:
        handles, labels = ax.get_legend_handles_labels()

# --- folded curves, centred but not drift-corrected ---
ax = fig.add_subplot(gs[1, :])
ax.set_facecolor(SURFACE)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(GRID)
ax.tick_params(length=0, colors=TEXT_SEC, labelsize=9.5)
ax.grid(axis="y", color=GRID, linewidth=0.8)
th = np.arange(9) * 22.5
for final, colr, lab in ((NOON_FINAL, NOON_C, "Noon, observed (drift left in)"),
                         (EVENING_FINAL, EVE_C, "Evening, observed (drift left in)")):
    f = np.array(final)
    ax.plot(th, f - f[:8].mean(), color=colr, linewidth=2.4, marker="o", markersize=8,
            markeredgecolor=SURFACE, label=lab)
ax.plot(th, pred["noon"], color=NOON_CMB_C, linewidth=2, linestyle=(0, (6, 3)), marker="D",
        markersize=6, markerfacecolor=SURFACE, markeredgecolor=NOON_CMB_C, label="Noon, CMB prediction")
ax.plot(th, pred["evening"], color=EVE_CMB_C, linewidth=2, linestyle=(0, (6, 3)), marker="D",
        markersize=6, markerfacecolor=SURFACE, markeredgecolor=EVE_CMB_C, label="Evening, CMB prediction")
ax.axhline(0, color=GRID, linewidth=1.4)
ax.set_xlim(-5, 185)
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_xticklabels(["N (0°)", "NE/SW", "E, W (90°)", "SE/NW", "S, N (180°)"])
ax.set_ylabel("Fringe displacement, centred (drift NOT removed)", color=TEXT_SEC, fontsize=10)
ax.set_xlabel("Azimuth of mark (half-turns folded, the paper's 'Final mean')", color=TEXT_SEC, fontsize=10)
ax.set_title("Final means with the drift left in: the 0° and 180° ends no longer meet",
             color=TEXT_PRI, fontsize=11.5, fontweight="bold", loc="left", pad=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
          fontsize=9.5, labelcolor=TEXT_SEC)

fig.legend(handles + [plt.Line2D([], [], color=DOWN, lw=2.2), plt.Line2D([], [], color=UP, lw=2.2)],
           labels + ["turn where fringes slid down", "turn where fringes slid up"],
           loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5, frameon=False,
           fontsize=9, labelcolor=TEXT_SEC)
fig.suptitle("Michelson–Morley 1887: phase analysis with the drift left in",
             x=0.06, ha="left", color=TEXT_PRI, fontsize=15, fontweight="bold", y=0.985)
fig.text(0.06, 0.955,
         f"A steady drift alone points the axis at {DRIFT_AXIS[-1]:.0f}° (sliding down) or {DRIFT_AXIS[+1]:.0f}° "
         "(sliding up). At these sidereal times those are within 10° of the CMB-predicted axes,\n"
         "so with the drift left in, each turn 'agrees' or 'disagrees' with the prediction depending on which way "
         "its fringes happened to slide. Needle length = k=2 amplitude (rim = 0.11 fringe).",
         color=TEXT_SEC, fontsize=8.8, style="italic", va="top")
out = HERE / "mm1887_raw_phase.png"
fig.subplots_adjust(top=0.86)
plt.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(out)
