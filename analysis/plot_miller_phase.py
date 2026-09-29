"""Phase (signal-axis) analysis of Miller's 1925-26 Mount Wilson sheets.

For each sheet: average its clean, drift-corrected turns and take the k=2
term's axis -- the arm-1 azimuth where the two-per-turn wiggle peaks, which
is where the model puts the horizontal wind axis (mod 180 deg).

Top row: observed axis of every sheet, per epoch, drawn as a compass needle
(length = amplitude). Middle row: the CMB-dipole prediction for the same
sheets' sidereal times. Bottom: axis vs sidereal time, 2-hour vector means.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from miller1925 import DATA, cmb_amplitude, cmb_axis, detrend, harmonics, load_sheet

SURFACE, TEXT_PRI, TEXT_SEC, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc"
DESK_NW, DESK_SW, PRED = "#2a78d6", "#eb6834", "#898781"
EPOCHS = [("Apr 1925", ("1925-03", "1925-04"), "#2a78d6"),
          ("Aug 1925", ("1925-08",), "#eb6834"),
          ("Sep 1925", ("1925-09",), "#1baf7a"),
          ("Feb 1926", ("1926-02",), "#c98500")]

sheets = []
for p in sorted(DATA.glob("dcm_*.csv")):
    s = load_sheet(p)
    if len(s["turns"]) < 5:
        continue
    amps, axis = harmonics(np.mean([detrend(t) for t in s["turns"]], axis=0))
    s.update(amp=amps[1], axis=axis, pred_axis=float(cmb_axis(s["lst"])),
             pred_amp=float(cmb_amplitude(s["lst"])))
    sheets.append(s)


def mean_axis(axes, w=None):
    z = np.average(np.exp(2j * np.radians(axes)), weights=w)
    return (np.degrees(np.angle(z)) / 2) % 180, abs(z)


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
    ax.set_title(title, color=TEXT_PRI, fontsize=10.5, fontweight="bold", pad=12)


def needle(ax, az, length, color, lw, alpha=1.0):
    t = np.radians([az, az + 180])
    ax.plot(t, [length, length], color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")
    ax.plot([t[0], t[0]], [0, length], color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")
    ax.plot([t[1], t[1]], [0, length], color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")


SCALE = 0.12  # fringes at the rim
fig = plt.figure(figsize=(13.5, 14), dpi=200)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15], hspace=0.5, wspace=0.5)

for col, (name, prefixes, _) in enumerate(EPOCHS):
    group = [s for s in sheets if s["date"][:7] in prefixes]
    obs_ax = fig.add_subplot(gs[0, col], projection="polar")
    pred_ax = fig.add_subplot(gs[1, col], projection="polar")
    dial(obs_ax, f"{name} · observed")
    dial(pred_ax, f"{name} · CMB prediction")
    for s in group:
        needle(obs_ax, s["axis"], min(s["amp"] / SCALE, 1), DESK_SW if s["desk_sw"] else DESK_NW, 1.1, 0.55)
        needle(pred_ax, s["pred_axis"], min(s["pred_amp"] / SCALE, 1), PRED, 1.1, 0.55)
    m, R = mean_axis([s["axis"] for s in group])
    pm, pR = mean_axis([s["pred_axis"] for s in group])
    needle(obs_ax, m, 0.95, TEXT_PRI, 2.6)
    needle(pred_ax, pm, 0.95, TEXT_PRI, 2.6)
    obs_ax.text(0.5, -0.14, f"{len(group)} sheets\nmean axis {m:.0f}°, R = {R:.2f}",
                transform=obs_ax.transAxes, va="top", ha="center", color=TEXT_SEC, fontsize=8.5)
    pred_ax.text(0.5, -0.14, f"mean axis {pm:.0f}°, R = {pR:.2f}",
                 transform=pred_ax.transAxes, va="top", ha="center", color=TEXT_SEC, fontsize=8.5)

# --- axis vs sidereal time, 2-hour vector means ---
ax = fig.add_subplot(gs[2, :])
ax.set_facecolor(SURFACE)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(GRID)
ax.tick_params(length=0, colors=TEXT_SEC, labelsize=9.5)
ax.grid(axis="y", color=GRID, linewidth=0.8)

t = np.linspace(0, 24, 600)
pa = cmb_axis(t)
pa_plot = pa.copy()
pa_plot[np.r_[False, np.abs(np.diff(pa)) > 90]] = np.nan
ax.plot(t, pa_plot, color=TEXT_PRI, linewidth=2, linestyle=(0, (6, 3)), label="CMB-dipole prediction", zorder=2)
for name, prefixes, colr in EPOCHS:
    group = [s for s in sheets if s["date"][:7] in prefixes]
    xs, ys = [], []
    for b in range(12):
        in_bin = [s for s in group if 2 * b <= s["lst"] < 2 * b + 2]
        if len(in_bin) >= 2:
            xs.append(2 * b + 1)
            ys.append(mean_axis([s["axis"] for s in in_bin])[0])
    ax.plot(xs, ys, color=colr, marker="o", markersize=8, markeredgecolor=SURFACE,
            linewidth=0, label=f"{name} observed", zorder=3)
ax.set_xlim(0, 24)
ax.set_ylim(0, 180)
ax.set_xticks(range(0, 25, 3))
ax.set_yticks([0, 45, 90, 135, 180])
ax.set_yticklabels(["0° N", "45° NE", "90° E", "135° SE", "180° S"])
ax.set_xlabel("Sidereal time (h)", color=TEXT_SEC, fontsize=10)
ax.set_ylabel("Signal axis azimuth (mod 180°)", color=TEXT_SEC, fontsize=10)
ax.set_title("Signal axis vs sidereal time (2-hour means, bins with ≥2 sheets)",
             color=TEXT_PRI, fontsize=11.5, fontweight="bold", loc="left", pad=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False,
          fontsize=9.5, labelcolor=TEXT_SEC)

fig.suptitle("Miller 1925–26: which way does the two-per-turn signal point?",
             x=0.06, ha="left", color=TEXT_PRI, fontsize=15, fontweight="bold", y=0.975)
fig.text(0.06, 0.953,
         "Each thin needle is one data sheet (~17 turns averaged); length = k=2 amplitude (rim = 0.12 fringe); "
         "bold needle = mean direction.\nObserved needles: blue = recording desk in NW corner, orange = desk in SW corner. "
         "Axis is 180°-ambiguous. R = 1 means every sheet points the same way.",
         color=TEXT_SEC, fontsize=8.8, style="italic", va="top")
out = Path(__file__).parent / "miller1925_phase.png"
plt.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(out)
