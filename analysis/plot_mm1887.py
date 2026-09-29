"""Re-plot Michelson & Morley (1887) observations against the Cahill-Kitto
CMB-dipole prediction for the same apparatus.

Top panel: raw per-turn readings (instrument drift dominates).
Bottom panel: the paper's "Final mean" (half-turns folded), with the linear
drift removed, versus the model prediction from cmb_pred.json.

Assumes mark 16 = North, marks advancing through East (as Fig. 6 is labelled),
and that the sign convention of the micrometer matches the model's
arm1-minus-arm2. The paper does not fix the latter, so the predicted curve's
phase is uncertain by 90 degrees of azimuth.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mm1887_table import EVENING, EVENING_FINAL, NOON, NOON_FINAL

HERE = Path(__file__).parent
pred = json.loads((HERE / "cmb_pred.json").read_text())

SURFACE, TEXT_PRI, TEXT_SEC, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc"
NOON_C, NOON_CMB_C = "#d64550", "#e2828a"
EVE_C, EVE_CMB_C = "#2a78d6", "#5b9bdb"
NOON_SHADES = ["#e2828a", "#d64550", "#9e2a33"]
EVE_SHADES = ["#5b9bdb", "#2a78d6", "#1a4f91"]


def drift_corrected(final):
    f = np.array(final)
    c = f - (f[8] - f[0]) * np.arange(9) / 8
    return c - c[:8].mean()


def second_harmonic(c):
    x = 2 * np.radians(np.arange(8) * 22.5)
    a = 2 / 8 * np.sum(c[:8] * np.cos(x))
    b = 2 / 8 * np.sum(c[:8] * np.sin(x))
    return np.hypot(a, b), (np.degrees(np.arctan2(b, a)) / 2) % 180


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0, colors=TEXT_SEC, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 11), dpi=200,
                               gridspec_kw={"height_ratios": [1, 1.25]})
fig.patch.set_facecolor(SURFACE)

# --- raw readings ---
style(ax1)
az_full = np.arange(17) * 22.5
for (day, row), col in zip(NOON.items(), NOON_SHADES):
    y = np.array(row) / 50
    ax1.plot(az_full, y, color=col, linewidth=2, marker="o", markersize=4)
    ax1.annotate(f"{day} noon", (360, y[-1]), xytext=(6, 0), textcoords="offset points",
                 color=TEXT_SEC, fontsize=9, va="center")
for (day, row), col in zip(EVENING.items(), EVE_SHADES):
    y = np.array(row) / 50
    ax1.plot(az_full, y, color=col, linewidth=2, marker="o", markersize=4)
    ax1.annotate(f"{day} eve", (360, y[-1]), xytext=(6, 0), textcoords="offset points",
                 color=TEXT_SEC, fontsize=9, va="center")
ax1.set_xlim(-8, 400)
ax1.set_xticks([0, 90, 180, 270, 360])
ax1.set_xticklabels(["N", "E", "S", "W", "N"])
ax1.set_ylabel("Raw micrometer reading (wavelengths)", color=TEXT_SEC, fontsize=10)
ax1.set_title("1 · Raw readings, one full turn each: slow drift dwarfs any signal",
              color=TEXT_PRI, fontsize=12.5, fontweight="bold", loc="left", pad=10)

# --- drift-corrected final means vs CMB prediction ---
style(ax2)
th = np.arange(9) * 22.5
noon = drift_corrected(NOON_FINAL)
eve = drift_corrected(EVENING_FINAL)
An, pn = second_harmonic(noon)
Ae, pe = second_harmonic(eve)
Apn, ppn = second_harmonic(np.array(pred["noon"]))
Ape, ppe = second_harmonic(np.array(pred["evening"]))

ax2.axhline(0, color=GRID, linewidth=1.4)
ax2.plot(th, noon, color=NOON_C, linewidth=2.4, marker="o", markersize=8,
         markeredgecolor=SURFACE, label=f"Noon, observed: A = {An:.4f}λ, peak {pn:.0f}°")
ax2.plot(th, eve, color=EVE_C, linewidth=2.4, marker="o", markersize=8,
         markeredgecolor=SURFACE, label=f"Evening, observed: A = {Ae:.4f}λ, peak {pe:.0f}°")
ax2.plot(th, pred["noon"], color=NOON_CMB_C, linewidth=2, linestyle=(0, (6, 3)),
         marker="D", markersize=6, markerfacecolor=SURFACE, markeredgecolor=NOON_CMB_C,
         label=f"Noon, CMB prediction: A = {Apn:.4f}λ, peak {ppn:.0f}°")
ax2.plot(th, pred["evening"], color=EVE_CMB_C, linewidth=2, linestyle=(0, (6, 3)),
         marker="D", markersize=6, markerfacecolor=SURFACE, markeredgecolor=EVE_CMB_C,
         label=f"Evening, CMB prediction: A = {Ape:.4f}λ, peak {ppe:.0f}°")
ax2.set_xlim(-5, 185)
ax2.set_ylim(-0.022, 0.022)
ax2.set_xticks([0, 45, 90, 135, 180])
ax2.set_xticklabels(["N (0°)", "NE/SW", "E, W (90°)", "SE/NW", "S, N (180°)"])
ax2.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
ax2.set_yticklabels(["-0.02λ", "-0.01λ", "0", "+0.01λ", "+0.02λ"])
ax2.set_xlabel("Azimuth of mark (half-turns folded, as in the paper's 'Final mean')",
               color=TEXT_SEC, fontsize=10, labelpad=8)
ax2.set_ylabel("Fringe displacement, linear drift removed", color=TEXT_SEC, fontsize=10)
ax2.set_title("2 · Final means (drift removed) vs Cahill–Kitto CMB-dipole prediction",
              color=TEXT_PRI, fontsize=12.5, fontweight="bold", loc="left", pad=10)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9.5,
           frameon=False, labelcolor=TEXT_SEC)

fig.suptitle("Michelson–Morley 1887, actual data (Am. J. Sci. 34, p. 340)",
             x=0.07, ha="left", color=TEXT_PRI, fontsize=15, fontweight="bold")
fig.text(0.07, 0.945,
         "Prediction: 11 m arms, λ=590 nm, air n=1.00029, lat 41.5°N, CMB apex RA 11.20h Dec −7.22°, "
         "369 km/s,\nat sidereal time 07:00 (noon) and 13:00 (evening), as in Cahill's table. "
         "Phase assumes mark 16 = N and model sign convention.",
         color=TEXT_SEC, fontsize=8.8, style="italic", va="top")
plt.tight_layout(rect=[0, 0.02, 1, 0.925])
out = HERE / "mm1887_actual_vs_cmb.png"
plt.savefig(out, facecolor=SURFACE)
print(f"obs noon A={An:.4f} pk={pn:.1f} | eve A={Ae:.4f} pk={pe:.1f} || pred noon A={Apn:.4f} pk={ppn:.1f} | eve A={Ape:.4f} pk={ppe:.1f}")
