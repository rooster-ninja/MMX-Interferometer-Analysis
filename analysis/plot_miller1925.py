"""Plot the Miller 1925-26 harmonic analysis (see miller1925.py)."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from miller1925 import DATA, cmb_amplitude, detrend, harmonics, load_sheet
from mm1887_table import EVENING, NOON

SURFACE, TEXT_PRI, TEXT_SEC, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc"
EPOCHS = [("Mar–Apr 1925", ("1925-03", "1925-04"), "#2a78d6", "o"),
          ("Aug 1925", ("1925-08",), "#eb6834", "s"),
          ("Sep 1925", ("1925-09",), "#1baf7a", "^"),
          ("Feb 1926", ("1926-02",), "#c98500", "D")]

sheets, turn_amps, sheet_amps = [], [], []
for p in sorted(DATA.glob("dcm_*.csv")):
    s = load_sheet(p)
    if len(s["turns"]) < 5:
        continue
    hh, mm = p.read_text().splitlines()[0].split(";")[2].split(":")
    s["civil"] = int(hh) + int(mm) / 60
    for t in s["turns"]:
        turn_amps.append(harmonics(t)[0])
    s["k2"] = harmonics(np.mean([detrend(t) for t in s["turns"]], axis=0))[0][1]
    sheet_amps.append(harmonics(np.mean([detrend(t) for t in s["turns"]], axis=0))[0])
    sheets.append(s)
turn_amps, sheet_amps = np.array(turn_amps), np.array(sheet_amps)

mm_turns = []
for rows in (NOON, EVENING):
    for r in rows.values():
        y = np.array(r) / 50
        y = y - (y[16] - y[0]) * np.arange(17) / 16
        mm_turns.append(2 * np.abs(np.fft.rfft(y[:16] - y[:16].mean()))[1:8] / 16)
mm_turns = np.array(mm_turns)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0, colors=TEXT_SEC, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(11, 12.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.32, wspace=0.12)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)

# --- harmonic spectrum ---
style(ax1)
k = np.arange(1, 8)
w = 0.26
for off, data, col, lab in [(-w, turn_amps.mean(0), "#8fb8e8", f"Miller, single turns (n={len(turn_amps)})"),
                            (0, sheet_amps.mean(0), "#2a78d6", f"Miller, sheet averages (n={len(sheet_amps)}, ~17 turns each)"),
                            (w, mm_turns.mean(0), "#eb6834", "Michelson–Morley 1887, single turns (n=6)")]:
    ax1.bar(k + off, data, width=w - 0.03, color=col, label=lab, zorder=3)
ax1.set_xticks(k)
ax1.set_xticklabels([f"k={i}" + ("\n(ether term)" if i == 2 else "") for i in k])
ax1.set_ylabel("Mean amplitude (fringes)", color=TEXT_SEC, fontsize=10)
ax1.legend(frameon=False, fontsize=9.5, labelcolor=TEXT_SEC, loc="upper right")
ax1.set_title("1 · Harmonics per turn (drift removed): Miller's k=2 survives averaging, k=3+ noise does not",
              color=TEXT_PRI, fontsize=11.5, fontweight="bold", loc="left", pad=10)

# --- k=2 vs sidereal and solar time ---
lst_grid = np.linspace(0, 24, 300)
for ax, key, xlabel, title in [(ax2, "lst", "Sidereal time (h)", "2 · Sheet k=2 amplitude vs sidereal time"),
                               (ax3, "civil", "Local solar (clock) time (h)", "3 · …vs solar time")]:
    style(ax)
    for name, prefixes, col, mk in EPOCHS:
        pts = [s for s in sheets if s["date"][:7] in prefixes]
        ax.scatter([s[key] for s in pts], [s["k2"] for s in pts], color=col, marker=mk, s=34,
                   edgecolor=SURFACE, linewidth=0.8, label=name, zorder=3)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel(xlabel, color=TEXT_SEC, fontsize=10)
    ax.set_title(title, color=TEXT_PRI, fontsize=11.5, fontweight="bold", loc="left", pad=10)
ax2.plot(lst_grid, cmb_amplitude(lst_grid), color=TEXT_PRI, linewidth=2, linestyle=(0, (6, 3)),
         label="CMB-dipole prediction", zorder=4)
ax2.set_ylabel("k=2 amplitude of sheet average (fringes)", color=TEXT_SEC, fontsize=10)
plt.setp(ax3.get_yticklabels(), visible=False)
h, l = ax2.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
           labelcolor=TEXT_SEC, bbox_to_anchor=(0.5, 0.035))

fig.suptitle("Dayton Miller, Mount Wilson 1925–26: 150 data sheets, 2,498 clean turns",
             x=0.07, ha="left", color=TEXT_PRI, fontsize=15, fontweight="bold", y=0.985)
fig.text(0.07, 0.955,
         "Data: Aetherise transcription of Miller's sheets (github.com/aetherise/aetherise). Readings in tenths of a fringe; "
         "flagged turns dropped.\nPrediction: Cahill–Kitto, 32 m arms, λ=570 nm, lat 34.2°N, n=1.00029 (sea level; "
         "~15% lower at Mt Wilson), CMB 369 km/s toward RA 11.20h Dec −7.22°.",
         color=TEXT_SEC, fontsize=8.6, style="italic", va="top")
out = Path(__file__).parent / "miller1925_harmonics.png"
fig.subplots_adjust(bottom=0.11)
plt.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(out)
