"""Harmonic analysis of D. C. Miller's Mount Wilson data sheets (1925-26).

Data: transcribed sheets from the Aetherise project (MIT licence),
https://github.com/aetherise/aetherise  (dcm/csv, commit 1e37225).

Sheet format: 4 header lines (line 1 field 5 = Miller's sidereal time),
then one row per turn: 17 readings in tenths of a fringe (mark 1..16, then
mark 1 again) and an optional flag column. Only unflagged turns are used:
flags mark bad/cancelled turns, mid-turn fringe re-adjustments, etc.

For each turn: linear drift removed (end-to-start), then Fourier amplitudes
for k = 1..7 cycles per turn. An ether effect can only appear at even k,
essentially k = 2; k = 1 and k = 3 measure disturbances of the apparatus.
The same is done for each sheet's average of its clean turns (Miller's own
unit of analysis).
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "miller1925_data"
C = 299792458.0
L, LAMBDA, LAT = 32.0, 570e-9, 34.2           # Miller's Mt Wilson interferometer
N_AIR = 1.00029                               # sea-level; ~1.00024 at Mt Wilson's altitude
CMB = dict(ra=11.20, dec=-7.22, v=369e3)


def load_sheet(path):
    lines = [l.rstrip("\n") for l in path.read_text().splitlines() if l.strip()]
    head = lines[0].split(";")
    hh, mm = head[4].split(":")
    lst = int(hh) + int(mm) / 60
    turns = []
    for line in lines[4:]:
        tok = line.split(";")
        vals, flag = tok[:17], (tok[17] if len(tok) > 17 else "")
        if flag.strip():
            continue
        turns.append([int(v) for v in vals])
    return {"name": path.stem, "date": head[1], "lst": lst,
            "turns": np.array(turns, dtype=float) / 10.0}   # -> fringes


def detrend(y):
    return y - (y[16] - y[0]) * np.arange(17) / 16


def harmonics(y):
    z = detrend(y)[:16]
    F = np.fft.rfft(z - z.mean())
    return 2 * np.abs(F[1:8]) / 16, np.angle(F[2])


def cmb_amplitude(lst_h):
    H = np.radians((lst_h - CMB["ra"]) * 15)
    d, la = np.radians(CMB["dec"]), np.radians(LAT)
    alt = np.arcsin(np.sin(d) * np.sin(la) + np.cos(d) * np.cos(la) * np.cos(H))
    vh = CMB["v"] * np.cos(alt)
    return L * (N_AIR**2 - 1) * vh**2 / (C**2 * LAMBDA)


def analyse():
    sheets = [load_sheet(p) for p in sorted(DATA.glob("dcm_*.csv"))]
    sheets = [s for s in sheets if len(s["turns"]) >= 5]
    turn_amps, sheet_amps, sheet_rows = [], [], []
    for s in sheets:
        for t in s["turns"]:
            turn_amps.append(harmonics(t)[0])
        avg = np.mean([detrend(t) for t in s["turns"]], axis=0)
        a, _ = harmonics(avg)
        sheet_amps.append(a)
        sheet_rows.append((s["name"], s["lst"], len(s["turns"]), a[1], cmb_amplitude(s["lst"])))
    return sheets, np.array(turn_amps), np.array(sheet_amps), sheet_rows


if __name__ == "__main__":
    sheets, ta, sa, rows = analyse()
    print(f"{len(sheets)} sheets, {len(ta)} clean turns")
    hdr = "".join(f"   k={k:<4d}" for k in range(1, 8))
    print(f"{'':28s}{hdr}")
    print(f"{'single turns, mean amp':28s}" + "".join(f"  {a:.4f} " for a in ta.mean(0)))
    print(f"{'sheet averages, mean amp':28s}" + "".join(f"  {a:.4f} " for a in sa.mean(0)))
    print(f"{'  fraction of sheets where k is largest':28s}" +
          "".join(f"  {f:6.0%} " for f in np.bincount(sa.argmax(1), minlength=7) / len(sa)))
    obs, pred = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
    print(f"\nsheet-average k=2: mean {obs.mean():.4f}, median {np.median(obs):.4f} fringes")
    print(f"CMB prediction at the sheets' sidereal times: mean {pred.mean():.4f}, "
          f"range {pred.min():.4f}-{pred.max():.4f} fringes")
    print(f"correlation of observed k=2 with predicted amplitude over sidereal time: "
          f"{np.corrcoef(obs, pred)[0,1]:+.2f}")
