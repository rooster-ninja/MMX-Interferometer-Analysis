"""Re-run the 1887 and Miller analyses with the drift left in, next to the
drift-removed versions.

A steady slide of D fringes over one turn, fed to the same Fourier fit,
produces a k=2 term of 0.163*|D| whose axis is fixed at 33.75 deg (sliding
down) or 123.75 deg (sliding up) -- set by the start mark, not the sky.
"""
import numpy as np

from miller1925 import DATA, cmb_axis, load_sheet
from mm1887_table import EVENING, NOON

DRIFT_AXIS = {+1: 123.75, -1: 33.75}


def fit(y, remove_drift):
    y = np.asarray(y, dtype=float)
    if remove_drift:
        y = y - (y[16] - y[0]) * np.arange(17) / 16
    z = y[:16]
    F = np.fft.rfft(z - z.mean())
    return 2 * np.abs(F[1:8]) / 16, (-np.degrees(np.angle(F[2])) / 2) % 180


def mean_axis(axes):
    z = np.mean(np.exp(2j * np.radians(axes)))
    return (np.degrees(np.angle(z)) / 2) % 180, abs(z)


def circ_corr(a, b):
    a, b = 2 * np.radians(np.asarray(a)), 2 * np.radians(np.asarray(b))
    sa, sb = np.sin(a[:, None] - a[None, :]), np.sin(b[:, None] - b[None, :])
    return np.sum(sa * sb) / np.sqrt(np.sum(sa**2) * np.sum(sb**2))


def mm1887():
    rows = []
    for session, table in (("noon", NOON), ("evening", EVENING)):
        for day, r in table.items():
            y = np.array(r) / 50
            raw, ax_raw = fit(y, False)
            cor, ax_cor = fit(y, True)
            rows.append((f"{day} {session}", y[16] - y[0], raw, ax_raw, cor, ax_cor))
    return rows


def miller():
    out = []
    for p in sorted(DATA.glob("dcm_*.csv")):
        s = load_sheet(p)
        if len(s["turns"]) < 5:
            continue
        avg = s["turns"].mean(axis=0)              # raw average turn
        raw, ax_raw = fit(avg, False)
        avg_c = np.mean([t - (t[16] - t[0]) * np.arange(17) / 16 for t in s["turns"]], axis=0)
        cor, ax_cor = fit(avg_c, True)
        turn_raw = np.array([fit(t, False)[0] for t in s["turns"]])
        turn_cor = np.array([fit(t, True)[0] for t in s["turns"]])
        out.append(dict(s, drift=avg[16] - avg[0], raw=raw, ax_raw=ax_raw, cor=cor, ax_cor=ax_cor,
                        turn_raw=turn_raw, turn_cor=turn_cor, pred=float(cmb_axis(s["lst"]))))
    return out


if __name__ == "__main__":
    print("=== Michelson-Morley 1887, per turn ===")
    print(f"{'turn':16s}{'drift':>8s} | {'raw k=1':>8s}{'k=2':>8s}{'k=3':>8s}{'axis':>7s} | {'corr k=1':>9s}{'k=2':>8s}{'k=3':>8s}{'axis':>7s}")
    for name, d, raw, axr, cor, axc in mm1887():
        print(f"{name:16s}{d:+8.3f} | {raw[0]:8.4f}{raw[1]:8.4f}{raw[2]:8.4f}{axr:7.1f} | {cor[0]:9.4f}{cor[1]:8.4f}{cor[2]:8.4f}{axc:7.1f}")

    sh = miller()
    print(f"\n=== Miller 1925-26: {len(sh)} sheets ===")
    tr = np.vstack([s["turn_raw"] for s in sh]); tc = np.vstack([s["turn_cor"] for s in sh])
    sr = np.array([s["raw"] for s in sh]); sc = np.array([s["cor"] for s in sh])
    for lab, a in (("single turns, raw", tr), ("single turns, drift removed", tc),
                   ("sheet averages, raw", sr), ("sheet averages, drift removed", sc)):
        print(f"{lab:32s}" + "".join(f"  k={k+1}:{v:.4f}" for k, v in enumerate(a.mean(0)[:4])))
    drift = np.array([s["drift"] for s in sh])
    print(f"\nmean |drift| per turn (sheet average): {np.abs(drift).mean():.3f} fringe; "
          f"sheets drifting down: {(drift < 0).sum()}, up: {(drift > 0).sum()}")
    for lab, key in (("raw", "ax_raw"), ("drift removed", "ax_cor")):
        ax = np.array([s[key] for s in sh])
        m, R = mean_axis(ax)
        near = np.mean([min(abs(((a - DRIFT_AXIS[np.sign(s['drift']) or 1]) + 90) % 180 - 90), 90) < 20
                        for a, s in zip(ax, sh)])
        print(f"axis {lab:14s}: mean {m:5.1f} deg, R={R:.2f}, corr with CMB axis {circ_corr(ax, [s['pred'] for s in sh]):+.2f}, "
              f"within 20 deg of the drift-artifact axis: {near:.0%}")
