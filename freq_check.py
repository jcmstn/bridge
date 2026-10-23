#!/usr/bin/env python3
"""
freq_check.py — is the lock-in excitation frequency in a flat region?
====================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-10-22

Takes ≥3 runs of the same lock-in sweep (HARM, HARM6, SOT2H, SOT1I) that
differ only in excitation frequency, and checks whether the signal depends
on frequency over that range (capacitive pickup, RC roll-off, thermal
quadrature, ...). Field points may differ between runs: every run is
interpolated onto the first run's x points, branch by branch (an up/down
sweep is two branches, matched in order, so hysteresis is not mistaken for
a frequency effect).

Per channel ({p}_X_ohm and {p}_Y_ohm, i.e. lock-in V / excitation current
RMS) a straight line y(f) is fitted at every common x point. Its change
across the tested range, Δ = slope · (f_max − f_min), is compared with
    scale  peak-to-peak of the run-mean curve (|mean| if flat), and
    se     standard error of Δ from the scatter about those lines.
A channel is NOT OK when rms(Δ)/scale > --tol AND rms(Δ)/se > 3, i.e. the
change is both large and above the noise. Every lock-in signal in the file
(1f, 2f, rxx_1f, ...) is checked, X and Y each, and all must pass; every run
must carry the same signals. The verdict is a hint: inspect
the plotted curves (top row) and residuals vs the mean (bottom row).

Measure the frequencies in shuffled order (e.g. 317, 113, 731), otherwise
slow drift between runs looks like a frequency trend.

Usage:
    uv run freq_check.py 120 121 122
    uv run freq_check.py 120 121 122 123 --sample ChipD_Device2
    uv run freq_check.py 120 121 122 --tol 0.05      # allow 5 % change
    uv run freq_check.py 120 121 122 --x magnet_current_A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from instruments.data_naming import read_raw
from plot import _DEFAULT_DATA_DIR, derive, find_raw, pick_x, read_header

# Header keys allowed to differ between the compared runs.
_IGNORED_KEYS = {"run", "timestamp", "status", "comment", "series", "T_K", "cooldown",
                 "excitation_frequency_Hz", "field_sweep_rows_A"}
_MUST_MATCH = ("sample", "device", "type")
Z_LIMIT = 3.0


def frequency_trend(freqs, curves) -> tuple[np.ndarray, float]:
    """curves: (n_runs, n_points) on a common grid. Returns (Δ per point,
    standard error of Δ). Needs ≥3 runs for a residual to exist."""
    f = np.asarray(freqs, float)
    y = np.asarray(curves, float)
    fc = f - f.mean()
    sxx = fc @ fc
    slope = fc @ (y - y.mean(axis=0)) / sxx
    resid = y - y.mean(axis=0) - np.outer(fc, slope)
    dof = y.shape[1] * (len(f) - 2)
    sigma = np.sqrt((resid ** 2).sum() / dof)
    span = f.max() - f.min()
    return slope * span, sigma / np.sqrt(sxx) * span


def judge(freqs, curves, tol: float) -> dict:
    delta, se = frequency_trend(freqs, curves)
    mean = np.asarray(curves, float).mean(axis=0)
    scale = np.ptp(mean) or abs(mean.mean()) or 1.0
    d_rms = float(np.sqrt(np.mean(delta ** 2)))
    rel = d_rms / scale
    z = d_rms / se if se > 0 else np.inf
    return {"delta_rms": d_rms, "scale": scale, "rel": rel, "z": z,
            "ok": not (rel > tol and z > Z_LIMIT)}


def signal_verdicts(results: dict[str, dict]) -> dict[str, list[str]]:
    """{p}_X_ohm/{p}_Y_ohm results -> {p: failing components}, e.g.
    {"1f": [], "rxx_1f": ["Y"]}."""
    out: dict[str, list[str]] = {}
    for ch, r in results.items():
        prefix, comp = ch[:-len("_X_ohm")], ch[-len("X_ohm")]
        out.setdefault(prefix, [])
        if not r["ok"]:
            out[prefix].append(comp)
    return out


def branches(df: pd.DataFrame, x: str) -> list[np.ndarray]:
    """Row indices of each monotonic sweep branch, split on the commanded
    magnet current when it varies (exact setpoints), else on x. A turning
    point belongs to the branch it ends."""
    col = "magnet_current_A" if "magnet_current_A" in df and df["magnet_current_A"].nunique() > 1 else x
    step = np.sign(np.diff(df[col].to_numpy(float)))
    step = pd.Series(step).replace(0, np.nan).ffill().bfill().to_numpy()
    labels = np.concatenate([[0], np.cumsum(np.r_[0, step[1:] != step[:-1]])]) if len(step) else np.zeros(1)
    return [np.flatnonzero(labels == k) for k in range(int(labels.max()) + 1)]


def common_grid(runs: list[pd.DataFrame], x: str) -> list[np.ndarray]:
    """Per branch: the first run's x points inside the overlap of all runs."""
    split = [branches(df, x) for df in runs]
    if len({len(s) for s in split}) > 1:
        sys.exit(f"Runs have different numbers of sweep branches {[len(s) for s in split]} "
                 "— not the same sweep.")
    grids = []
    for k, ref in enumerate(split[0]):
        xs = [df[x].to_numpy(float)[s[k]] for df, s in zip(runs, split)]
        lo, hi = max(np.nanmin(v) for v in xs), min(np.nanmax(v) for v in xs)
        g = xs[0][(xs[0] >= lo) & (xs[0] <= hi)]
        grids.append(g)
    return grids


def on_grid(df: pd.DataFrame, x: str, y: str, grids: list[np.ndarray]) -> np.ndarray:
    out = []
    for idx, g in zip(branches(df, x), grids):
        part = df.iloc[idx][[x, y]].dropna().sort_values(x)
        out.append(np.interp(g, part[x], part[y]) if len(part) else np.full(len(g), np.nan))
    return np.concatenate(out)


def frequency_of(header: dict[str, str], df: pd.DataFrame) -> float:
    try:
        return float(header["excitation_frequency_Hz"])
    except (KeyError, ValueError):
        if "excitation_frequency_Hz" in df:
            return float(df["excitation_frequency_Hz"].median())
    sys.exit(f"Run {header.get('run')}: no excitation_frequency_Hz in header or columns")


def check_same(headers: list[dict[str, str]]) -> None:
    for key in _MUST_MATCH:
        values = [h.get(key, "") for h in headers]
        if len(set(values)) > 1:
            sys.exit(f"Runs differ in {key}: {values} — nothing to compare.")
    for key in sorted(set().union(*headers) - _IGNORED_KEYS):
        values = [h.get(key, "") for h in headers]
        if len(set(values)) > 1:
            print(f"WARNING: runs differ in {key}: {values}")


def plot(runs, freqs, labels, x, channels, grids, results, header) -> None:
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))
    order = np.argsort(freqs)
    n = len(channels)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.5), squeeze=False, sharex="col")
    grid_x = np.concatenate(grids)
    bounds = np.cumsum([len(g) for g in grids])[:-1]
    for j, ch in enumerate(channels):
        curves = np.array([on_grid(df, x, ch, grids) for df in runs])
        mean = curves.mean(axis=0)
        top, bottom = axes[0, j], axes[1, j]
        for i in order:
            df = runs[i]
            top.plot(df[x], df[ch], "o-", ms=2, lw=0.8, color=colors[i], label=labels[i])
            for seg_x, seg_y in zip(np.split(grid_x, bounds), np.split(curves[i] - mean, bounds)):
                bottom.plot(seg_x, seg_y, "o-", ms=2, lw=0.8, color=colors[i])
        r = results[ch]
        top.set_title(f"{ch}\nΔ={r['rel']:.1%} of scale, z={r['z']:.1f}  "
                      f"{'OK' if r['ok'] else 'NOT OK'}",
                      fontsize=9, color="black" if r["ok"] else "red")
        top.set_ylabel(ch)
        bottom.set_ylabel(f"{ch} − mean")
        bottom.set_xlabel(x)
        for ax in (top, bottom):
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    h = header.get
    fig.suptitle(f"{h('sample', '')}  {h('device', '')}  {h('type', '')}  frequency check: "
                 + ", ".join(f"{freqs[i]:g}" for i in order) + " Hz", fontsize=10)
    fig.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Check a lock-in sweep for excitation-frequency dependence.")
    ap.add_argument("runs", type=int, nargs="+", help="≥3 run numbers at different frequencies")
    ap.add_argument("--sample")
    ap.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--x", help="x column (default: auto from the type code)")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="allowed change across the frequency range, fraction of the signal scale (default 0.02)")
    args = ap.parse_args()
    if len(args.runs) < 3:
        sys.exit("Need at least 3 runs (a line through 2 points has no noise estimate).")

    paths = [find_raw(args.data_root, r, args.sample) for r in args.runs]
    headers = [read_header(p) for p in paths]
    runs = [derive(read_raw(p))[0] for p in paths]
    for p, df in zip(paths, runs):
        if df.empty:
            sys.exit(f"{p.name}: no data rows")
        print(p)
    check_same(headers)

    freqs = np.array([frequency_of(h, df) for h, df in zip(headers, runs)])
    if len(set(freqs)) < len(freqs):
        sys.exit(f"Frequencies are not all different: {freqs.tolist()} Hz")

    x = args.x or pick_x(runs[0], headers[0].get("type", ""))
    if any(x not in df for df in runs):
        sys.exit(f"x column {x!r} missing in some runs")
    channels = list(dict.fromkeys(c for df in runs for c in df.columns
                                  if c.endswith(("_X_ohm", "_Y_ohm"))))
    for c in channels:
        if missing := [r for r, df in zip(args.runs, runs) if c not in df]:
            sys.exit(f"{c} missing in run(s) {missing} — the runs measured different signals.")
    if not channels:
        sys.exit("No lock-in X/Y channels (need {p}_X_V and excitation_current_A_rms).")

    grids = common_grid(runs, x)
    if sum(len(g) for g in grids) < 2:
        sys.exit(f"Runs share fewer than 2 points in {x}.")

    labels = [f"#{r}  {f:g} Hz" for r, f in zip(args.runs, freqs)]
    print(f"\nx = {x}, {sum(len(g) for g in grids)} common points, "
          f"f = {freqs.min():g}–{freqs.max():g} Hz, tol = {args.tol:.1%}, z limit = {Z_LIMIT:g}")
    print(f"{'channel':<16}{'Δ_rms [Ω]':>12}{'scale [Ω]':>12}{'Δ/scale':>10}{'z':>8}  verdict")
    results = {}
    for ch in channels:
        curves = np.array([on_grid(df, x, ch, grids) for df in runs])
        r = results[ch] = judge(freqs, curves, args.tol)
        print(f"{ch:<16}{r['delta_rms']:>12.4g}{r['scale']:>12.4g}{r['rel']:>10.2%}{r['z']:>8.1f}  "
              f"{'OK' if r['ok'] else 'NOT OK'}")
    signals = signal_verdicts(results)
    print()
    for prefix, failing in signals.items():
        print(f"{prefix:<16}" + (f"NOT OK ({', '.join(failing)})" if failing else "OK"))
    bad = [p for p, failing in signals.items() if failing]
    print("\nVERDICT: " + ("OK — no significant frequency dependence in this range"
                           if not bad else f"NOT OK — frequency dependent: {', '.join(bad)}"))
    print("This is a hint only: inspect the curves and residuals before deciding.")
    plot(runs, freqs, labels, x, channels, grids, results, headers[0])


if __name__ == "__main__":
    main()
