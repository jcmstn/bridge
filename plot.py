#!/usr/bin/env python3
"""
plot.py — quick-look plot of one bridge run, any type code
==========================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-10-14

Finds a run's raw file by number, picks the swept quantity as x (per type
code, first one that actually varies), adds derived columns, and shows one
panel per measured/derived quantity vs x in an interactive window. Nothing
is written to disk.

Derived columns (only when not already in the file):
    {p}_X/Y/R_ohm         lock-in {p}_X/Y/R_V / excitation_current_A_rms
                          (demods report RMS; HARM, HARM6, SOT2H, SOT1I)
    [hall_|drain_]resistance[_even]_ohm
                          [hall_|drain_]voltage[_even]_V / the matching
                          current (drain_current_A, else sense_current_A,
                          else current_A)
    dVdI_ohm              differential resistance, IV only

When the current is constant the derived Ω panel replaces its V panel
(same shape, rescaled). SEM columns ({name}_sem_{unit}) become error bars.

Usage:
    uv run plot.py 115
    uv run plot.py 115 --sample ChipD_Device2      # when several samples have run 115
    uv run plot.py 115 --sample joacim/ChipD_Device2   # same sample name under two users
    uv run plot.py 115 --x magnet_current_A        # override the auto-picked x
    uv run plot.py 115 --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from instruments.data_naming import read_raw, text_encoding

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Swept quantity per type code, in preference order; the first column that
# varies wins, else point_index.
_FIELD = ("magnet_field_mT", "magnet_current_A")
X_CANDIDATES = {
    "IV": ("current_A",),
    "HALL": _FIELD, "BSWP": _FIELD, "HARM": _FIELD, "HARM6": _FIELD,
    "GSWP": ("gate_voltage_V",),
    "DIFFR": ("bias_V",),
    "NOISE": ("frequency_Hz",),
    "RT": ("temperature_1_K", "elapsed_s"),
    "SOTPS": ("pulse_amplitude_V", "assist_field_measured_mT", "magnet_current_A"),
    "SOT2H": ("pulse_amplitude_V", "assist_field_measured_mT", "magnet_current_A"),
    "SOT1I": ("pulse_current_A", "assist_field_measured_mT", "magnet_current_A"),
    "SOTPD": ("pulse_current_A", "assist_field_measured_mT", "magnet_current_A"),
    "NLSW": ("pulse_current_A", "init_field_measured_mT", "magnet_current_A"),
}
_GENERIC_X = tuple(dict.fromkeys(c for cands in X_CANDIDATES.values() for c in cands))

# Bookkeeping / instrument settings — never a y panel.
_SKIP_Y = re.compile(
    r"^(point_index|series_index|elapsed_s|settle_elapsed_s|sample_duration_s|hall_bar_.*)$"
    r"|_sem_|_n_samples$|_overload$|_time_constant_s$|_ref_phase_deg$|_drift_K$"
)
_DC_VOLTAGE = re.compile(r"^(hall_|drain_)?voltage(_even)?_V$")
_LOCKIN_X = re.compile(r"^(.+)_X_V$")


def find_raw(data_root: Path, run: int, sample: str | None) -> Path:
    """Sample folders (any dir holding raw/) may sit at any depth, e.g.
    data/USER/SAMPLE. `sample` matches the folder name or a trailing path
    like USER/SAMPLE."""
    sample_dirs = sorted(r.parent for r in data_root.rglob("raw") if r.is_dir())
    if sample:
        want = Path(sample).parts
        sample_dirs = [d for d in sample_dirs
                       if d.relative_to(data_root).parts[-len(want):] == want]
    hits = [p for d in sample_dirs
            for p in sorted((d / "raw").glob(f"{d.name}_{run:04d}_*.csv"))]
    if not hits:
        sys.exit(f"No raw file for run {run} under {data_root}"
                 + (f" matching sample {sample!r}" if sample else ""))
    if len(hits) > 1:
        names = "\n  ".join(str(p.relative_to(data_root)) for p in hits)
        sys.exit(f"Run {run} exists in several samples — pass --sample "
                 f"(NAME or USER/NAME):\n  {names}")
    return hits[0]


def read_header(path: Path) -> dict[str, str]:
    header = {}
    with open(path, encoding=text_encoding(path)) as f:
        for line in f:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            header[key.strip()] = value.strip()
    return header


def _varies(df: pd.DataFrame, col: str) -> bool:
    return col in df and df[col].nunique(dropna=True) > 1


def _sem_of(col: str) -> str:
    name, _, unit = col.rpartition("_")
    return f"{name}_sem_{unit}"


def derive(df: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Add derived columns. Returns (df, V columns superseded by an Ω twin
    because their current is constant)."""
    df = df.copy()
    hidden: set[str] = set()

    def add_ratio(v: str, i_col: str, out: str) -> None:
        if out not in df:
            current = df[i_col].replace(0, np.nan)
            df[out] = df[v] / current
            if _sem_of(v) in df:
                df[_sem_of(out)] = df[_sem_of(v)] / current.abs()
        if not _varies(df, i_col):
            hidden.add(v)

    if "excitation_current_A_rms" in df:
        for col in list(df.columns):
            if m := _LOCKIN_X.match(col):
                p = m.group(1)
                for comp in ("X", "Y", "R"):
                    if f"{p}_{comp}_V" in df:
                        add_ratio(f"{p}_{comp}_V", "excitation_current_A_rms", f"{p}_{comp}_ohm")

    for col in list(df.columns):
        if m := _DC_VOLTAGE.match(col):
            prefix, even = m.group(1) or "", m.group(2) or ""
            i_col = next((c for c in (f"{prefix}current_A", "sense_current_A", "current_A")
                          if c in df), None)
            if i_col:
                add_ratio(col, i_col, f"{prefix}resistance{even}_ohm")

    if _varies(df, "current_A") and "voltage_V" in df and "dVdI_ohm" not in df:
        d_i = np.gradient(df["current_A"].to_numpy(float))
        d_v = np.gradient(df["voltage_V"].to_numpy(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            df["dVdI_ohm"] = np.where(d_i != 0, d_v / d_i, np.nan)

    return df, hidden


def pick_x(df: pd.DataFrame, type_code: str) -> str:
    for col in X_CANDIDATES.get(type_code, _GENERIC_X):
        if _varies(df, col):
            return col
    return "point_index" if "point_index" in df else df.columns[0]


def y_columns(df: pd.DataFrame, x: str, hidden: set[str]) -> list[str]:
    return [c for c in df.columns
            if c != x and c not in hidden and not _SKIP_Y.search(c)
            and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
            and _varies(df, c)]


def plot(df: pd.DataFrame, x: str, ys: list[str], header: dict[str, str]) -> None:
    import matplotlib.pyplot as plt

    ncols = min(3, len(ys)) or 1
    nrows = max(1, math.ceil(len(ys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    for ax, y in zip(axes.flat, ys):
        sem = df[_sem_of(y)] if _sem_of(y) in df else None
        ax.errorbar(df[x], df[y], yerr=sem, fmt="o-", ms=3, lw=1, capsize=2)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if x == "frequency_Hz":
            ax.set_xscale("log")
        if y.startswith("asd_"):
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(ys):]:
        ax.set_visible(False)
    if not ys:
        axes[0, 0].text(0.5, 0.5, "nothing varies in this file", ha="center", va="center")

    h = header.get
    title = (f"#{h('run', '?')}  {h('sample', '')}  {h('device', '')}  {h('type', '')}"
             f"\nT_set={h('T_setpoint_K') or '–'} K  series={h('series', '')}"
             f"  status={h('status', '')}")
    if h("comment"):
        title += f"\n{h('comment')}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick-look plot of one bridge run.")
    ap.add_argument("run", type=int)
    ap.add_argument("--sample")
    ap.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--x", help="x column (default: auto from the type code)")
    args = ap.parse_args()

    path = find_raw(args.data_root, args.run, args.sample)
    header = read_header(path)
    df, hidden = derive(read_raw(path))
    if df.empty:
        sys.exit(f"{path.name}: no data rows")
    if args.x and args.x not in df:
        sys.exit(f"--x {args.x!r} not in columns: {', '.join(df.columns)}")
    x = args.x or pick_x(df, header.get("type", ""))
    print(path)
    plot(df, x, y_columns(df, x, hidden), header)


if __name__ == "__main__":
    main()
