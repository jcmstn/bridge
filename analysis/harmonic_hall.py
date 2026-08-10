#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmonic Hall analysis for an in-plane magnetised film measured with an
out-of-plane external field.

Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-06

Fits the hard-axis 1f loop (H_K_eff) and the 2f damping-like torque slope
(H_DL) from a dual-harmonic Hall sweep, with a sanity check at every step —
nothing here is trusted until it's been checked, and a fit that converges is
not itself evidence the model is right. See docs/harmonic-hall-theory.md for
the geometry, sign conventions, and the full R_2f derivation this relies on
(change the sign conventions in the code below if your setup differs).
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
#  1.  CONFIGURATION  --  EVERYTHING YOU MUST TELL THE SCRIPT
# =============================================================================
@dataclass
class Config:
    # ---- file -------------------------------------------------------------
    csv_path: str = "/mnt/user-data/uploads/example.csv"
    outdir: str = "/mnt/user-data/outputs"

    # ---- EXCITATION  (REQUIRED -- analysis is qualitative without these) ---
    # Current *through the device*. If you drive with a voltage source through
    # a ballast resistor, put the actual current here, not the set voltage.
    current_A: Optional[float] = None          # e.g. 1.0e-3
    # Is `current_A` an RMS value or a zero-to-peak amplitude?
    current_is_rms: bool = True
    # Does the lock-in report RMS or peak amplitudes? (SRS -> RMS by default,
    # Zurich MFLI -> configurable, CHECK YOURS. This factor of sqrt(2)
    # propagates linearly into H_DL and hence into your spin Hall angle.)
    lockin_reports_rms: bool = True
    excitation_frequency_Hz: Optional[float] = None   # e.g. 137.0

    # ---- FIELD GEOMETRY ---------------------------------------------------
    # Polar angle of H away from the film normal, degrees. 0 = perfectly
    # out-of-plane. Even 1-2 deg of misalignment puts a large in-plane field
    # component on an easy-plane magnet and will distort the hard-axis loop.
    field_polar_misalignment_deg: Optional[float] = None
    # Azimuthal angle between the current direction (+x) and the in-plane
    # projection of H / the in-plane easy axis, degrees.
    field_azimuth_deg: Optional[float] = None

    # ---- DEVICE GEOMETRY (needed only for spin Hall efficiency) -----------
    ni_thickness_nm: Optional[float] = None
    nm_thickness_nm: Optional[float] = None
    nm_material: Optional[str] = None
    hallbar_width_um: Optional[float] = None
    hallbar_length_um: Optional[float] = None     # current-path length; not used by any
                                                   # calc in this file yet, kept for the record
    film_thickness_nm: Optional[float] = None     # generic single-layer thickness as recorded
                                                   # by the run (mfli_dual_harmonic doesn't know
                                                   # about a NM/NI split) -- use ni_thickness_nm /
                                                   # nm_thickness_nm instead if you need that split
    ni_resistivity_uohm_cm: Optional[float] = None
    nm_resistivity_uohm_cm: Optional[float] = None
    Ms_kA_per_m: Optional[float] = None          # independent (VSM/SQUID) value

    # ---- LOCK-IN SETTINGS (needed for the settling / noise checks) --------
    time_constant_s: Optional[float] = None
    filter_order: Optional[int] = None
    ref_phase_1f_deg: Optional[float] = None     # what you actually programmed
    ref_phase_2f_deg: Optional[float] = None

    # ---- ANALYSIS OPTIONS -------------------------------------------------
    # Rotate X/Y numerically by this much before analysis (deg). Leave at 0.0
    # for the first pass so the raw phase problem is visible in the plots.
    software_phase_rotation_1f_deg: float = 0.0
    software_phase_rotation_2f_deg: float = 0.0
    # Fraction of the max |field| beyond which the film is assumed saturated
    # out-of-plane. Used for the background/null check before H_K is known.
    saturation_guess_frac: float = 0.85
    # Tolerances for the sanity checks
    tol_phase_1f_deg: float = 2.0
    tol_phase_2f_vs_double_deg: float = 5.0
    tol_even_odd_ratio_1f: float = 0.10
    tol_hysteresis_frac: float = 0.05

    # ---- COLUMN NAMES -----------------------------------------------------
    col_field_mT: str = "magnet_field_mT"
    col_magnet_I: str = "magnet_current_A"
    col_time: str = "timestamp"
    col_1f_X: str = "1f_X_V"
    col_1f_Y: str = "1f_Y_V"
    col_1f_R: str = "1f_R_V"
    col_1f_th: str = "1f_theta_deg"
    col_1f_sd: str = "1f_R_std_V"
    col_2f_X: str = "2f_X_V"
    col_2f_Y: str = "2f_Y_V"
    col_2f_R: str = "2f_R_V"
    col_2f_th: str = "2f_theta_deg"
    col_2f_sd: str = "2f_R_std_V"


# =============================================================================
#  2.  SANITY REPORT INFRASTRUCTURE
# =============================================================================
class Report:
    """Collects PASS / WARN / FAIL / INFO items and prints a readable summary."""

    LEVELS = {"PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3}

    def __init__(self):
        self.items: list[dict] = []

    def add(self, level: str, name: str, message: str, value=None):
        self.items.append(
            {"level": level, "name": name, "message": message, "value": value}
        )

    def passed(self, name, msg, value=None):
        self.add("PASS", name, msg, value)

    def info(self, name, msg, value=None):
        self.add("INFO", name, msg, value)

    def warn(self, name, msg, value=None):
        self.add("WARN", name, msg, value)

    def fail(self, name, msg, value=None):
        self.add("FAIL", name, msg, value)

    @property
    def n_fail(self):
        return sum(1 for i in self.items if i["level"] == "FAIL")

    @property
    def n_warn(self):
        return sum(1 for i in self.items if i["level"] == "WARN")

    def render(self) -> str:
        mark = {"PASS": "[ ok ]", "INFO": "[info]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
        lines = ["", "=" * 78, " SANITY REPORT", "=" * 78]
        for it in self.items:
            lines.append(f"{mark[it['level']]}  {it['name']}")
            for ln in it["message"].split("\n"):
                lines.append(f"          {ln}")
        lines.append("-" * 78)
        lines.append(f" {self.n_fail} failure(s), {self.n_warn} warning(s)")
        lines.append("=" * 78)
        return "\n".join(lines)


# =============================================================================
#  3.  LOADING AND PRE-PROCESSING
# =============================================================================
def find_data_files(directory: Path, extensions=(".csv",)) -> list[Path]:
    """Return measurement files directly inside `directory`, sorted by name.
    Mirrors ahe_core.find_data_files() -- used by harmonic_hall_gui.py to
    list the files a chosen directory offers for batch analysis."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def load_data(cfg: Config, rep: Report) -> pd.DataFrame:
    path = Path(cfg.csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    required = [cfg.col_field_mT, cfg.col_1f_X, cfg.col_1f_Y, cfg.col_2f_X, cfg.col_2f_Y]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    rep.info("file", f"{path.name}: {len(df)} rows, {len(df.columns)} columns")

    # Field in tesla, signed.
    df["B_T"] = df[cfg.col_field_mT] * 1e-3

    # Elapsed time, if a timestamp exists.
    if cfg.col_time in df.columns:
        try:
            t = pd.to_datetime(df[cfg.col_time])
            df["t_s"] = (t - t.iloc[0]).dt.total_seconds()
        except Exception:
            df["t_s"] = np.arange(len(df), dtype=float)
    else:
        df["t_s"] = np.arange(len(df), dtype=float)

    return df


def rotate_phase(X, Y, deg):
    """Rotate the demodulated vector by -deg, i.e. apply a reference-phase shift."""
    a = np.radians(deg)
    return X * np.cos(a) + Y * np.sin(a), -X * np.sin(a) + Y * np.cos(a)


def to_resistance(V, cfg: Config) -> tuple[np.ndarray, bool]:
    """
    Convert lock-in volts to ohms. Returns (value, is_calibrated).
    If the current is unknown, returns volts unchanged and flags it, so that
    the whole pipeline still runs and every plot is still produced -- only the
    absolute scale of H_DL is then meaningless.
    """
    if cfg.current_A is None:
        return np.asarray(V, float), False
    I = float(cfg.current_A)
    # Put the current on the same footing (rms vs peak) as the lock-in output.
    if cfg.current_is_rms and not cfg.lockin_reports_rms:
        I = I * np.sqrt(2.0)
    if (not cfg.current_is_rms) and cfg.lockin_reports_rms:
        I = I / np.sqrt(2.0)
    return np.asarray(V, float) / I, True


# Maps a column mfli_dual_harmonic.py's build_run_metadata() writes into every
# row of its output CSV to the Config field it fills in. See that function's
# docstring (bridge/mfli/mfli_dual_harmonic.py) for what each column means.
# excitation_current_A_rms (not _peak) is used because the demod X/Y/R this
# script reads are RMS -- see 'demod_output_convention' in the same file --
# so current_A and the lock-in output are put on the same footing directly,
# with no peak/RMS conversion left for apply_run_metadata() to get wrong.
RUN_METADATA_COLUMNS = {
    "excitation_frequency_Hz":       "excitation_frequency_Hz",
    "excitation_current_A_rms":      "current_A",
    "demod1_time_constant_s":        "time_constant_s",
    "demod1_filter_order":           "filter_order",
    "demod1_ref_phase_deg":          "ref_phase_1f_deg",
    "demod2_ref_phase_deg":          "ref_phase_2f_deg",
    "hall_bar_width_um":             "hallbar_width_um",
    "hall_bar_length_um":            "hallbar_length_um",
    "hall_bar_thickness_nm":         "film_thickness_nm",
    "field_angle_from_oop_deg":      "field_polar_misalignment_deg",
}


def apply_run_metadata(df: pd.DataFrame, cfg: Config, rep: Report) -> Config:
    """
    mfli_dual_harmonic.py's run_measurement() (see build_run_metadata() there)
    writes excitation/filter/phase/geometry metadata as columns on every row
    of its output CSV rather than a separate sidecar file. Pull it in here so
    a Config field only needs to be set by hand for files that predate that
    metadata, were produced by something else, or need a value the file can't
    supply (Ms, resistivities, nm_material, field_azimuth_deg -- none of
    those are recorded by mfli_dual_harmonic.py either).

    A Config field that is already set (not None) is left alone but
    cross-checked against the file's value: silently trusting a stale
    manually-set field over a mismatched file is a worse failure mode than a
    loud warning, since it would look like the analysis ran clean.

    Returns a new Config (the input is not mutated).
    """
    updates: dict = {}
    for csv_col, cfg_field in RUN_METADATA_COLUMNS.items():
        if csv_col not in df.columns:
            continue
        values = df[csv_col].dropna().unique()
        if len(values) == 0:
            continue
        if len(values) > 1:
            rep.warn(
                f"metadata: {csv_col}",
                f"not constant across the file ({len(values)} distinct values) -- "
                "using the first point's value. This usually means "
                "set_excitation_frequency() or update_filter() was called mid-run; "
                "if so, split the file and analyze each segment separately.",
            )
        value = values[0]
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()

        current = getattr(cfg, cfg_field)
        if current is None:
            updates[cfg_field] = value
            rep.info(f"metadata: {cfg_field}", f"{value}  (from '{csv_col}' in file)")
        else:
            mismatch = (
                not np.isclose(current, value, rtol=1e-6, atol=1e-12)
                if isinstance(value, float) else current != value
            )
            if mismatch:
                rep.warn(
                    f"metadata: {cfg_field}",
                    f"Config has {current!r} but the file says {value!r} "
                    f"(from '{csv_col}'). Keeping the manually-set Config value -- "
                    "clear it to use the file's.",
                )

    if "current_A" in updates:
        # excitation_current_A_rms is already RMS and demod output is already
        # RMS (see the comment on RUN_METADATA_COLUMNS above) -- force these
        # so to_resistance() doesn't apply a spurious sqrt(2) conversion.
        updates["current_is_rms"] = True
        updates["lockin_reports_rms"] = True

    if not updates:
        return cfg
    return replace(cfg, **updates)


# =============================================================================
#  4.  SANITY CHECKS
# =============================================================================

def estimate_noise(x: np.ndarray) -> float:
    """
    Robust point-to-point noise estimate.

    A first difference, std(diff(x))/sqrt(2), is what most people use, but it
    counts the genuine field-dependent change between adjacent points as
    "noise" and therefore massively overestimates it on a steep part of the
    loop. The second difference cancels any locally linear trend:
        var(x[i+2] - 2x[i+1] + x[i]) = 6 * sigma^2
    and a median-absolute-deviation scaling makes it insensitive to the few
    genuinely curved points (the kinks at +-H_K).
    """
    if len(x) < 4:
        return float(np.std(x))
    d2 = np.diff(x, n=2)
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(1.4826 * mad / np.sqrt(6.0))


def check_internal_consistency(df: pd.DataFrame, cfg: Config, rep: Report):
    """R and theta must be consistent with X and Y. If not, someone has
    post-processed one of them and you no longer know which is authoritative."""
    for tag, cX, cY, cR, cT in [
        ("1f", cfg.col_1f_X, cfg.col_1f_Y, cfg.col_1f_R, cfg.col_1f_th),
        ("2f", cfg.col_2f_X, cfg.col_2f_Y, cfg.col_2f_R, cfg.col_2f_th),
    ]:
        if cR not in df or cT not in df:
            continue
        R_calc = np.hypot(df[cX], df[cY])
        th_calc = np.degrees(np.arctan2(df[cY], df[cX]))
        dR = np.max(np.abs(R_calc - df[cR]) / np.maximum(np.abs(df[cR]), 1e-30))
        dT = np.max(np.abs(((th_calc - df[cT] + 180) % 360) - 180))
        if dR < 1e-6 and dT < 1e-4:
            rep.passed(
                f"{tag} X/Y/R/theta consistency",
                f"R and theta reproduce X,Y to {dR:.1e} (rel) and {dT:.1e} deg.",
            )
        else:
            rep.warn(
                f"{tag} X/Y/R/theta consistency",
                f"Mismatch: rel dR={dR:.2e}, dtheta={dT:.2e} deg.\n"
                "One of the columns has been altered independently of the others.",
            )


def check_phase(df: pd.DataFrame, cfg: Config, rep: Report) -> dict:
    """
    The central check for this experiment.

    (a) The 1f phase must be ~0 (or at least field-independent). At the drive
        frequencies used here we are ~7 orders of magnitude below FMR, so the
        magnetisation follows the field adiabatically and R_xy^1f is a *static*
        resistance. There is no legitimate mechanism to put weight in Y.

    (b) The 2f phase should equal 2 x (1f phase) IF the phase error is a pure
        propagation delay. A residual deviation is diagnostic:
          - frequency-independent  -> instrumental, correct it;
          - frequency-dependent    -> likely a real thermal (ANE) quadrature
                                      component with its own time constant
                                      tau_th; do NOT null it away, it is signal.
        This script can only see one frequency, so it reports the deviation
        and tells you to repeat at several f.
    """
    th1 = np.asarray(df[cfg.col_1f_th], float) if cfg.col_1f_th in df else np.degrees(
        np.arctan2(df[cfg.col_1f_Y], df[cfg.col_1f_X])
    )
    th2 = np.asarray(df[cfg.col_2f_th], float) if cfg.col_2f_th in df else np.degrees(
        np.arctan2(df[cfg.col_2f_Y], df[cfg.col_2f_X])
    )
    # Unwrap around the circle before averaging.
    m1 = np.degrees(np.arctan2(np.mean(np.sin(np.radians(th1))),
                               np.mean(np.cos(np.radians(th1)))))
    m2 = np.degrees(np.arctan2(np.mean(np.sin(np.radians(th2))),
                               np.mean(np.cos(np.radians(th2)))))
    s1, s2 = np.std(((th1 - m1 + 180) % 360) - 180), np.std(((th2 - m2 + 180) % 360) - 180)

    if abs(m1) <= cfg.tol_phase_1f_deg:
        rep.passed("1f phase null", f"mean theta_1f = {m1:+.3f} deg (sd {s1:.3f}).")
    else:
        rep.fail(
            "1f phase null",
            f"mean theta_1f = {m1:+.3f} deg (sd {s1:.3f}), outside "
            f"+-{cfg.tol_phase_1f_deg} deg.\n"
            "Re-null the reference phase ON R_xy, in the 4-terminal configuration,\n"
            "at the frequency and current you actually use. A 2-terminal null on\n"
            "the current channel is a different circuit (different cabling RC,\n"
            "different contact impedance, includes the source output stage) and\n"
            "does not transfer.",
        )

    dev = ((m2 - 2 * m1 + 180) % 360) - 180
    if abs(dev) <= cfg.tol_phase_2f_vs_double_deg:
        rep.passed(
            "2f phase vs 2 x 1f",
            f"theta_2f = {m2:+.3f} deg, 2*theta_1f = {2*m1:+.3f} deg, "
            f"deviation {dev:+.3f} deg -- consistent with a pure delay.",
        )
    else:
        rep.warn(
            "2f phase vs 2 x 1f",
            f"theta_2f = {m2:+.3f} deg but 2*theta_1f = {2*m1:+.3f} deg; "
            f"deviation {dev:+.3f} deg.\n"
            "This is NOT automatically an instrumental error to be nulled away.\n"
            "A pure signal-chain delay tau gives dtheta_n = n * dtheta_1 exactly;\n"
            "an RC pole gives arctan(w R C), which is only linear in w while\n"
            "wRC << 1. And a Joule-heating-driven Nernst term has its own thermal\n"
            "time constant tau_th, giving a GENUINE quadrature component at 2f.\n"
            "ACTION: repeat this sweep at >=3 frequencies spanning a factor ~10.\n"
            "  deviation constant in f -> instrumental, rotate it out.\n"
            "  deviation grows with f  -> thermal; keep it and extrapolate the\n"
            "                             extracted fields to f -> 0 instead.",
        )

    # Field dependence of the phase is another red flag: a purely instrumental
    # phase cannot know about the applied field.
    for tag, th, sd in [("1f", th1, s1), ("2f", th2, s2)]:
        if sd > 5.0:
            rep.warn(
                f"{tag} phase stability",
                f"theta_{tag} scatters by {sd:.2f} deg over the sweep.\n"
                "An instrumental phase is field-independent by construction. Large\n"
                "scatter means either poor SNR (check against the std columns) or a\n"
                "field-dependent admixture of a second signal with a different phase.",
            )
    return {"theta_1f_mean_deg": m1, "theta_1f_sd_deg": s1,
            "theta_2f_mean_deg": m2, "theta_2f_sd_deg": s2,
            "theta_2f_minus_2theta_1f_deg": dev}


def split_branches(df: pd.DataFrame, rep: Report) -> list[np.ndarray]:
    """Split the sweep at its turning points so that up and down branches can
    be compared (hysteresis is a physical statement; averaging them blindly
    destroys it)."""
    B = df["B_T"].values
    d = np.sign(np.diff(B))
    d[d == 0] = 1
    turns = np.where(np.diff(d) != 0)[0] + 1
    edges = [0, *(turns + 1), len(B)]
    branches = [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    branches = [b for b in branches if len(b) > 5]
    rep.info(
        "sweep structure",
        f"{len(branches)} branch(es): "
        + ", ".join(
            f"[{b[0]}..{b[-1]}] {B[b[0]]*1e3:+.0f} -> {B[b[-1]]*1e3:+.0f} mT"
            for b in branches
        ),
    )
    if len(branches) < 2:
        rep.warn(
            "sweep structure",
            "Only one branch found. Without both sweep directions you cannot\n"
            "separate real magnetic hysteresis from magnet remanence or drift.",
        )
    return branches


def check_field_calibration(df: pd.DataFrame, cfg: Config, rep: Report):
    """The magnet current -> field relation should be single-valued and close
    to linear for an air-core / unsaturated magnet. Deviations reveal iron-core
    hysteresis or a saturating pole piece, which shifts your H_K estimate."""
    if cfg.col_magnet_I not in df:
        rep.warn("field calibration", "no magnet current column; cannot check.")
        return
    I = df[cfg.col_magnet_I].values
    B = df["B_T"].values
    if np.ptp(I) == 0:
        return
    p = np.polyfit(I, B, 1)
    resid = B - np.polyval(p, I)
    rel = np.max(np.abs(resid)) / max(np.ptp(B), 1e-12)
    msg = (f"B = {p[0]*1e3:.3f} mT/A * I {p[1]*1e3:+.3f} mT; "
           f"max nonlinearity/hysteresis = {rel*100:.2f} % of the field span.")
    if rel < 0.02:
        rep.passed("field calibration", msg)
    else:
        rep.warn(
            "field calibration",
            msg + "\nThat is large enough to matter for H_K_eff. If it is a loop\n"
            "(different on up and down sweeps) it is core remanence: use the\n"
            "measured field, never the set current, as the abscissa.",
        )


def check_settling(df: pd.DataFrame, cfg: Config, rep: Report):
    """Each point must be taken many time constants after the field settles,
    otherwise the 'signal' is a filter transient and looks like hysteresis."""
    if "t_s" not in df or len(df) < 3:
        return
    dt = np.median(np.diff(df["t_s"].values))
    if cfg.time_constant_s is None:
        rep.info(
            "settling",
            f"median {dt:.1f} s per point. Lock-in time constant not supplied,\n"
            "so I cannot verify settling. Rule of thumb: wait >= 5 tau for a\n"
            "6 dB/oct filter, >= 10 tau for 24 dB/oct.",
        )
        return
    n = dt / cfg.time_constant_s
    order = cfg.filter_order or 1
    need = {1: 5, 2: 7, 3: 9, 4: 10}.get(order, 10)
    if n >= need:
        rep.passed("settling", f"{dt:.1f} s per point = {n:.1f} tau (need >~{need}).")
    else:
        rep.fail(
            "settling",
            f"{dt:.1f} s per point is only {n:.1f} tau, need >~{need} for a\n"
            f"filter order {order}. Points are contaminated by the previous field.",
        )


def check_noise(df: pd.DataFrame, cfg: Config, rep: Report):
    """Compare the point-to-point scatter with the standard deviation the
    lock-in reported. If the scatter is much larger, something is drifting
    between points that is not captured by the within-point averaging."""
    for tag, cX, csd in [("1f", cfg.col_1f_X, cfg.col_1f_sd),
                         ("2f", cfg.col_2f_X, cfg.col_2f_sd)]:
        if csd not in df:
            continue
        x = df[cX].values
        scatter = estimate_noise(x)
        reported = np.median(df[csd].values)
        if reported <= 0:
            continue
        ratio = scatter / reported
        msg = (f"point-to-point scatter {scatter:.3e} V vs reported sd "
               f"{reported:.3e} V (ratio {ratio:.1f}).")
        if ratio < 5:
            rep.passed(f"{tag} noise consistency", msg)
        else:
            rep.warn(
                f"{tag} noise consistency",
                msg + "\nScatter far exceeds the within-point statistics: look for\n"
                "slow drift (temperature, contact, magnet) between points rather\n"
                "than white noise. Averaging longer will not help.",
            )


def odd_even_decompose(B, S):
    """Return (B_grid, odd, even) using interpolation of S(-B)."""
    o = np.argsort(B)
    Bs, Ss = B[o], S[o]
    # keep the symmetric overlap only
    Bmax = min(abs(Bs[0]), abs(Bs[-1]))
    g = Bs[(Bs >= -Bmax) & (Bs <= Bmax)]
    if len(g) < 5:
        return None, None, None
    Sp = np.interp(g, Bs, Ss)
    Sm = np.interp(-g, Bs, Ss)
    return g, 0.5 * (Sp - Sm), 0.5 * (Sp + Sm)


def check_symmetries(df, cfg, rep, branches):
    """
    Parity is the cheapest and strongest physical check available.

    1f: R_xy^1f is the AHE, R_AHE * m_z, and m_z(-H) = -m_z(H) for a hard-axis
        loop -> the 1f signal must be ODD in H. Any even component is an offset
        (misaligned Hall contacts picking up R_xx), a thermoelectric voltage, or
        capacitive pickup. It carries no torque information and must be removed.

    2f: the DL-torque term goes as sqrt(1-m_z^2) * dR/dH, which is EVEN in H.
        A thermal/Nernst term with M reversing sign is typically ODD. So the
        parity decomposition of the 2f signal already separates the two
        contributions before any fitting.
    """
    out = {}
    br = branches[0] if branches else np.arange(len(df))
    B = df["B_T"].values[br]
    for tag, cX in [("1f", cfg.col_1f_X), ("2f", cfg.col_2f_X)]:
        g, odd, even = odd_even_decompose(B, df[cX].values[br])
        if g is None:
            continue
        a_odd, a_even = np.max(np.abs(odd)), np.max(np.abs(even))
        # An overall constant offset dominates 'even' trivially; report both raw
        # and offset-subtracted versions.
        even_ac = even - np.mean(even)
        out[tag] = {"amp_odd": float(a_odd), "amp_even": float(a_even),
                    "amp_even_ac": float(np.max(np.abs(even_ac)))}
        if tag == "1f":
            r = a_even / max(a_odd, 1e-30)
            msg = (f"odd amp {a_odd:.3e}, even amp {a_even:.3e} "
                   f"(even/odd = {r:.2f}).")
            if r < cfg.tol_even_odd_ratio_1f:
                rep.passed("1f parity (must be odd)", msg)
            else:
                rep.fail(
                    "1f parity (must be odd)",
                    msg + "\nThe AHE is odd in H for a hard-axis loop. A dominant even\n"
                    "part means your 1f channel is not dominated by the Hall signal:\n"
                    "most likely a longitudinal (R_xx) offset from misaligned Hall\n"
                    "probes. Subtract it, but also check whether it is large enough\n"
                    "that the 2f channel is dominated by the SAME offset modulated by\n"
                    "heating -- in which case the torque extraction is meaningless.",
                )
        else:
            rep.info(
                "2f parity decomposition",
                f"odd amp {out[tag]['amp_odd']:.3e}, even(ac) amp "
                f"{out[tag]['amp_even_ac']:.3e}.\n"
                "Even-in-H part -> damping-like torque channel.\n"
                "Odd-in-H part  -> thermal / Nernst-like channel.\n"
                "If the odd part dominates, you are measuring heat, not torque.",
            )
    return out


def check_hysteresis(df, cfg, rep, branches):
    """A hard-axis (out-of-plane) loop of an easy-plane film should be
    essentially anhysteretic. Hysteresis here means field misalignment (an
    in-plane component switching the magnetisation) or magnet remanence."""
    if len(branches) < 2:
        return
    b1, b2 = branches[0], branches[1]
    B1, S1 = df["B_T"].values[b1], df[cfg.col_1f_X].values[b1]
    B2, S2 = df["B_T"].values[b2], df[cfg.col_1f_X].values[b2]
    lo = max(min(B1.max(), B2.max()) * -1, min(B1.min(), B2.min()) * -1)
    g = np.linspace(max(B1.min(), B2.min()), min(B1.max(), B2.max()), 200)
    o1, o2 = np.argsort(B1), np.argsort(B2)
    d = np.interp(g, B1[o1], S1[o1]) - np.interp(g, B2[o2], S2[o2])
    span = np.ptp(np.concatenate([S1, S2]))
    r = np.max(np.abs(d)) / max(span, 1e-30)
    msg = f"max up/down difference = {r*100:.1f} % of the 1f signal span."
    if r < cfg.tol_hysteresis_frac:
        rep.passed("hysteresis (hard-axis loop)", msg)
    else:
        rep.warn(
            "hysteresis (hard-axis loop)",
            msg + "\nA field applied along the hard axis of an easy-plane film should\n"
            "give a reversible, anhysteretic loop. Openness implies (a) the field\n"
            "is misaligned and has an in-plane component that switches the\n"
            "magnetisation, or (b) magnet core remanence, or (c) drift/heating\n"
            "during the sweep. Check the field-calibration item above, and repeat\n"
            "the sweep in the opposite order to see whether it tracks time or field.",
        )


def check_signal_present(df, cfg, rep: Report) -> bool:
    """
    The most basic check of all, and the one most often skipped: does the
    1f Hall signal actually depend on the applied field?

    For an easy-plane Ni film swept out-of-plane to well beyond mu0*M_eff, the
    AHE must swing from -R_AHE to +R_AHE. If the field-dependent part of the
    1f signal is comparable to the point-to-point noise, then whatever fit
    follows is fitting noise, and every 'extracted' number downstream is
    meaningless no matter how small its formal error bar looks.
    """
    ok = True
    for tag, cX, csd in [("1f", cfg.col_1f_X, cfg.col_1f_sd),
                         ("2f", cfg.col_2f_X, cfg.col_2f_sd)]:
        x = df[cX].values
        # Field-dependent part = total variation minus the constant offset.
        var = np.ptp(x)
        noise = estimate_noise(x)
        snr = var / max(noise, 1e-30)
        msg = (f"field-dependent swing = {var:.3e} V on an offset of "
               f"{np.mean(x):.3e} V; point-to-point noise {noise:.3e} V "
               f"(swing/noise = {snr:.1f}).")
        if snr > 10:
            rep.passed(f"{tag} signal present", msg)
        else:
            ok = False
            rep.fail(
                f"{tag} signal present",
                msg + "\nThe signal does not vary with field beyond the noise. Everything\n"
                "downstream of this is a fit to noise. Before analysing anything:\n"
                "  - confirm the voltage taps are the Hall pair, not the current pair;\n"
                "  - confirm the current is actually flowing (measure R_xx);\n"
                "  - confirm the magnet is energised and the sample is in the field;\n"
                "  - confirm the sample is where you think it is in the field profile.",
            )
    return ok


def check_missing_metadata(cfg: Config, rep: Report):
    need = {
        "current_A": "AC current through the device -- without it V cannot be\n"
                     "converted to ohms and H_DL has no absolute scale.",
        "excitation_frequency_Hz": "drive frequency -- needed to interpret the 2f\n"
                                   "phase and to plan the frequency-dependence test.",
        "field_polar_misalignment_deg": "out-of-plane field alignment -- for an\n"
                                        "easy-plane magnet a 1-2 deg error puts a\n"
                                        "large in-plane field on the sample.",
        "hallbar_width_um": "needed for current density.",
        "ni_thickness_nm": "needed for current density and for xi_DL.",
        "nm_thickness_nm": "needed for the parallel-resistor current shunting.",
        "Ms_kA_per_m": "independent Ms to cross-check H_K_eff from the 1f fit.",
        "time_constant_s": "needed to verify settling per point.",
    }
    miss = [k for k in need if getattr(cfg, k) is None]
    if not miss:
        rep.passed("metadata", "all required metadata supplied.")
        return
    rep.warn(
        "metadata incomplete",
        "The following are not set; the affected results are reported in raw\n"
        "units or skipped entirely:\n"
        + "\n".join(f"  - {k}: {need[k]}" for k in miss),
    )


# =============================================================================
#  5.  FITTING
# =============================================================================
def model_1f(B, R_AHE, Hk, offset):
    return R_AHE * np.clip(B / Hk, -1.0, 1.0) + offset


def fit_first_harmonic(B, S, rep: Report):
    """
    Hard-axis AHE loop:  R_1f = R_AHE * clip(H/H_K_eff) + offset.

    H_K_eff obtained this way is the *effective* anisotropy field, i.e.
    mu0*M_eff = mu0*Ms - 2K_u/Ms. For Ni with no strong interface anisotropy it
    should come out near mu0*Ms ~ 0.6 T. If the fit returns something far from
    that, distrust the fit before you believe new physics.
    """
    span = np.ptp(S)
    if span <= 0 or not np.isfinite(span):
        rep.fail("1f fit", "1f signal has zero span; nothing to fit.")
        return None
    p0 = [span / 2, max(np.max(np.abs(B)) * 0.5, 1e-3), np.mean(S)]
    try:
        popt, pcov = curve_fit(model_1f, B, S, p0=p0, maxfev=40000)
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        rep.fail("1f fit", f"curve_fit failed: {e}")
        return None
    R_AHE, Hk, off = popt
    if Hk < 0:
        Hk, R_AHE = -Hk, -R_AHE
    resid = S - model_1f(B, *popt)
    rms = np.std(resid)
    r = rms / max(abs(R_AHE), 1e-30)
    res = {"R_AHE": float(R_AHE), "R_AHE_err": float(perr[0]),
           "Hk_eff_T": float(Hk), "Hk_eff_err_T": float(perr[1]),
           "offset": float(off), "resid_rms": float(rms), "resid_rel": float(r)}

    rep.info("1f fit", f"R_AHE = {R_AHE:.4e} +- {perr[0]:.1e}, "
                       f"mu0 H_K_eff = {Hk:.4f} +- {perr[1]:.4f} T, "
                       f"offset = {off:.4e}, residual = {r*100:.2f} % of R_AHE.")
    if np.max(np.abs(B)) < 1.15 * Hk:
        rep.warn(
            "1f saturation range",
            f"max |B| = {np.max(np.abs(B)):.3f} T is not comfortably above the\n"
            f"fitted H_K_eff = {Hk:.3f} T. H_K and R_AHE are then strongly\n"
            "correlated and the 'saturated' 2f background cannot be measured.\n"
            "Extend the sweep to at least 1.5 x H_K_eff.",
        )
    else:
        rep.passed("1f saturation range",
                   f"max |B| = {np.max(np.abs(B)):.2f} T > 1.15 x H_K_eff "
                   f"= {1.15*Hk:.2f} T; the saturated plateau is resolved.")
    if r > 0.05:
        rep.warn("1f fit quality",
                 f"residual is {r*100:.1f} % of R_AHE -- the clipped-linear\n"
                 "hard-axis model does not describe the data. Possible causes:\n"
                 "field misalignment, an unsubtracted R_xx offset, a second\n"
                 "magnetic phase, or a genuinely non-uniform magnetisation.")
    return res


def fit_second_harmonic(B, S2, fit1, rep: Report):
    """
    Least-squares fit of R_2f to the torque basis 0.5 * (dR_1f/dH_z) *
    sqrt(1 - m_z^2), giving H_DL (slope) and the non-torque background
    (intercept). See docs/harmonic-hall-theory.md for the full derivation.

    Two things the checks below verify:

    * The torque term VANISHES at out-of-plane saturation (sqrt(1-m_z^2) -> 0).
      Whatever 2f signal remains above H_K_eff is background, and it should be
      consistent with the fitted intercept. If it is not, the model is wrong.

    * The field-like torque and the Oersted field are along y and to first order
      do not modulate m_z, so they do NOT appear here. Do not report an H_FL
      from this geometry.
    """
    if fit1 is None:
        rep.fail("2f fit", "no valid 1f fit; cannot build the torque basis.")
        return None
    R_AHE, Hk = fit1["R_AHE"], fit1["Hk_eff_T"]
    mz = np.clip(B / Hk, -1.0, 1.0)
    dRdH = np.where(np.abs(B) < Hk, R_AHE / Hk, 0.0)
    basis = 0.5 * dRdH * np.sqrt(np.clip(1 - mz**2, 0, 1))

    if np.ptp(basis) <= 0:
        rep.fail("2f fit", "torque basis is identically zero.")
        return None

    A = np.vstack([basis, np.ones_like(basis)]).T
    coef, *_ = np.linalg.lstsq(A, S2, rcond=None)
    H_DL, bg = coef
    resid = S2 - A @ coef
    rms = np.std(resid)
    r = rms / max(np.ptp(S2), 1e-30)

    # Independent estimate of the background from the saturated region only.
    sat = np.abs(B) > 1.05 * Hk
    bg_sat = float(np.mean(S2[sat])) if sat.sum() > 3 else np.nan

    res = {"mu0_H_DL_T": float(H_DL), "background": float(bg),
           "background_from_saturation": bg_sat,
           "resid_rms": float(rms), "resid_rel": float(r)}
    rep.info("2f fit",
             f"mu0 H_DL = {H_DL:.4e} T (per the applied current), "
             f"background = {bg:.4e}, residual = {r*100:.1f} % of the 2f span.")

    if np.isfinite(bg_sat):
        d = abs(bg_sat - bg) / max(abs(np.ptp(S2)), 1e-30)
        if d < 0.1:
            rep.passed(
                "2f saturation null",
                f"fitted background {bg:.3e} agrees with the directly measured\n"
                f"saturated background {bg_sat:.3e} to {d*100:.1f} % of the span.\n"
                "The torque term does vanish at saturation, as it must.",
            )
        else:
            rep.fail(
                "2f saturation null",
                f"fitted background {bg:.3e} disagrees with the measured saturated\n"
                f"background {bg_sat:.3e} ({d*100:.0f} % of the span).\n"
                "The DL contribution to the AHE MUST go to zero when m || z. If the\n"
                "model cannot reproduce that, the 2f signal is not dominated by the\n"
                "DL torque: suspect thermal (ANE) contributions, an R_xx offset\n"
                "modulated by Joule heating, or a phase error mixing X and Y.",
            )
    if r > 0.15:
        rep.warn("2f fit quality",
                 f"residual {r*100:.0f} % of the span. The single-basis model is\n"
                 "insufficient. Add the parity decomposition (even-in-H = torque,\n"
                 "odd-in-H = thermal) and fit them separately, and repeat at\n"
                 "several frequencies to identify the thermal part.")
    return res


# =============================================================================
#  6.  PLOTTING
# =============================================================================
def make_plots(df, cfg, rep, branches, fit1, fit2, calibrated, fig: Optional[Figure] = None) -> Figure:
    """
    Draw the 3x3 analysis grid onto `fig` (creating one if not given) and
    return it -- caller decides what to do with it: run_analysis() saves it
    to a PNG for the CLI/script path, harmonic_hall_gui.py instead embeds the
    same Figure directly in a Qt canvas so the GUI and the saved PNG never
    drift apart.

    Always clears and rebuilds the whole figure rather than reusing existing
    axes, even when `fig` is passed in from a previous call (e.g. the GUI
    stepping to the next file): panel (2,1) below creates a twinx() axis,
    which `Axes.clear()` does not remove, so reusing axes across calls would
    leak a duplicate twin axis every time.
    """
    B = df["B_T"].values * 1e3  # mT for display
    X1, Y1 = df["_X1"].values, df["_Y1"].values
    X2, Y2 = df["_X2"].values, df["_Y2"].values
    unit = r"$R$ ($\Omega$)" if calibrated else r"$V$ (V)"

    if fig is None:
        fig = Figure(figsize=(16.5, 13.5))
    else:
        fig.clear()
    ax = np.array([[fig.add_subplot(3, 3, i * 3 + j + 1) for j in range(3)] for i in range(3)])
    fig.suptitle("Harmonic Hall: in-plane magnet, out-of-plane field sweep",
                 fontsize=14, y=0.995)

    def branchplot(a, y, **kw):
        for i, b in enumerate(branches):
            a.plot(B[b], y[b], marker="o", ms=2.5, lw=0.9,
                   label=f"branch {i+1}", **kw)

    # (0,0) 1f in-phase and quadrature
    a = ax[0, 0]
    branchplot(a, X1)
    a.plot(B, Y1, ".", ms=2, color="crimson", alpha=0.6, label=r"$Y_{1f}$")
    a.set_xlabel(r"$\mu_0 H_z$ (mT)"); a.set_ylabel(r"1f  " + unit)
    a.set_title(r"1f: $X$ (should be the full AHE) and $Y$ (should be $\approx 0$)")
    a.legend(fontsize=7); a.grid(alpha=0.3)

    # (0,1) 1f fit
    a = ax[0, 1]
    a.plot(B, X1, ".", ms=3, color="k", label="data")
    if fit1:
        bb = np.linspace(B.min(), B.max(), 500)
        a.plot(bb, model_1f(bb * 1e-3, fit1["R_AHE"], fit1["Hk_eff_T"],
                            fit1["offset"]), "-", color="tab:red",
               label=rf"fit: $\mu_0H_K^{{eff}}$={fit1['Hk_eff_T']:.3f} T")
        for s in (-1, 1):
            a.axvline(s * fit1["Hk_eff_T"] * 1e3, ls="--", color="tab:blue", lw=0.8)
    a.set_xlabel(r"$\mu_0 H_z$ (mT)"); a.set_ylabel(r"$R_{xy}^{1f}$ " + unit)
    a.set_title("1f hard-axis loop + clipped-linear fit")
    a.legend(fontsize=7); a.grid(alpha=0.3)

    # (0,2) phases
    a = ax[0, 2]
    th1 = np.degrees(np.arctan2(Y1, X1)); th2 = np.degrees(np.arctan2(Y2, X2))
    a.plot(B, th1, ".", ms=3, label=r"$\theta_{1f}$")
    a.plot(B, th2, ".", ms=3, label=r"$\theta_{2f}$")
    a.axhline(0, color="k", lw=0.6)
    _cm = lambda t: np.degrees(np.arctan2(np.mean(np.sin(np.radians(t))),
                                          np.mean(np.cos(np.radians(t)))))
    a.axhline(2 * _cm(th1), color="tab:green", ls="--", lw=1,
              label=r"$2\theta_{1f}$ (expected if pure delay)")
    a.set_xlabel(r"$\mu_0 H_z$ (mT)"); a.set_ylabel("phase (deg)")
    a.set_title(r"phase vs field (must be field-independent)")
    a.legend(fontsize=7); a.grid(alpha=0.3)

    # (1,0) 2f X and Y
    a = ax[1, 0]
    branchplot(a, X2)
    a.plot(B, Y2, ".", ms=2, color="crimson", alpha=0.6, label=r"$Y_{2f}$")
    a.set_xlabel(r"$\mu_0 H_z$ (mT)"); a.set_ylabel(r"2f  " + unit)
    a.set_title(r"2f: $X$ and $Y$")
    a.legend(fontsize=7); a.grid(alpha=0.3)

    # (1,1) parity decomposition of 2f
    a = ax[1, 1]
    br = branches[0] if branches else np.arange(len(df))
    g, odd, even = odd_even_decompose(df["B_T"].values[br], X2[br])
    if g is not None:
        a.plot(g * 1e3, even - np.mean(even), "-o", ms=2.5,
               label="even in $H$  (torque)")
        a.plot(g * 1e3, odd, "-o", ms=2.5, label="odd in $H$  (thermal/Nernst)")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel(r"$\mu_0 H_z$ (mT)"); a.set_ylabel(r"$R_{xy}^{2f}$ " + unit)
    a.set_title("2f parity decomposition")
    a.legend(fontsize=7); a.grid(alpha=0.3)

    # (1,2) 2f vs torque basis -> slope is H_DL
    a = ax[1, 2]
    if fit1:
        mz = np.clip(df["B_T"].values / fit1["Hk_eff_T"], -1, 1)
        dRdH = np.where(np.abs(df["B_T"].values) < fit1["Hk_eff_T"],
                        fit1["R_AHE"] / fit1["Hk_eff_T"], 0.0)
        basis = 0.5 * dRdH * np.sqrt(np.clip(1 - mz**2, 0, 1))
        a.plot(basis, X2, ".", ms=3, color="k", label="data")
        if fit2:
            bb = np.linspace(basis.min(), basis.max(), 50)
            a.plot(bb, fit2["mu0_H_DL_T"] * bb + fit2["background"], "-",
                   color="tab:red",
                   label=rf"slope $=\mu_0H_{{DL}}={fit2['mu0_H_DL_T']:.3e}$ T")
        a.set_xlabel(r"$\frac{1}{2}\,\frac{dR_{1f}}{dH_z}\sqrt{1-m_z^2}$")
        a.set_ylabel(r"$R_{xy}^{2f}$ " + unit)
        a.set_title("torque extraction (slope = $H_{DL}$)")
        a.legend(fontsize=7); a.grid(alpha=0.3)

    # (2,0) field calibration
    a = ax[2, 0]
    if cfg.col_magnet_I in df:
        a.plot(df[cfg.col_magnet_I], B, ".", ms=3)
        a.set_xlabel("magnet current (A)"); a.set_ylabel(r"$\mu_0 H_z$ (mT)")
        a.set_title("magnet calibration (look for a loop = core remanence)")
        a.grid(alpha=0.3)

    # (2,1) time series -> drift
    a = ax[2, 1]
    a.plot(df["t_s"] / 60, X1, ".", ms=3, label="1f X")
    a.set_xlabel("elapsed time (min)"); a.set_ylabel("1f " + unit)
    a2 = a.twinx(); a2.plot(df["t_s"] / 60, X2, ".", ms=3, color="tab:orange")
    a2.set_ylabel("2f " + unit, color="tab:orange")
    a.set_title("drift check: signal vs wall-clock time")
    a.grid(alpha=0.3)

    # (2,2) report text
    a = ax[2, 2]; a.axis("off")
    lines = []
    for it in rep.items:
        if it["level"] in ("FAIL", "WARN"):
            lines.append(f"[{it['level']}] {it['name']}")
    if not lines:
        lines = ["all checks passed"]
    a.text(0.0, 1.0, "Flagged checks\n" + "\n".join(lines[:22]),
           va="top", ha="left", fontsize=8, family="monospace")

    fig.tight_layout()
    return fig


# =============================================================================
#  7.  MAIN
# =============================================================================
@dataclass
class AnalysisResult:
    """Everything run_analysis() produces: the Config actually used (after
    apply_run_metadata() filled in whatever the file supplied), the sanity
    report, both fits (None if a fit failed), the drawn Figure, and the few
    extra flags run()'s JSON summary needs."""
    cfg: Config
    rep: Report
    fit1: Optional[dict]
    fit2: Optional[dict]
    fig: Figure
    phase: dict
    calibrated: bool
    signal_present: bool


def run_analysis(cfg: Config, fig: Optional[Figure] = None) -> AnalysisResult:
    """
    The full analysis pipeline: load, apply file metadata, run every sanity
    check, fit, and draw the plot grid -- but, unlike run() below, hand back
    the Figure instead of unconditionally saving+closing it, and don't print
    or write anything to disk. This is what run() and harmonic_hall_gui.py
    both call; run() is the thin CLI wrapper that adds printing and the
    sanity_report.txt/results.json/PNG writes, the GUI instead embeds the
    returned Figure live and lets the user step to the next file.

    Pass an existing `fig` (e.g. the GUI's persistent canvas figure) to draw
    into it instead of creating a new one -- see make_plots()'s docstring for
    why it's always cleared and rebuilt rather than updated in place.
    """
    rep = Report()
    df = load_data(cfg, rep)
    cfg = apply_run_metadata(df, cfg, rep)

    check_missing_metadata(cfg, rep)
    check_internal_consistency(df, cfg, rep)
    signal_ok = check_signal_present(df, cfg, rep)
    phase = check_phase(df, cfg, rep)
    check_field_calibration(df, cfg, rep)
    check_settling(df, cfg, rep)
    check_noise(df, cfg, rep)

    # Optional software phase rotation (default: none, so problems stay visible)
    X1, Y1 = rotate_phase(df[cfg.col_1f_X].values, df[cfg.col_1f_Y].values,
                          cfg.software_phase_rotation_1f_deg)
    X2, Y2 = rotate_phase(df[cfg.col_2f_X].values, df[cfg.col_2f_Y].values,
                          cfg.software_phase_rotation_2f_deg)
    X1, cal = to_resistance(X1, cfg); Y1, _ = to_resistance(Y1, cfg)
    X2, _ = to_resistance(X2, cfg);   Y2, _ = to_resistance(Y2, cfg)
    df["_X1"], df["_Y1"], df["_X2"], df["_Y2"] = X1, Y1, X2, Y2
    if not cal:
        rep.warn("units",
                 "current_A not set: all 'resistances' below are actually VOLTS\n"
                 "and mu0*H_DL is not on an absolute scale. Shapes and symmetries\n"
                 "are still meaningful; magnitudes are not.")

    branches = split_branches(df, rep)
    check_symmetries(df, cfg, rep, branches)
    check_hysteresis(df, cfg, rep, branches)

    br = branches[0] if branches else np.arange(len(df))
    B = df["B_T"].values[br]
    fit1 = fit_first_harmonic(B, X1[br], rep)
    fit2 = fit_second_harmonic(B, X2[br], fit1, rep) if fit1 else None

    fig = make_plots(df, cfg, rep, branches, fit1, fit2, cal, fig=fig)

    return AnalysisResult(
        cfg=cfg, rep=rep, fit1=fit1, fit2=fit2, fig=fig,
        phase=phase, calibrated=cal, signal_present=signal_ok,
    )


def write_outputs(result: AnalysisResult) -> dict[str, Path]:
    """
    Write the PNG, *_sanity_report.txt and *_results.json for one
    run_analysis() result to result.cfg.outdir, prefixed with the input
    CSV's stem so multiple files can share one outdir (as
    harmonic_hall_gui.py's "analyzed" folder does) without each analysis
    overwriting the last one's outputs. Shared by run() (CLI) and
    harmonic_hall_gui.py's "Save && Next" so the two never drift apart.

    Returns the three paths written, keyed 'png' / 'report' / 'results'.
    """
    cfg, rep = result.cfg, result.rep
    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg.csv_path).stem
    png = outdir / f"{stem}_harmonic_hall_analysis.png"
    report_path = outdir / f"{stem}_sanity_report.txt"
    results_path = outdir / f"{stem}_results.json"

    result.fig.savefig(png, dpi=150)
    report_path.write_text(rep.render())
    results_path.write_text(json.dumps(
        {"config": asdict(cfg), "phase": result.phase, "fit_1f": result.fit1,
         "fit_2f": result.fit2, "calibrated_to_ohms": result.calibrated,
         "signal_present": result.signal_present, "checks": rep.items},
        indent=2, default=str))
    return {"png": png, "report": report_path, "results": results_path}


def run(cfg: Config):
    """CLI/script entry point: run_analysis() + write_outputs() + printing."""
    result = run_analysis(cfg)
    print(result.rep.render())
    paths = write_outputs(result)
    print(f"\nwrote: {paths['png']}\n       {paths['report']}\n       {paths['results']}")
    return result.rep, result.fit1, result.fit2


if __name__ == "__main__":
    cfg = Config(csv_path="/Volumes/Untitled/data/chipL_dev21_8-20-10-6_800uA_20260807_111011.csv",
                 outdir="./report",
                 current_A=100e-6, excitation_frequency_Hz=67.33,
                 time_constant_s=0.3, filter_order=4)
    if len(sys.argv) > 1:
        cfg.csv_path = sys.argv[1]
    run(cfg)
