#!/usr/bin/env python3
"""
Dual-MFLI Harmonic-Hall Phase Calibration
==========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-05

Calibrates the leader MFLI's 1f reference phase against the sample's own
resistive Hall response, rather than a separate standard resistor, then
verifies the result is trustworthy before you rely on it for a real
harmonic-Hall measurement (see mfli_dual_harmonic.py).

Why the device is its own standard resistor:
  At 1st harmonic, a Hall bar's transverse voltage is dominated by the
  planar/anomalous/ordinary Hall effect — a purely resistive response,
  V ∝ R·I(t), that must be exactly in phase with the drive current. Any
  measured Y at 1f is therefore instrumental delay (cabling, contact
  impedance, source/ADC), not physics. Calibrating in situ, on the real
  device, captures cabling/contact/electronics delay that a dummy resistor
  measured separately would not reproduce.

What this script does, beyond the single-point null in
mfli_dual_harmonic.auto_null_phase():
  1. Ramps the magnet to a field near saturation (where the PHE/AHE 1f
     signal is large and well-behaved) and nulls the leader's 1f Y
     quadrature there.
  2. Sweeps the field bidirectionally with that phase fixed, and checks the
     null *holds* across the whole range — a residual that grows away from
     the calibration point is a red flag (nonlinearity, contact issues, or
     a real quadrature contribution) worth investigating before trusting
     the 2f data.
  3. Uses the same sweep to empirically identify which of X2f/Y2f carries
     the real, field-dependent 2nd-harmonic signal. V_2ω is a cosine, not
     a sine (from sin²(ωt) = ½ − ½cos(2ωt)), so the channel that was
     correct at 1f does not automatically carry over to 2f.
  4. Optionally repeats the calibration point at a few excitation
     amplitudes and a few frequencies, to help separate a genuine
     resistive/SOT-driven signal (linear in current, phase scales linearly
     with frequency) from Joule-heating/anomalous-Nernst contamination
     (which does neither).

Requirements: same as mfli_dual_harmonic.py
    pip install zhinst-core zhinst-utils numpy pandas pyvisa
"""

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional
import threading

import numpy as np
import pandas as pd
import zhinst.core as zi

from mfli.mfli_dual_harmonic import (
    AcquisitionConfig,  # noqa: F401  (re-exported for callers that want it)
    DemodConfig,
    FilterConfig,
    GaussmeterConfig,
    KepkoBOPGL,
    LakeShore475,
    MagnetConfig,
    MercuryITC,
    OutputConfig,
    PhaseCalibrationResult,
    TemperatureControllerConfig,
    _DATA_DIR,
    acquire_averaged,
    auto_null_phase,
    bidirectional_current_sweep,
    configure_demodulator,
    configure_output,
    connect,
    connect_device,
    connect_gaussmeter,
    connect_magnet,
    connect_temperature_controller,
    read_field_mT,
    read_temperature,
    set_magnet_current,
    setup_mds,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_output,
    shutdown_temperature_controller,
    sync_follower_oscillator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    """
    Bidirectional magnet-current sweep used both to check the 1f null holds
    across the field range and to identify the real 2f channel — the same
    swept data answers both questions (see run_field_sweep_diagnostic).
    """
    i_min_A: float         = -20.0
    i_max_A: float         = 20.0
    n_points: int          = 11     # points per sweep direction (see bidirectional_current_sweep)
    settling_time_s: float = 1.5    # dead time after each magnet step
    n_averages: int        = 20     # demod samples averaged per point


@dataclass
class AmplitudeCheckConfig:
    """Optional check: repeat the calibration point at several output amplitudes."""
    enabled:       bool        = False
    amplitudes_V:  List[float] = field(default_factory=lambda: [0.05, 0.1])
    n_averages:    int         = 20


@dataclass
class FrequencyCheckConfig:
    """Optional check: repeat the 1f null at several excitation frequencies."""
    enabled:         bool        = False
    frequencies_Hz:  List[float] = field(default_factory=lambda: [13.333, 17.777, 23.333])
    n_averages:      int         = 20
    max_iterations:  int         = 5
    tol_deg:         float       = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HoldCheckResult:
    """Whether the 1f Y-null established at one field point holds across a full sweep."""
    max_residual_ratio:  float          # worst |Y1f|/R1f seen anywhere in the sweep
    mean_residual_ratio: float
    worst_point_index:   int
    drift_flag:          bool           # True if max_residual_ratio exceeds tolerance
    df:                  pd.DataFrame   # full per-point sweep data


@dataclass
class ChannelIdResult:
    """Which of X2f/Y2f empirically carries the field-dependent signal."""
    x2f_structure:   float   # peak-to-peak(X2f) / noise(X2f) across the sweep
    y2f_structure:   float   # peak-to-peak(Y2f) / noise(Y2f) across the sweep
    signal_channel:  str     # "X" or "Y"
    structure_ratio: float   # dominant / other — how decisively one channel wins


@dataclass
class AmplitudeCheckPoint:
    amplitude_V: float
    current_A:   float   # amplitude_V / series_R_ohm
    signal_2f_V: float   # value in the identified 2f signal channel
    r2f_V:       float


@dataclass
class AmplitudeCheckResult:
    points:                      List[AmplitudeCheckPoint]
    df:                          pd.DataFrame
    gain_ratio:                  float   # (signal/I) at the highest amplitude ÷ at the lowest
    thermal_contamination_flag:  bool
    note:                        str


@dataclass
class FrequencyCheckPoint:
    frequency_Hz:       float
    optimal_phase_deg:  float
    converged:          bool


@dataclass
class FrequencyCheckResult:
    points:               List[FrequencyCheckPoint]
    df:                   pd.DataFrame
    slope_deg_per_Hz:     float
    r_squared:            float
    equivalent_delay_s:   float   # -slope / 360 — the fixed cable/electronics delay this implies
    linear_flag:          bool    # True if the phase-vs-frequency fit is good (R² > 0.98)
    note:                 str


@dataclass
class PhaseCalibrationReport:
    null_result:       PhaseCalibrationResult
    hold_check:        HoldCheckResult
    channel_id:        ChannelIdResult
    amplitude_check:   Optional[AmplitudeCheckResult]
    frequency_check:   Optional[FrequencyCheckResult]
    sweep_csv_path:    str


# ─────────────────────────────────────────────────────────────────────────────
# Field sweep diagnostic  ── shared data source for the hold check + channel ID
# ─────────────────────────────────────────────────────────────────────────────

def run_field_sweep_diagnostic(
    daq: zi.ziDAQServer,
    demod1_cfg: DemodConfig,
    demod2_cfg: DemodConfig,
    sweep_cfg: SweepConfig,
    magnet: KepkoBOPGL,
    magnet_cfg: MagnetConfig,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg: Optional[GaussmeterConfig] = None,
    stop_event: Optional[threading.Event] = None,
    on_point: Optional[Callable[[dict], None]] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg: Optional[TemperatureControllerConfig] = None,
) -> pd.DataFrame:
    """
    Sweep the magnet current bidirectionally (see bidirectional_current_sweep)
    and record 1f + 2f X/Y/R at every point, with the field measured live if
    a gaussmeter is given.

    `temp_ctrl`/`temp_cfg`, if given, log the sample/probe temperature
    (temperature_1_K / temperature_2_K) at each point via the shared
    MercuryiTC controller (see mercury_itc.py). Passing `temp_ctrl=None`
    (e.g. because the MercuryiTC isn't connected) simply leaves those
    columns empty — it's never a reason to stop the measurement.
    """
    currents_A = bidirectional_current_sweep(sweep_cfg.i_min_A, sweep_cfg.i_max_A, sweep_cfg.n_points)
    records: List[dict] = []

    for idx, I in enumerate(currents_A):
        if stop_event is not None and stop_event.is_set():
            log.info("Sweep diagnostic aborted after %d / %d points.", idx, len(currents_A))
            break

        log.info("── Sweep point %d / %d   I_magnet=%.4f A ──────────────────", idx + 1, len(currents_A), I)
        set_magnet_current(magnet, magnet_cfg, I)
        time.sleep(sweep_cfg.settling_time_s)

        d1 = acquire_averaged(daq, demod1_cfg, sweep_cfg.n_averages)
        d2 = acquire_averaged(daq, demod2_cfg, sweep_cfg.n_averages)
        field_mT = read_field_mT(gaussmeter, gauss_cfg) if gaussmeter is not None else None
        residual_ratio = abs(d1["y_mean"]) / d1["r_mean"] if d1["r_mean"] > 0 else float("nan")

        temp_1_K, temp_2_K = read_temperature(temp_ctrl, temp_cfg) \
            if temp_cfg is not None else (None, None)

        record = {
            "point_index":       idx,
            "magnet_current_A":  I,
            "magnet_field_mT":   field_mT,
            "temperature_1_K":   temp_1_K,
            "temperature_2_K":   temp_2_K,
            "1f_X_V":            d1["x_mean"],
            "1f_Y_V":            d1["y_mean"],
            "1f_R_V":            d1["r_mean"],
            "1f_X_std_V":        d1["x_std"],
            "1f_Y_std_V":        d1["y_std"],
            "1f_residual_ratio": residual_ratio,
            "2f_X_V":            d2["x_mean"],
            "2f_Y_V":            d2["y_mean"],
            "2f_R_V":            d2["r_mean"],
            "2f_X_std_V":        d2["x_std"],
            "2f_Y_std_V":        d2["y_std"],
        }
        records.append(record)
        log.info(
            "   1f  R=%.4e V  |Y|/R=%.2e%s",
            d1["r_mean"], residual_ratio,
            f"  (B={field_mT:.2f} mT)" if field_mT is not None else "",
        )
        if on_point is not None:
            on_point(record)

    return pd.DataFrame(records)


def evaluate_hold_check(df: pd.DataFrame, tol_ratio: float = 0.02) -> HoldCheckResult:
    """
    Check whether the 1f Y-null holds across the *entire* field sweep, not
    just the single point auto_null_phase() was run at. A residual that
    stays small everywhere means the calibrated phase reflects a fixed
    instrumental delay; one that grows at some fields is worth investigating
    (possible nonlinearity, contact issues, or a real quadrature
    contribution) before trusting the 2f data.

    tol_ratio is on the same |Y|/R scale auto_null_phase itself converges
    against, not a phase in degrees, so it stays meaningful close to R≈0.
    """
    ratios = df["1f_residual_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        raise RuntimeError("No usable 1f data in the sweep diagnostic (all R=0, or no points collected).")
    max_r = float(ratios.max())
    return HoldCheckResult(
        max_residual_ratio=max_r,
        mean_residual_ratio=float(ratios.mean()),
        worst_point_index=int(ratios.idxmax()),
        drift_flag=max_r > tol_ratio,
        df=df,
    )


def identify_2f_channel(df: pd.DataFrame) -> ChannelIdResult:
    """
    Empirically determine whether X2f or Y2f carries the physical,
    field-dependent 2nd-harmonic signal — don't assume it's X just because
    X1f was the resistive/in-phase channel at 1f; V_2ω is a cosine, not a
    sine, so the channel labels don't necessarily carry over.

    "Structured" is each channel's peak-to-peak variation across the field
    sweep divided by its own point-to-point noise (acquire_averaged's
    x_std/y_std) — a signal-to-noise-like ratio, rather than raw
    peak-to-peak amplitude, since a channel can have a larger raw
    offset/amplitude yet still be the flat, noise-dominated one.
    """
    def structure(val_col: str, std_col: str) -> float:
        ptp = float(np.ptp(df[val_col].to_numpy()))
        noise = float(df[std_col].mean())
        return ptp / noise if noise > 0 else ptp

    x_struct = structure("2f_X_V", "2f_X_std_V")
    y_struct = structure("2f_Y_V", "2f_Y_std_V")
    lo = min(x_struct, y_struct)
    return ChannelIdResult(
        x2f_structure=x_struct,
        y2f_structure=y_struct,
        signal_channel="X" if x_struct >= y_struct else "Y",
        structure_ratio=(max(x_struct, y_struct) / lo) if lo > 0 else float("inf"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional scaling checks
# ─────────────────────────────────────────────────────────────────────────────

def run_amplitude_check(
    daq: zi.ziDAQServer,
    out_cfg: OutputConfig,
    demod2_cfg: DemodConfig,
    cfg: AmplitudeCheckConfig,
    signal_channel: str,
    settling_time_s: float,
) -> AmplitudeCheckResult:
    """
    Repeat the measurement at each amplitude in cfg.amplitudes_V and compare
    how the identified 2f signal channel scales with drive current.

    Why this separates real physics from an artifact: the resistive/SOT-
    driven 2f term comes from I·δθ with δθ ∝ I, so it should scale
    ~linearly with amplitude (signal/I ≈ constant). A Joule-heating-driven
    anomalous Nernst contribution instead tracks ΔT, which grows faster
    than linearly with I (ΔT ∝ I²R), so signal/I creeping up at higher
    amplitude flags thermal contamination — no phase adjustment removes it.

    Restores the original output amplitude before returning.
    """
    if len(cfg.amplitudes_V) < 2:
        raise ValueError("amplitude_check needs at least 2 amplitudes to compare scaling.")

    points: List[AmplitudeCheckPoint] = []
    for amp in cfg.amplitudes_V:
        configure_output(daq, replace(out_cfg, amplitude_V=amp))
        time.sleep(settling_time_s)
        d2 = acquire_averaged(daq, demod2_cfg, cfg.n_averages)
        signal = d2["x_mean"] if signal_channel == "X" else d2["y_mean"]
        I = amp / out_cfg.series_R_ohm
        points.append(AmplitudeCheckPoint(amplitude_V=amp, current_A=I, signal_2f_V=signal, r2f_V=d2["r_mean"]))
        log.info("   Amplitude check: V=%.4f V  I≈%.3e A  2f[%s]=%.4e V", amp, I, signal_channel, signal)

    configure_output(daq, out_cfg)
    time.sleep(settling_time_s)

    df = pd.DataFrame([p.__dict__ for p in points])
    gains = df["signal_2f_V"] / df["current_A"]
    lo_gain, hi_gain = float(gains.iloc[0]), float(gains.iloc[-1])
    gain_ratio = hi_gain / lo_gain if lo_gain != 0 else float("inf")
    thermal_flag = abs(gain_ratio - 1.0) > 0.2   # >20% deviation from constant gain

    note = (
        "signal/I roughly constant across amplitudes — consistent with a resistive/SOT origin."
        if not thermal_flag else
        "signal/I changes by more than 20% across amplitudes — check for Joule-heating/ANE "
        "contamination rather than a phase error."
    )
    return AmplitudeCheckResult(points=points, df=df, gain_ratio=gain_ratio,
                                 thermal_contamination_flag=thermal_flag, note=note)


def run_frequency_check(
    daq: zi.ziDAQServer,
    out_cfg: OutputConfig,
    demod1_cfg: DemodConfig,
    follower: str,
    cfg: FrequencyCheckConfig,
    settling_time_s: float,
) -> FrequencyCheckResult:
    """
    Repeat the 1f phase null at each frequency in cfg.frequencies_Hz and
    check whether the optimal phaseshift scales linearly with frequency.

    A fixed instrumental time delay t_d (cables, source, ADC group delay)
    produces a phase lag that grows linearly with frequency:
    phase_deg = -360 * f * t_d. A good linear fit means the calibration is
    explained by pure electrical delay — reassuring. A poor fit suggests a
    genuinely frequency-dependent (e.g. thermal) response that phase
    calibration alone can't separate out.

    Restores the original frequency and re-nulls before returning, so the
    device isn't left mid-sweep at the last test frequency.
    """
    if len(cfg.frequencies_Hz) < 2:
        raise ValueError("frequency_check needs at least 2 frequencies to fit a slope.")

    points: List[FrequencyCheckPoint] = []
    for f in cfg.frequencies_Hz:
        f_out_cfg = replace(out_cfg, frequency_Hz=f)
        configure_output(daq, f_out_cfg)
        sync_follower_oscillator(daq, f_out_cfg, follower)
        time.sleep(settling_time_s)
        result = auto_null_phase(daq, demod1_cfg, n_averages=cfg.n_averages,
                                  max_iterations=cfg.max_iterations, tol_deg=cfg.tol_deg)
        points.append(FrequencyCheckPoint(frequency_Hz=f, optimal_phase_deg=result.phase_after_deg,
                                           converged=result.converged))
        log.info("   Frequency check: f=%.4f Hz  optimal phase=%.4f°  (%s)",
                 f, result.phase_after_deg, "converged" if result.converged else "NOT converged")

    configure_output(daq, out_cfg)
    sync_follower_oscillator(daq, out_cfg, follower)
    time.sleep(settling_time_s)
    auto_null_phase(daq, demod1_cfg, n_averages=cfg.n_averages,
                     max_iterations=cfg.max_iterations, tol_deg=cfg.tol_deg)

    df = pd.DataFrame([p.__dict__ for p in points])
    freqs = df["frequency_Hz"].to_numpy()
    # Unwrap before fitting so a ±180° wraparound between adjacent test
    # frequencies doesn't masquerade as a huge, wrong slope.
    phases = np.degrees(np.unwrap(np.radians(df["optimal_phase_deg"].to_numpy())))
    slope, intercept = np.polyfit(freqs, phases, 1)
    pred = slope * freqs + intercept
    ss_res = float(np.sum((phases - pred) ** 2))
    ss_tot = float(np.sum((phases - phases.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    equivalent_delay_s = -slope / 360.0
    linear_flag = r_squared > 0.98

    note = (
        f"Optimal phase scales ~linearly with frequency (R²={r_squared:.4f}) — consistent "
        f"with a fixed ~{equivalent_delay_s * 1e6:.2f} µs instrumental delay."
        if linear_flag else
        f"Optimal phase vs. frequency fit is poor (R²={r_squared:.4f}) — consider a genuine "
        "frequency-dependent (e.g. thermal) response rather than pure electrical delay."
    )
    return FrequencyCheckResult(points=points, df=df, slope_deg_per_Hz=float(slope),
                                 r_squared=r_squared, equivalent_delay_s=equivalent_delay_s,
                                 linear_flag=linear_flag, note=note)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_calibration(
    daq: zi.ziDAQServer,
    leader: str,
    follower: str,
    out_cfg: OutputConfig,
    demod1_cfg: DemodConfig,
    demod2_cfg: DemodConfig,
    sweep_cfg: SweepConfig,
    magnet: KepkoBOPGL,
    magnet_cfg: MagnetConfig,
    calibration_current_A: float,
    null_n_averages: int,
    null_max_iterations: int,
    null_tol_deg: float,
    output_csv: str,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg: Optional[GaussmeterConfig] = None,
    hold_tol_ratio: float = 0.02,
    amplitude_check_cfg: Optional[AmplitudeCheckConfig] = None,
    frequency_check_cfg: Optional[FrequencyCheckConfig] = None,
    stop_event: Optional[threading.Event] = None,
    on_point: Optional[Callable[[dict], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg: Optional[TemperatureControllerConfig] = None,
) -> PhaseCalibrationReport:
    """
    Run the full phase-calibration procedure end to end:

      1. Ramp the magnet to `calibration_current_A` (ideally saturation, so
         the PHE/AHE 1f signal is large and well-behaved) and null the
         leader's 1f Y quadrature there.
      2. Sweep the field bidirectionally with the now-fixed phase and check
         the null *holds* across the whole range, not just the calibration
         point — the same sweep also gives the data needed to...
      3. ...empirically identify which of X2f/Y2f carries the real,
         field-dependent 2nd-harmonic signal.
      4. Optionally repeat the calibration point at a few current
         amplitudes and a few excitation frequencies, to help separate real
         electrical delay / resistive SOT response from Joule-heating/ANE
         contamination.

    Returns a PhaseCalibrationReport bundling every step's result; the raw
    per-point sweep data is also written to `output_csv` so a crash mid-run
    doesn't lose it.
    """
    def status(msg: str) -> None:
        log.info(msg)
        if on_status is not None:
            on_status(msg)

    status(f"Ramping magnet to calibration point {calibration_current_A:.4f} A ...")
    set_magnet_current(magnet, magnet_cfg, calibration_current_A)
    time.sleep(sweep_cfg.settling_time_s)

    status("Nulling 1f Y quadrature (leader demod phaseshift) ...")
    null_result = auto_null_phase(daq, demod1_cfg, n_averages=null_n_averages,
                                   max_iterations=null_max_iterations, tol_deg=null_tol_deg)
    if not null_result.converged:
        log.warning(
            "Phase null did not fully converge after %d iteration(s) (|Y|/R=%.2e) — "
            "check cabling/contacts before trusting the rest of this calibration.",
            null_result.iterations, null_result.residual_ratio,
        )

    status("Sweeping field to verify the null holds and identify the 2f signal channel ...")
    df = run_field_sweep_diagnostic(
        daq, demod1_cfg, demod2_cfg, sweep_cfg, magnet, magnet_cfg,
        gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
        stop_event=stop_event, on_point=on_point,
        temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    hold_check = evaluate_hold_check(df, tol_ratio=hold_tol_ratio)
    if hold_check.drift_flag:
        log.warning(
            "1f null does NOT hold across the full sweep — max |Y|/R=%.2e at point %d. "
            "Possible nonlinearity, contact issues, or a real quadrature contribution — "
            "investigate before trusting the 2f data.",
            hold_check.max_residual_ratio, hold_check.worst_point_index,
        )

    channel_id = identify_2f_channel(df)
    status(
        f"2f signal channel identified: {channel_id.signal_channel}2f "
        f"(structure ratio {channel_id.structure_ratio:.1f}x over the other channel)."
    )

    amplitude_check = None
    if amplitude_check_cfg is not None and amplitude_check_cfg.enabled:
        status("Running current-amplitude scaling check ...")
        amplitude_check = run_amplitude_check(
            daq, out_cfg, demod2_cfg, amplitude_check_cfg,
            channel_id.signal_channel, sweep_cfg.settling_time_s,
        )
        if amplitude_check.thermal_contamination_flag:
            log.warning("Amplitude check: %s", amplitude_check.note)

    frequency_check = None
    if frequency_check_cfg is not None and frequency_check_cfg.enabled:
        status("Running frequency scaling check ...")
        frequency_check = run_frequency_check(
            daq, out_cfg, demod1_cfg, follower, frequency_check_cfg, sweep_cfg.settling_time_s,
        )
        if not frequency_check.linear_flag:
            log.warning("Frequency check: %s", frequency_check.note)

    status("Phase calibration complete.")
    return PhaseCalibrationReport(
        null_result=null_result, hold_check=hold_check, channel_id=channel_id,
        amplitude_check=amplitude_check, frequency_check=frequency_check,
        sweep_csv_path=output_csv,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def format_report(report: PhaseCalibrationReport) -> str:
    n, h, c = report.null_result, report.hold_check, report.channel_id
    lines = [
        "=" * 72,
        "PHASE CALIBRATION REPORT",
        "=" * 72,
        f"1f null:         {n.phase_before_deg:.4f}° → {n.phase_after_deg:.4f}°  "
        f"({n.iterations} iteration(s), {'converged' if n.converged else 'DID NOT CONVERGE'}, "
        f"|Y|/R={n.residual_ratio:.2e})",
        f"Null holds?      max |Y|/R over full sweep = {h.max_residual_ratio:.2e}  "
        f"(mean {h.mean_residual_ratio:.2e})  "
        f"{'-> DRIFT DETECTED, see log' if h.drift_flag else '-> OK'}",
        f"2f channel:      {c.signal_channel}2f carries the field-dependent signal  "
        f"(structure ratio {c.structure_ratio:.1f}x)",
    ]
    if report.amplitude_check is not None:
        a = report.amplitude_check
        lines.append(f"Amplitude check: gain ratio (hi/lo amplitude) = {a.gain_ratio:.3f}  — {a.note}")
    if report.frequency_check is not None:
        f = report.frequency_check
        lines.append(
            f"Frequency check: slope={f.slope_deg_per_Hz:.4f} °/Hz  R²={f.r_squared:.4f}  "
            f"equivalent delay={f.equivalent_delay_s * 1e6:.2f} µs — {f.note}"
        )
    lines.append(f"Sweep data saved to: {report.sweep_csv_path}")
    lines.append("=" * 72)
    return "\n".join(lines)


def print_report(report: PhaseCalibrationReport) -> None:
    print("\n" + format_report(report) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── configure your devices and calibration point here ──────────
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    LEADER   = "dev7885"    # Current source + 1f measurement
    FOLLOWER = "dev7886"    # 2f measurement

    daq = connect("localhost", 8004)
    connect_device(daq, LEADER,   interface="1GbE")
    connect_device(daq, FOLLOWER, interface="1GbE")
    setup_mds(daq, leader=LEADER, follower=FOLLOWER)

    out_cfg = OutputConfig(
        device       = LEADER,
        frequency_Hz = 17.777,      # Hz — avoid 50/60 Hz harmonics
        amplitude_V  = 0.1,         # V
        series_R_ohm = 10000,       # Ω
    )
    configure_output(daq, out_cfg)
    sync_follower_oscillator(daq, out_cfg, FOLLOWER)

    shared_filter = FilterConfig(time_constant_s=0.3, order=4, sinc_filter=True)
    demod1_cfg = DemodConfig(device=LEADER,   demod_index=0, harmonic=1,
                              input_range_V=1.0, sample_rate_Hz=857.0, filter=shared_filter)
    configure_demodulator(daq, demod1_cfg)
    demod2_cfg = DemodConfig(device=FOLLOWER, demod_index=0, harmonic=2,
                              input_range_V=1.0, sample_rate_Hz=857.0, filter=shared_filter)
    configure_demodulator(daq, demod2_cfg)

    magnet_cfg = MagnetConfig(
        visa_resource        = "GPIB0::6::INSTR",
        current_limit_A      = 35,     # ← safe continuous limit for your magnet
        voltage_compliance_V = 15.0,
        ramp_step_A          = 0.1,
        ramp_delay_s          = 0.05,
    )
    magnet = connect_magnet(magnet_cfg)

    gauss_cfg = GaussmeterConfig(
        visa_resource = "GPIB0::12::INSTR",   # ← your Lake Shore 475's GPIB address
        unit          = "T",
        n_averages    = 10,
        read_delay_s  = 0.05,
    )
    gaussmeter = connect_gaussmeter(gauss_cfg)

    # ── Temperature (Oxford Instruments MercuryiTC, optional) ────────────────
    # Not every rig has one, and not every MercuryiTC has two probes wired up
    # — connect_temperature_controller() returns None rather than raising if
    # it can't be reached, and the measurement runs fine either way.
    temp_cfg = TemperatureControllerConfig(
        visa_resource = "TCPIP0::192.168.1.5::7020::SOCKET",  # ← set to your iTC's address
        sensor_uids   = ("MB1.T1",),   # ← 1 or 2 board UIDs, e.g. ("MB1.T1", "DB5.T1")
    )
    temp_ctrl = connect_temperature_controller(temp_cfg)

    # Sweep used to verify the null holds and identify the 2f channel — pick
    # the same range you'd actually use for a Hall-effect measurement.
    sweep_cfg = SweepConfig(
        i_min_A         = -20.0,
        i_max_A         = 20.0,
        n_points        = 11,
        settling_time_s = 1.5,
        n_averages      = 20,
    )

    # Optional extra checks (both off by default — enable to run them):
    amplitude_check_cfg = AmplitudeCheckConfig(
        enabled      = False,
        amplitudes_V = [0.05, 0.1],
        n_averages   = 20,
    )
    frequency_check_cfg = FrequencyCheckConfig(
        enabled         = False,
        frequencies_Hz  = [13.333, 17.777, 23.333],
        n_averages      = 20,
        max_iterations  = 5,
        tol_deg         = 0.02,
    )

    output_csv = str(_DATA_DIR / f"phase_calibration_{datetime.now():%Y%m%d_%H%M%S}.csv")

    try:
        report = run_phase_calibration(
            daq, LEADER, FOLLOWER, out_cfg, demod1_cfg, demod2_cfg, sweep_cfg,
            magnet, magnet_cfg,
            calibration_current_A = 20.0,   # ← pick a point near saturation (e.g. i_max_A)
            null_n_averages       = 20,
            null_max_iterations   = 5,
            null_tol_deg          = 0.02,
            output_csv            = output_csv,
            gaussmeter            = gaussmeter,
            gauss_cfg             = gauss_cfg,
            amplitude_check_cfg   = amplitude_check_cfg,
            frequency_check_cfg   = frequency_check_cfg,
            temp_ctrl             = temp_ctrl,
            temp_cfg              = temp_cfg,
        )
        print_report(report)
    finally:
        shutdown_output(daq, out_cfg)
        shutdown_magnet(magnet, magnet_cfg)
        shutdown_gaussmeter(gaussmeter)
        shutdown_temperature_controller(temp_ctrl)


if __name__ == "__main__":
    main()
