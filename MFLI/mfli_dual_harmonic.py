#!/usr/bin/env python3
"""
Dual MFLI Lock-in Harmonic Measurement with MDS
================================================
Drives a current through the sample via:
    V_out (MFLI_1 Signal Output) ──[ R_series ]──> sample

Measures two differential voltage signals:
    MFLI_1 Signal Input 1 → Demodulator at 1f  (1st harmonic)
    MFLI_2 Signal Input 1 → Demodulator at 2f  (2nd harmonic)

Both MFLIs are synchronized via the Multi-Device Synchronization (MDS)
module so their oscillators share the same reference phase.

IMPORTANT — oscillator frequency is NOT shared automatically by MDS:
MDS synchronizes the sample clock and start trigger across devices, not
the per-device oscillator frequency *value*. Each device's local
oscillator still free-runs at whatever frequency you set it to. If the
follower's demodulator frequency doesn't exactly match the leader's
excitation frequency, its 2f output will show a slow beat instead of a
stable phasor. sync_follower_oscillator() below sets this explicitly —
don't skip it.

Magnetic field sweep:
  A Kepco BOP-GL bipolar power supply (see kepco_magnet.KepkoBOPGL) drives
  current through an electromagnet to provide the field axis for e.g. a
  Hall-effect measurement. bidirectional_current_sweep() builds an
  up-then-down current list so hysteresis is visible in the 1f/2f data.
  The actual field at the sample is measured directly with a Lake Shore
  475 DSP Gaussmeter (see lakeshore475.LakeShore475) at each point, rather
  than inferred from the magnet current via a calibration constant.

Extensibility:
  Add new sweep variables to MeasurementPoint and a corresponding
  set_action callable — the run_measurement loop handles the rest.

Run metadata (see build_run_metadata()):
  Every row of the output CSV also carries the run-level metadata a
  harmonic-Hall analysis needs to go from raw voltages to an absolute
  quantity: excitation frequency and current (both peak and RMS, with the
  MFLI's peak-amplitude / RMS-demodulator conventions spelled out), each
  demodulator's filter time-constant/order and its reference phase as
  actually programmed (read live, since auto_null_phase() or
  set_excitation_frequency() can change it mid-run), plus Hall bar
  dimensions and the external-field angle from the out-of-plane axis if
  supplied via SampleGeometryConfig (optional — None leaves those columns
  blank rather than blocking a run; the TUI exposes them as optional
  fields). set_excitation_frequency() lets a MeasurementPoint.set_action
  drive a frequency sweep, e.g. to repeat the sweep at ≥3 frequencies and
  separate instrumental phase from thermal quadrature.

Requirements:
    pip install zhinst-core zhinst-utils numpy pandas pyvisa
"""

import sys
import time
import math
import logging
import threading
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

import zhinst.core as zi
import zhinst.utils as ziutils

# The MFLI DAQ-server helpers, Kepco magnet and Lake Shore 475 drivers live
# in the shared bridge/instruments folder — add it to sys.path directly
# (it's not installed as a normal package).
_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent / "instruments"
if str(_INSTRUMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTRUMENTS_DIR))
from mfli_daq import (  # noqa: E402
    connect,
    connect_device,
    setup_mds,
    check_mds_status,
    sync_follower_oscillator,
    acquire_averaged,
)
from kepco_magnet import (  # noqa: E402
    KepkoBOPGL,
    MagnetConfig,
    connect_magnet,
    set_magnet_current,
    shutdown_magnet,
)
from lakeshore475 import (  # noqa: E402
    LakeShore475,
    GaussmeterConfig,
    connect_gaussmeter,
    read_field_mT,
    shutdown_gaussmeter,
)
from mercury_itc import (  # noqa: E402
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    shutdown_temperature_controller,
)

# Data lives outside "bridge" (a sibling of it) so measurement output never
# ends up inside the git-tracked source tree.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses  ── change all your parameters here ──────────────
# ─────────────────────────────────────────────────────────────────────────────
# MagnetConfig and GaussmeterConfig are the same shape as bridge/DC's — they
# live in bridge/instruments/ (see the kepco_magnet / lakeshore475 imports
# above) instead of being redefined here.

@dataclass
class OutputConfig:
    """Voltage source → current source configuration."""
    device: str         = "dev1234"   # MFLI acting as leader + current source
    out_ch: int         = 0           # Signal Output index (0-based)
    osc_index: int      = 0           # Oscillator index
    frequency_Hz: float = 317.3       # Excitation frequency  [Hz]
                                      #   Recommended ~300-1000 Hz: below that
                                      #   you're in the 1/f noise region of
                                      #   contacts/amplifier/thermal drift.
                                      #   Also avoid exact multiples of 50/60 Hz.
    amplitude_V: float  = 0.1         # Output amplitude      [V, peak)
    series_R_ohm: float = 1e6         # Series resistor       [Ω]
                                      #   → I_exc ≈ amplitude_V / series_R_ohm


@dataclass
class FilterConfig:
    """Lock-in filter parameters (shared shape, set per demodulator)."""
    time_constant_s: float = 0.3      # Low-pass time constant  [s]
    order: int             = 4        # Filter order  (1–8)
    sinc_filter: bool      = True     # 4th-order sinc on top (extra harmonic rejection)


@dataclass
class DemodConfig:
    """One demodulator channel (1f or 2f)."""
    device: str                        # Device ID  (leader or follower)
    demod_index: int                   # Demodulator index on that device (0-based)
    harmonic: int                      # 1 → 1f,  2 → 2f
    osc_index: int      = 0            # Oscillator to lock to
    input_ch: int       = 0            # Signal Input index (0-based)
    differential: bool  = True         # Enable differential (IN+ / IN−) mode
    ac_coupling: bool   = True         # AC-couple the input
    input_range_V: float = 1.0         # Input range  [V]
    sample_rate_Hz: float = 857.0      # Demodulator output rate  [Sa/s]
                                       #   must be > 2× highest signal bandwidth
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class AcquisitionConfig:
    """Timing and averaging parameters."""
    settling_time_s: float = 1.5      # Dead-time after parameter change  [s]
                                      #   Rule of thumb: ≥ 5 × TC for a 1st-order
                                      #   filter, ≥ 10 × TC for 4th-order (the
                                      #   default here) — a higher-order filter
                                      #   settles more slowly per time constant.
    n_averages: int        = 50       # Number of independent demod samples to average
    output_file: str       = "lockin_data.csv"


@dataclass
class SampleGeometryConfig:
    """
    Sample and field geometry needed to turn raw 1f/2f voltages into
    absolute quantities (resistivity, spin Hall / damping-like field)
    after the fact. None of this is readable from any instrument, so
    every field defaults to None (left blank in the saved metadata)
    rather than blocking a run — the TUI exposes these as optional
    fields for exactly that reason.
    """
    hall_bar_length_um:       Optional[float] = None  # Current-path length between voltage probes
    hall_bar_width_um:        Optional[float] = None  # Channel width
    hall_bar_thickness_nm:    Optional[float] = None  # Film/channel thickness
    field_angle_from_oop_deg: Optional[float] = None  # External field angle from the out-of-plane
                                                        # (film normal) axis; 0° = fully out-of-plane,
                                                        # 90° = in-plane


@dataclass
class PhaseCalibrationResult:
    """Outcome of auto_null_phase() — see that function's docstring."""
    phase_before_deg: float    # demod phaseshift node value before calibration
    phase_after_deg:  float    # demod phaseshift node value on return (already applied)
    iterations:       int      # number of measure/adjust cycles actually run
    x_V:              float    # last-measured X at the final phase
    y_V:              float    # last-measured Y at the final phase (the nulled quadrature)
    r_V:              float    # last-measured R
    residual_ratio:   float    # |Y| / R at the final phase — ~0 means well nulled
    converged:        bool     # whether |Y|/R reached tol_deg before max_iterations


# ─────────────────────────────────────────────────────────────────────────────
# Instrument setup helpers
# ─────────────────────────────────────────────────────────────────────────────
# connect, connect_device, setup_mds and sync_follower_oscillator are
# imported from bridge/instruments/mfli_daq.py above unchanged — every MFLI
# program connects to the LabOne data server, MDS-syncs a leader/follower
# pair, and re-syncs the follower's oscillator the same way. Only
# configure_output (this program's pure-AC excitation topology, not shared
# with e.g. mfli_diff_resistance_vs_bias.py's AC+DC-bias one) and
# set_excitation_frequency (dual_harmonic-specific) stay local.

def configure_output(daq: zi.ziDAQServer, cfg: OutputConfig) -> None:
    """Set up the voltage output that drives the current through the sample."""
    d = cfg.device

    # The sigouts/N/{amplitudes,enables} node index is a hardware "mixer
    # channel", NOT the demodulator index — it depends on device type and
    # installed options (e.g. a base MFLI without the MD/MF option exposes
    # mixer channel 1 for output 0, not 0). Ask zhinst.utils to resolve it
    # rather than hardcoding it, or you'll get a NotFoundError like
    # "Could not find any node that matches path .../amplitudes/0".
    discovery = zi.ziDiscovery()
    props = discovery.get(discovery.find(d))
    mixer_c = ziutils.default_output_mixer_channel(props, cfg.out_ch)

    daq.setDouble(f"/{d}/oscs/{cfg.osc_index}/freq",               cfg.frequency_Hz)
    daq.setDouble(f"/{d}/sigouts/{cfg.out_ch}/amplitudes/{mixer_c}", cfg.amplitude_V)
    daq.setDouble(f"/{d}/sigouts/{cfg.out_ch}/range",              max(0.01, cfg.amplitude_V * 2))
    daq.setInt(   f"/{d}/sigouts/{cfg.out_ch}/on",                 1)
    daq.setInt(   f"/{d}/sigouts/{cfg.out_ch}/enables/{mixer_c}",  1)
    daq.setInt(   f"/{d}/sigouts/{cfg.out_ch}/imp50",              0)   # High-Z output
    daq.sync()
    I_nA = cfg.amplitude_V / cfg.series_R_ohm * 1e9
    log.info(
        "Output: %s  f=%.4f Hz  Vpp=%.4f V  R=%.2e Ω  → I≈%.3f nA  (mixer_c=%d)",
        d, cfg.frequency_Hz, cfg.amplitude_V, cfg.series_R_ohm, I_nA, mixer_c,
    )


def set_excitation_frequency(
    daq: zi.ziDAQServer,
    out_cfg: OutputConfig,
    frequency_Hz: float,
    follower: Optional[str] = None,
    follower_osc_index: int = 0,
) -> None:
    """
    Change the excitation frequency mid-run and mutate `out_cfg` in place
    so build_run_metadata() and every downstream log line see the value
    actually in effect — use this from a MeasurementPoint.set_action to
    run a frequency sweep (repeating the sweep at ≥3 frequencies is the
    only way to separate instrumental phase from thermal quadrature; see
    the module docstring).

    If `follower` is given, re-syncs its oscillator too — see
    sync_follower_oscillator()'s docstring for why that's required on
    every frequency change, not just once at startup.
    """
    out_cfg.frequency_Hz = frequency_Hz
    daq.setDouble(f"/{out_cfg.device}/oscs/{out_cfg.osc_index}/freq", frequency_Hz)
    daq.sync()
    if follower is not None:
        sync_follower_oscillator(daq, out_cfg, follower, follower_osc_index)
    log.info("Excitation frequency set to %.4f Hz", frequency_Hz)


def shutdown_output(daq: zi.ziDAQServer, cfg: OutputConfig) -> None:
    """Turn off the MFLI signal output that configure_output() enabled."""
    daq.setInt(f"/{cfg.device}/sigouts/{cfg.out_ch}/on", 0)
    daq.sync()
    log.info("Output %s/sigouts/%d disabled", cfg.device, cfg.out_ch)


def configure_demodulator(daq: zi.ziDAQServer, cfg: DemodConfig) -> None:
    """Configure a single demodulator for a specific harmonic."""
    d   = cfg.device
    di  = cfg.demod_index
    flt = cfg.filter

    # Oscillator / harmonic
    daq.setInt(   f"/{d}/demods/{di}/oscselect",   cfg.osc_index)
    daq.setInt(   f"/{d}/demods/{di}/harmonic",    cfg.harmonic)

    # Filter
    daq.setDouble(f"/{d}/demods/{di}/timeconstant", flt.time_constant_s)
    daq.setInt(   f"/{d}/demods/{di}/order",         flt.order)
    daq.setInt(   f"/{d}/demods/{di}/sinc",          int(flt.sinc_filter))

    # Output rate
    daq.setDouble(f"/{d}/demods/{di}/rate",          cfg.sample_rate_Hz)

    # ADC / input selection
    daq.setInt(   f"/{d}/demods/{di}/adcselect",     cfg.input_ch)
    daq.setInt(   f"/{d}/demods/{di}/enable",        1)

    # Signal Input configuration
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/diff",  int(cfg.differential))
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/ac",    int(cfg.ac_coupling))
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/imp50", 0)   # 10 MΩ, not 50 Ω
    daq.setDouble(f"/{d}/sigins/{cfg.input_ch}/range", cfg.input_range_V)
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/on",    1)

    daq.sync()
    log.info(
        "Demod %s/demod%d  harmonic=%df  TC=%.3f s  order=%d  rate=%.1f Sa/s  "
        "diff=%s  ac=%s  imp=10MΩ",
        d, di, cfg.harmonic, flt.time_constant_s, flt.order, cfg.sample_rate_Hz,
        cfg.differential, cfg.ac_coupling,
    )


def update_filter(daq: zi.ziDAQServer, cfg: DemodConfig, new_filter: FilterConfig) -> None:
    """
    Hot-swap the filter on a running demodulator.
    Call this mid-sweep if you want to change TC or order without full reinit.
    """
    cfg.filter = new_filter
    d, di = cfg.device, cfg.demod_index
    daq.setDouble(f"/{d}/demods/{di}/timeconstant", new_filter.time_constant_s)
    daq.setInt(   f"/{d}/demods/{di}/order",         new_filter.order)
    daq.setInt(   f"/{d}/demods/{di}/sinc",          int(new_filter.sinc_filter))
    daq.sync()
    log.info("Filter updated: %s/demod%d  TC=%.3f s  order=%d",
             d, di, new_filter.time_constant_s, new_filter.order)


def get_demod_phase_deg(daq: zi.ziDAQServer, cfg: DemodConfig) -> float:
    """Read a demodulator's reference phase-shift node (degrees)."""
    return daq.getDouble(f"/{cfg.device}/demods/{cfg.demod_index}/phaseshift")


def set_demod_phase_deg(daq: zi.ziDAQServer, cfg: DemodConfig, phase_deg: float) -> None:
    """
    Set a demodulator's reference phase-shift (degrees) — the ZI-supported
    way to compensate cable/contact/electronics delay without touching the
    excitation itself. This is the same node LabOne's front-panel "Phase"
    field (and its "Auto" button) writes to.

    It's per-demodulator: adjusting it on one demod does not affect any
    other demod, even one sharing the same oscillator (e.g. this doesn't
    touch the follower's 2f phase), and it is independent of Multi-Device
    Synchronization — MDS aligns sample clocks/trigger across devices, not
    per-demod reference phase, so this is safe to call any time after
    configure_demodulator().
    """
    daq.setDouble(f"/{cfg.device}/demods/{cfg.demod_index}/phaseshift", phase_deg)
    daq.sync()


def build_run_metadata(
    daq: zi.ziDAQServer,
    out_cfg: OutputConfig,
    demod1_cfg: DemodConfig,
    demod2_cfg: DemodConfig,
    geometry_cfg: Optional[SampleGeometryConfig] = None,
) -> dict:
    """
    Assemble the run-level metadata a harmonic-Hall analysis needs to turn
    raw 1f/2f voltages into an absolute quantity (e.g. a spin Hall angle)
    and to separate real physics from instrumental phase offsets.

    Excitation current is derived from `out_cfg` assuming the series
    resistor dominates the load impedance (I ≈ V_out / R_series).
    amplitude_V is the MFLI sigouts amplitude node, which LabOne defines
    as the *peak* (0-to-peak) value of the drive sine wave — not RMS, not
    peak-to-peak — so both _peak and _rms current are reported here
    rather than leaving that conversion to whoever reads the CSV later.

    Demodulator X/Y/R follow the opposite (ZI) convention: they report
    the RMS amplitude of the input signal's component at the reference
    frequency, not peak — see demod_output_convention below.

    Reference phases are read live from the device (not copied from
    DemodConfig) because auto_null_phase() can change them after
    configure_demodulator() ran — this should reflect what was actually
    programmed at acquisition time, not what was requested initially.

    `geometry_cfg` (Hall bar dimensions, external-field angle from the
    out-of-plane axis) is never available from an instrument — pass None
    (the default) to leave those columns blank rather than blocking a run.
    """
    geometry_cfg = geometry_cfg or SampleGeometryConfig()
    I_peak_A = out_cfg.amplitude_V / out_cfg.series_R_ohm
    return {
        "excitation_frequency_Hz":       out_cfg.frequency_Hz,
        "excitation_current_A_peak":     I_peak_A,
        "excitation_current_A_rms":      I_peak_A / math.sqrt(2.0),
        "excitation_current_convention": (
            "peak (0-to-peak); I = V_out/R_series assumes the series "
            "resistor dominates the load impedance"
        ),
        "demod_output_convention": (
            "RMS; ZI demodulator X/Y/R nodes report the RMS amplitude of "
            "the input signal's component at the reference frequency"
        ),
        "demod1_time_constant_s":   demod1_cfg.filter.time_constant_s,
        "demod1_filter_order":      demod1_cfg.filter.order,
        "demod1_ref_phase_deg":     get_demod_phase_deg(daq, demod1_cfg),
        "demod2_time_constant_s":   demod2_cfg.filter.time_constant_s,
        "demod2_filter_order":      demod2_cfg.filter.order,
        "demod2_ref_phase_deg":     get_demod_phase_deg(daq, demod2_cfg),
        "hall_bar_length_um":       geometry_cfg.hall_bar_length_um,
        "hall_bar_width_um":        geometry_cfg.hall_bar_width_um,
        "hall_bar_thickness_nm":    geometry_cfg.hall_bar_thickness_nm,
        "field_angle_from_oop_deg": geometry_cfg.field_angle_from_oop_deg,
    }


def auto_null_phase(
    daq: zi.ziDAQServer,
    cfg: DemodConfig,
    n_averages: int = 20,
    max_iterations: int = 5,
    tol_deg: float = 0.02,
    settle_time_s: Optional[float] = None,
) -> PhaseCalibrationResult:
    """
    Null the Y quadrature of `cfg`'s demodulator by adjusting its reference
    phaseshift node — equivalent to LabOne's "Auto" phase button.

    Why this is the right calibration target: at 1st harmonic, a Hall bar's
    transverse voltage is dominated by the planar/anomalous/ordinary Hall
    effect, a purely resistive response (V ∝ R·I(t)) that must be exactly
    in phase with the drive current. Any measured Y at 1f is therefore
    instrumental delay — cabling, contact impedance, source/ADC — not
    physics. Calibrating against the device's own signal (rather than a
    separate standard resistor) captures that real, in-situ delay. Run
    this with the sample actually driven, ideally at a saturated field
    point where the PHE/AHE signal is large and well-behaved.

    Note this only calibrates `cfg`'s own device/demod. In a dual-MFLI
    setup where 1f and 2f are measured on different physical devices, this
    does NOT establish the 2f device's phase — that needs its own check
    (see the module docstring / TUI hint on verifying which of X2f/Y2f
    actually carries the field-dependent signal).

    Iterates because a single large correction can interact with the
    filter's own delay/settling; each round re-measures before deciding
    whether to adjust further. Stops once |residual angle| < tol_deg or
    `max_iterations` is reached. The node's sign convention (whether
    increasing phaseshift increases or decreases measured Y) isn't assumed
    — if a correction makes the residual worse, the sign is flipped.

    `settle_time_s` (default 5x the demod's own filter time_constant_s,
    matching the settling convention used elsewhere in this module) is
    slept after every phaseshift write, before the next measurement.
    This is not optional bookkeeping: writing phaseshift rotates the
    pre-filter mixer output exactly like a step change in the input
    signal, so the demod's low-pass filter needs to settle again just
    like after any other signal change — daq.sync() only confirms the
    register write reached the device, it does not wait for the filter's
    output to converge. Skipping this delay means every iteration after
    the first reads the filter mid-transient rather than its settled
    value; since a sudden phase rotation is not a monotonic transient in
    X/Y, that shows up as the residual bouncing around unpredictably
    instead of shrinking, which starves this function's sign-flip logic
    of a trustworthy "did that help?" signal and it never converges.
    """
    if settle_time_s is None:
        settle_time_s = 5.0 * cfg.filter.time_constant_s

    phase_before = get_demod_phase_deg(daq, cfg)
    phase = phase_before
    sign = 1.0
    prev_abs_residual_deg: Optional[float] = None
    result: Optional[PhaseCalibrationResult] = None

    for i in range(max_iterations):
        d = acquire_averaged(daq, cfg, n_averages)
        if d["r_mean"] <= 0:
            raise RuntimeError(
                f"No signal on {cfg.device}/demod{cfg.demod_index} (R=0) — "
                "can't null a phase against zero amplitude."
            )
        residual_deg = math.degrees(math.atan2(d["y_mean"], d["x_mean"]))
        abs_residual = abs(residual_deg)
        converged = abs_residual < tol_deg
        result = PhaseCalibrationResult(
            phase_before_deg=phase_before, phase_after_deg=phase, iterations=i + 1,
            x_V=d["x_mean"], y_V=d["y_mean"], r_V=d["r_mean"],
            residual_ratio=abs(d["y_mean"]) / d["r_mean"], converged=converged,
        )
        if converged or i == max_iterations - 1:
            break
        if prev_abs_residual_deg is not None and abs_residual > prev_abs_residual_deg:
            # Last correction made the residual worse — this node's sign
            # convention is opposite to what we assumed; flip and continue.
            sign = -sign
        phase = (phase + sign * residual_deg + 180.0) % 360.0 - 180.0
        set_demod_phase_deg(daq, cfg, phase)
        time.sleep(settle_time_s)   # let the demod filter settle to the new phase
        prev_abs_residual_deg = abs_residual

    log.info(
        "Phase null on %s/demod%d: %.4f° → %.4f°  (%d iteration(s), %s, |Y|/R=%.2e)",
        cfg.device, cfg.demod_index, phase_before, result.phase_after_deg,
        result.iterations, "converged" if result.converged else "did not fully converge",
        result.residual_ratio,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Magnet / gaussmeter control
# ─────────────────────────────────────────────────────────────────────────────
# connect_magnet, set_magnet_current, shutdown_magnet, connect_gaussmeter,
# read_field_mT and shutdown_gaussmeter are imported from bridge/instruments/
# (kepco_magnet.py, lakeshore475.py) above unchanged — identical to
# bridge/DC's field-sweep programs.

def bidirectional_current_sweep(i_min: float, i_max: float, n_points: int) -> np.ndarray:
    """
    Build a current sweep that goes i_min → i_max → i_min.

    Sweeping both directions (rather than just up) reveals hysteresis in
    the sample response — useful for a Hall-effect measurement where the
    1f signal (longitudinal/MR) and 2f signal (transverse/Hall) are each
    expected to behave differently under field reversal. The turn-around
    point (i_max) is not duplicated.
    """
    up   = np.linspace(i_min, i_max, n_points)
    down = np.linspace(i_max, i_min, n_points)[1:]
    return np.concatenate([up, down])


# ─────────────────────────────────────────────────────────────────────────────
# Data acquisition
# ─────────────────────────────────────────────────────────────────────────────
# acquire_averaged is imported from bridge/instruments/mfli_daq.py above
# unchanged — it only needs cfg.device/.demod_index/.sample_rate_Hz, which
# DemodConfig (above) already has.

# ─────────────────────────────────────────────────────────────────────────────
# Measurement point  ── extend this for sweeping external parameters ──────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPoint:
    """
    One point in the measurement sequence.

    The magnetic field sweep (magnet_current_A below) is a worked example
    of the general pattern for sweeping any external parameter:

        1.  Add a plain field here:
                gate_V: float = 0.0

        2.  Supply a set_action that applies it:
                set_action = lambda daq: gate.set_voltage(point.gate_V)

        3.  Add the field to the `record` dict inside run_measurement()
            so it is logged to the CSV.

    The set_action is called first at each point, then the script settles
    and acquires — no other changes are needed.

    Note there is no `magnet_field_mT` input field: the field isn't known
    ahead of the sweep, it's measured live by the Lake Shore 475 Gaussmeter
    inside run_measurement() and only appears in the output `record`.
    """
    # ── Magnetic field sweep (Kepco magnet) ────────────────────────────────
    magnet_current_A: Optional[float] = None   # Setpoint applied to the magnet

    # ── Add further sweep variables below ──────────────────────────────────
    # gate_V:     float = 0.0        # Example: gate voltage
    # temperature_K: float = 300.0   # Example: temperature

    # ── Optional override of acquisition settings per point ───────────────
    # Useful if you need a longer settling time at certain field values, etc.
    settling_override_s: Optional[float] = None

    # ── Action performed before settling+acquisition ──────────────────────
    set_action: Optional[Callable[[zi.ziDAQServer], None]] = field(
        default=None, repr=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    daq:        zi.ziDAQServer,
    out_cfg:    OutputConfig,         # Drive/excitation config — source of frequency & current metadata
    demod1_cfg: DemodConfig,          # 1f channel
    demod2_cfg: DemodConfig,          # 2f channel
    acq_cfg:    AcquisitionConfig,
    points:     List[MeasurementPoint],
    stop_event: Optional[threading.Event] = None,
    on_point:   Optional[Callable[[dict], None]] = None,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg:  Optional[GaussmeterConfig] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg:  Optional[TemperatureControllerConfig] = None,
    geometry_cfg: Optional[SampleGeometryConfig] = None,
    mds=None,
) -> pd.DataFrame:
    """
    Iterate over `points`, acquire 1f and 2f at each, log to CSV.

    Returns a DataFrame of all recorded data.
    The CSV is written after every point so a crash never loses data.

    `stop_event`, if given, is checked before each point — set it to break
    out of the sweep early (e.g. from a UI abort button) while still
    returning the data collected so far, so callers can run their normal
    shutdown/cleanup path instead of killing the process outright.

    `mds`, if given (the module handle setup_mds() returns), is re-checked
    at every point via check_mds_status() — MDS can silently drop out of
    sync mid-sweep (a loose Ref/Trigger cable), which would otherwise
    corrupt the 2f data with no indication. A drop is logged as an error
    and recorded per-point ("mds_synced" column) rather than aborting the
    run, since a transient read glitch shouldn't kill an otherwise-good
    sweep — but it means the affected rows are identifiable afterward.

    `on_point`, if given, is called with each point's `record` dict right
    after it's appended — lets a caller (e.g. a live TUI) show progress
    without polling the output CSV.

    `gaussmeter`/`gauss_cfg`, if given, are used to measure the actual
    field at each point (after settling, alongside the 1f/2f acquisition)
    instead of leaving `magnet_field_mT` unset.

    `temp_ctrl`/`temp_cfg`, if given, log the sample/probe temperature
    (temperature_1_K / temperature_2_K) at each point via the shared
    MercuryiTC controller (see mercury_itc.py). Passing `temp_ctrl=None`
    (e.g. because the MercuryiTC isn't connected) simply leaves those
    columns empty — it's never a reason to stop the measurement.

    `out_cfg` and `geometry_cfg` feed build_run_metadata(), which is
    called fresh at every point (not once before the loop) and merged
    into `record`: excitation frequency/current (peak *and* RMS, with the
    convention spelled out), demod filter time-constant/order, and the
    reference phase actually programmed on each demodulator right then —
    read live because a phase re-null or a frequency change via
    set_excitation_frequency() can happen between points.  `geometry_cfg`
    (Hall bar dimensions, field angle from the out-of-plane axis) is
    never available from an instrument; pass None (the default) to leave
    those columns blank rather than blocking the run — see
    SampleGeometryConfig.

    ── Adding more measurements per point ─────────────────────────────────
    Just extend the `record` dict below with any quantity you want to log:
    e.g. a resistance, or an additional demodulator.
    """
    records: List[dict] = []

    for idx, pt in enumerate(points):
        if stop_event is not None and stop_event.is_set():
            log.info("Measurement aborted after %d / %d points.", idx, len(points))
            break

        if pt.magnet_current_A is not None:
            log.info("── Point %d / %d   I_magnet=%.4f A ──────────────────",
                      idx + 1, len(points), pt.magnet_current_A)
        else:
            log.info("── Point %d / %d ──────────────────────────────────", idx + 1, len(points))

        # ── 1. Apply external parameter ────────────────────────────────────
        if pt.set_action is not None:
            pt.set_action(daq)

        # ── 1b. MDS sync re-check ────────────────────────────────────────────
        mds_synced = check_mds_status(mds) if mds is not None else None
        if mds_synced is False:
            log.error("   MDS sync has dropped — 2f data from this point on "
                       "may be corrupted (garbage/beating phasor) until it's "
                       "re-established. Check Ref/Trigger cabling.")

        # ── 2. Settle ──────────────────────────────────────────────────────
        settle = pt.settling_override_s if pt.settling_override_s is not None \
                 else acq_cfg.settling_time_s
        log.info("   Settling %.2f s ...", settle)
        time.sleep(settle)

        # ── 3. Acquire 1f ──────────────────────────────────────────────────
        d1 = acquire_averaged(daq, demod1_cfg, acq_cfg.n_averages)
        log.info("   1f  R=%.4e V  θ=%.2f°  σ_R=%.2e V  (n=%d)",
                 d1["r_mean"], d1["theta_mean"], d1["r_std"], d1["n_samples"])
        if d1["overload"]:
            log.warning("   1f input is OVERLOADED — this reading is not trustworthy.")

        # ── 4. Acquire 2f ──────────────────────────────────────────────────
        d2 = acquire_averaged(daq, demod2_cfg, acq_cfg.n_averages)
        log.info("   2f  R=%.4e V  θ=%.2f°  σ_R=%.2e V  (n=%d)",
                 d2["r_mean"], d2["theta_mean"], d2["r_std"], d2["n_samples"])
        if d2["overload"]:
            log.warning("   2f input is OVERLOADED — this reading is not trustworthy.")

        # ── 4b. Measure field (Lake Shore 475 Gaussmeter) ───────────────────
        field_mT = None
        if gaussmeter is not None and gauss_cfg is not None:
            field_mT = read_field_mT(gaussmeter, gauss_cfg)
            log.info("   B=%.4f mT (measured)", field_mT)

        # ── 4c. Read temperature (MercuryiTC, optional) ─────────────────────
        temp_1_K, temp_2_K = read_temperature(temp_ctrl, temp_cfg) \
            if temp_cfg is not None else (None, None)

        # ── 4d. Run metadata (excitation, filters, phases, geometry) ────────
        # Built fresh each point — see build_run_metadata()'s docstring for
        # why this isn't hoisted above the loop.
        run_meta = build_run_metadata(daq, out_cfg, demod1_cfg, demod2_cfg, geometry_cfg)

        # ── 5. Build record ────────────────────────────────────────────────
        record: dict = {
            "point_index": idx,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mds_synced":  mds_synced,
            # ── Magnet sweep ─────────────────────────────────────────────────
            "magnet_current_A": pt.magnet_current_A,
            "magnet_field_mT":  field_mT,
            # ── Temperature (MercuryiTC) ─────────────────────────────────────
            "temperature_1_K":  temp_1_K,
            "temperature_2_K":  temp_2_K,
            # ── Add further external sweep-parameter columns here, e.g.:
            # "gate_V":      pt.gate_V,
            # ── 1f ─────────────────────────────────────────────────────────
            "1f_X_V":      d1["x_mean"],
            "1f_Y_V":      d1["y_mean"],
            "1f_R_V":      d1["r_mean"],
            "1f_theta_deg":d1["theta_mean"],
            "1f_R_std_V":  d1["r_std"],
            "1f_overload": d1["overload"],
            # ── 2f ─────────────────────────────────────────────────────────
            "2f_X_V":      d2["x_mean"],
            "2f_Y_V":      d2["y_mean"],
            "2f_R_V":      d2["r_mean"],
            "2f_theta_deg":d2["theta_mean"],
            "2f_R_std_V":  d2["r_std"],
            "2f_overload": d2["overload"],
            # ── Run metadata (excitation/demod/geometry — see build_run_metadata) ──
            **run_meta,
            # ── Add further quantities here, e.g. from other instruments ───
        }
        records.append(record)

        if on_point is not None:
            on_point(record)

        # ── 6. Write incrementally (never lose data on a crash) ────────────
        Path(acq_cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(acq_cfg.output_file, index=False)

    log.info("Measurement complete. %d points saved to '%s'.", len(records), acq_cfg.output_file)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── configure your devices and sweep here ──────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Device IDs ──────────────────────────────────────────────────────────
    LEADER   = "dev7885"    # Current source + 1f measurement
    FOLLOWER = "dev7886"    # 2f measurement

    # ── Connect ─────────────────────────────────────────────────────────────
    daq = connect("localhost", 8004)
    connect_device(daq, LEADER,   interface="1GbE")
    connect_device(daq, FOLLOWER, interface="1GbE")

    # ── MDS ─────────────────────────────────────────────────────────────────
    mds = setup_mds(daq, leader=LEADER, follower=FOLLOWER)

    # ── Output (V → I via series resistor) ──────────────────────────────────
    out_cfg = OutputConfig(
        device        = LEADER,
        frequency_Hz  = 317.3,       # Hz — recommended ~300-1000 Hz band, away
                                      #   from 1/f noise and 50 Hz harmonics
        amplitude_V   = 0.1,         # V
        series_R_ohm  = 10000,         # Ω  → I_exc ≈ 100 nA
    )
    configure_output(daq, out_cfg)
    sync_follower_oscillator(daq, out_cfg, FOLLOWER)   # do NOT skip — see module docstring

    # ── Filters ─────────────────────────────────────────────────────────────
    #   Settling rule: settling_time_s ≥ 5×TC (order 1) or ≥ 10×TC (order 4, below)
    shared_filter = FilterConfig(
        time_constant_s = 0.3,       # s
        order           = 4,
        sinc_filter     = True,
    )

    # ── 1f demodulator  (on leader) ─────────────────────────────────────────
    demod1_cfg = DemodConfig(
        device         = LEADER,
        demod_index    = 0,
        harmonic       = 1,
        input_range_V  = 1.0,
        sample_rate_Hz = 857.0,
        filter         = shared_filter,
    )
    configure_demodulator(daq, demod1_cfg)

    # ── 2f demodulator  (on follower) ───────────────────────────────────────
    demod2_cfg = DemodConfig(
        device         = FOLLOWER,
        demod_index    = 0,
        harmonic       = 2,
        input_range_V  = 1.0,
        sample_rate_Hz = 857.0,
        filter         = shared_filter,
    )
    configure_demodulator(daq, demod2_cfg)

    # ── Acquisition settings ─────────────────────────────────────────────────
    acq_cfg = AcquisitionConfig(
        settling_time_s = 15,       # ≥ 10 × TC = 10 × 0.3 = 3 s (magnet settling
                                     #   dominates here, not the filter)
        n_averages      = 50,
        output_file     = str(_DATA_DIR / f"harmonic_hall_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    # ── Magnet (Kepco BOP-GL current source) ─────────────────────────────────
    magnet_cfg = MagnetConfig(
        visa_resource        = "GPIB0::6::INSTR",
        current_limit_A      = 35,    # ← safe continuous limit for your magnet
        voltage_compliance_V = 15.0,
        ramp_step_A          = 0.1,
        ramp_delay_s         = 0.05,
    )
    magnet = connect_magnet(magnet_cfg)

    # ── Gaussmeter (Lake Shore 475, measures the actual field) ────────────────
    gauss_cfg = GaussmeterConfig(
        visa_resource = "GPIB0::12::INSTR",   # ← set to your 475's GPIB address
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

    # ── Sample geometry (optional — fill in whatever you know) ───────────────
    # None of these are readable from any instrument, so they default to
    # None and are simply saved as blank metadata columns if left unset.
    geometry_cfg = SampleGeometryConfig(
        hall_bar_length_um       = None,   # e.g. 20.0
        hall_bar_width_um        = None,   # e.g. 5.0
        hall_bar_thickness_nm    = None,   # e.g. 5.0
        field_angle_from_oop_deg = None,   # e.g. 0.0 for a fully out-of-plane field
    )

    # ── Measurement points ───────────────────────────────────────────────────
    #
    # ① Single acquisition (no sweep):
    #   points = [MeasurementPoint()]
    #
    # ② Sweep the magnet current both directions between two setpoints
    #    (e.g. for a Hall-effect measurement — 1f ≈ longitudinal/MR signal,
    #    2f ≈ transverse/Hall signal, both vs. field, forward and reverse).
    #    The field itself (magnet_field_mT in the output) is measured live
    #    by the gaussmeter at each point, not computed from the current:
    currents_A = bidirectional_current_sweep(i_min=-20.0, i_max=20.0, n_points=21)

    points = [
        MeasurementPoint(
            magnet_current_A = I,
            set_action = lambda daq, I=I: set_magnet_current(magnet, magnet_cfg, I),
        )
        for I in currents_A
    ]
    #
    # ③ Example: change filter per point (e.g. coarser TC at large fields):
    #
    #   points = [
    #       MeasurementPoint(
    #           magnet_current_A = I,
    #           set_action = lambda daq, I=I: (
    #               set_magnet_current(magnet, magnet_cfg, I),
    #               update_filter(daq, demod1_cfg, FilterConfig(time_constant_s=0.1)),
    #               update_filter(daq, demod2_cfg, FilterConfig(time_constant_s=0.1)),
    #           ),
    #           settling_override_s = 0.5,   # shorter TC → shorter settling
    #       )
    #       for I in currents_A
    #   ]

    # ── Run ──────────────────────────────────────────────────────────────────
    # The magnet drives an inductive load, so always ramp it back to zero and
    # disable the output — even if the measurement raises partway through.
    # Likewise, always disable the MFLI signal output so it doesn't keep
    # driving current through the sample after the script exits.
    try:
        df = run_measurement(daq, out_cfg, demod1_cfg, demod2_cfg, acq_cfg, points,
                              gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
                              temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
                              geometry_cfg=geometry_cfg, mds=mds)
        print("\n", df.to_string(index=False))
    finally:
        shutdown_output(daq, out_cfg)
        shutdown_magnet(magnet, magnet_cfg)
        shutdown_gaussmeter(gaussmeter)
        shutdown_temperature_controller(temp_ctrl)


if __name__ == "__main__":
    main()
