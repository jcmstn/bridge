#!/usr/bin/env python3
"""
Differential Resistance (dV/dI) vs. DC Bias — Dual MFLI (Leader + Follower)
=============================================================================
Companion script to mfli_dual_harmonic.py / mfli_noise_spectrum.py — same
house style, different measurement. Instead of harmonics or noise, this
script superimposes a small AC excitation on top of a DC bias applied to
the DUT, sweeps that DC bias, and records the complex ratio dV/dI at each
point (i.e. the differential/AC resistance, NOT the total V/I). This is
the standard way to get an "I-V-curve-equivalent" characterization out of
a lock-in setup, and is far more informative than a single-point AC
resistance for anything that isn't perfectly ohmic (contacts, tunnel
junctions, diodes, gated 2D systems, etc.) — nonlinearities that a single
DC multimeter reading or a single lock-in point would hide show up
directly as R_diff(V) curvature here.

Wiring
------
    LEADER
      Signal Output 1 ──[ R_series ]── DUT ── Current Input 1 (leader)
                                                 (transimpedance amp reads I
                                                  directly, in amps. R_series
                                                  here is just a current-
                                                  limiting/protection
                                                  resistor — it is NOT used
                                                  in the I calculation, unlike
                                                  the old voltage-across-
                                                  R_series scheme.)

    FOLLOWER
      Signal Input 1  (differential) ──── across the DUT itself
                                           (2-terminal), or across the
                                           inner voltage-sense leads
                                           (4-terminal / Kelvin) ── gives V_DUT

    The DC bias is NOT a separate wire — it's added on top of the AC
    excitation directly at the leader's Signal Output via the analog
    adder (the "offset" node), so the same two wires (Signal Out 1 →
    R_series → DUT) carry both the AC probe signal and the DC bias.

    MDS cabling (see setup_mds() docstring): Ref Out (leader) → Ref In
    (follower), and Trigger Out 1 (leader) fanned out to Trigger In 1 on
    BOTH units.

    IMPORTANT — oscillator frequency is NOT shared automatically by MDS:
    MDS synchronizes the sample clock and start trigger across devices,
    not the per-device oscillator frequency *value*. Each device's local
    oscillator still free-runs at whatever frequency you set it to. If the
    follower's demodulator frequency doesn't exactly match the leader's
    excitation frequency, its X/Y output will show a slow beat instead of
    a stable phasor, and any V/I or phase calculation against the leader
    channel becomes meaningless. sync_follower_oscillator() below sets
    this explicitly — don't skip it.

Method
------
For each DC bias point:
  1. Set the DC offset on the leader's Signal Output (bias applied to DUT).
  2. Settle (≥ 5 × time constant).
  3. Acquire averaged 1f X/Y on the leader's Current Input → I directly (A)
  4. Acquire averaged 1f X/Y on the follower → phasor across the DUT   → V
  5. Z_diff = V_DUT_phasor / I_phasor   (complex differential impedance)
       R_diff       = Re(Z_diff)   ← the number you want
       X_reactive   = Im(Z_diff)   ← should be ~0 for an ohmic contact;
                                      a nonzero value flags stray
                                      capacitance/inductance/bad contact
       |Z|, phase   = magnitude/phase, for diagnosing contact quality

Practical guidance:
  - Keep ac_amplitude_V small compared to any bias step over which R_diff
    changes appreciably (the usual small-signal lock-in assumption) —
    otherwise each point is a chord average across the nonlinearity
    rather than a true local derivative.
  - series_R_ohm is now just a current-limiting/protection resistor
    between the output and the DUT — it plays no part in the I
    calculation (I is read directly off the Current Input's transimpedance
    amp). Keep it small enough that its voltage drop doesn't eat into the
    compliance headroom needed to reach bias_max_V/bias_min_V across the
    DUT.
  - current_cfg.input_range must be sized in AMPS (not volts) to the
    actual DUT current at each bias point — this is the Current Input's
    own range node, independent of the follower's voltage range.

Requirements:
    pip install zhinst-core zhinst-utils numpy pandas matplotlib
"""

import time
import logging
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

import zhinst.core as zi
import zhinst.utils as ziutils

# Data lives outside the source tree, same convention as mfli_dual_harmonic.py
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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

@dataclass
class OutputConfig:
    """
    Leader's Signal Output: small AC excitation + DC bias, summed on the
    same output via the analog adder ("offset" node), driven through
    series_R_ohm into the DUT.
    """
    device: str          = "dev1234"   # MFLI acting as leader + bias source
    out_ch: int          = 0           # Signal Output index (0-based)
    osc_index: int       = 0           # Oscillator index
    frequency_Hz: float  = 137.0       # AC excitation frequency [Hz]
                                       #   (avoid 50/60 Hz harmonics)
    ac_amplitude_V: float = 5e-3       # AC excitation amplitude [V, peak]
                                       #   keep small — see module docstring
    series_R_ohm: float  = 1e5         # Current-limiting/protection resistor
                                       #   [Ω] — NOT used in the I calculation;
                                       #   I is read directly off the leader's
                                       #   Current Input.
    bias_min_V: float    = -0.5        # DC bias sweep lower bound [V]
    bias_max_V: float    = 0.5         # DC bias sweep upper bound [V]


@dataclass
class FilterConfig:
    """Lock-in filter parameters (shared shape, set per demodulator)."""
    time_constant_s: float = 0.3      # Low-pass time constant  [s]
    order: int             = 4        # Filter order  (1–8)
    sinc_filter: bool      = True     # 4th-order sinc on top (extra harmonic rejection)


@dataclass
class DemodConfig:
    """One demodulator channel (current-sense on leader, or voltage on follower)."""
    device: str                        # Device ID  (leader or follower)
    label: str                         # Human-readable name, for logs/CSV
    demod_index: int    = 0            # Demodulator index on that device (0-based)
    harmonic: int       = 1            # 1f — this is a linear-response measurement
    osc_index: int      = 0            # Oscillator to lock to (on THIS device)
    input_ch: int       = 0            # Signal/Current Input index (0-based)
    use_current_input: bool = False    # True: demodulate the device's dedicated
                                       #   Current Input (currins, transimpedance
                                       #   amp — reads amps directly) instead of
                                       #   a Signal Input (sigins, reads volts).
                                       #   Only input_ch=0 is wired up below;
                                       #   verify the adcselect enum in the
                                       #   LabOne node tree before using another
                                       #   channel.
    differential: bool  = True         # Signal Input only: enable differential
                                       #   (IN+ / IN−) mode. Ignored for currins
                                       #   (no differential mode there).
    ac_coupling: bool   = False        # Signal Input only: DC-couple — we need to
                                       #   see the bias-induced operating point,
                                       #   not just the AC component. Ignored for
                                       #   currins.
    input_range: float = 1.0           # Input range — Volts for a Signal Input,
                                       #   Amps for a Current Input. Units depend
                                       #   on use_current_input.
    sample_rate_Hz: float = 857.0      # Demodulator output rate  [Sa/s]
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class AcquisitionConfig:
    """Timing and averaging parameters."""
    settling_time_s: float = 1.5      # Dead-time after a bias step  [s]
                                       #   Rule of thumb: ≥ 5 × TC
    n_averages: int        = 50       # Independent demod samples to average per point
    output_file: str       = "diff_resistance_vs_bias.csv"


@dataclass
class BiasPoint:
    """One point in the DC bias sweep."""
    bias_V: float
    settling_override_s: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Instrument setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def connect(host: str = "localhost", port: int = 8004, api_level: int = 6) -> zi.ziDAQServer:
    """Open a session to the LabOne data server."""
    daq = zi.ziDAQServer(host, port, api_level)
    log.info("Connected to ZI data server at %s:%d", host, port)
    return daq


def connect_device(daq: zi.ziDAQServer, device: str, interface: str = "1GbE") -> None:
    """Connect a device to the data server (no-op if already connected)."""
    try:
        daq.connectDevice(device, interface)
        log.info("Connected device %s via %s", device, interface)
    except RuntimeError:
        log.info("Device %s already connected", device)


def setup_mds(daq: zi.ziDAQServer, leader: str, follower: str) -> None:
    """
    Configure Multi-Device Synchronization between two MFLIs.

    Per the LabOne MultiDeviceSync module reference, there is no separate
    "leader" node — the role is inferred from *order* in the comma-separated
    `devices` list (first entry = leader) and must match the physical
    cabling. The MFLI requires BOTH of the following (ZSync is not an MFLI
    feature — that's UHFQA/SHF-family hardware):
      - Ref clock: BNC cable from the leader's Ref Out to the follower's
        Ref In.
      - Trigger: the leader's Trigger Out 1 fanned out (e.g. via a 1-to-N
        power divider, equal cable lengths) to Trigger In 1 on *both* the
        follower and the leader itself.

    NOTE: this synchronizes clocks and the measurement start instant — it
    does NOT copy oscillator frequency values between devices. See
    sync_follower_oscillator() below.
    """
    mds = daq.multiDeviceSyncModule()

    mds.set("start", 0)
    mds.set("group", 0)
    mds.execute()   # starts the module's worker thread — without this, "start"
                     # is never actually processed and status sits at 0 forever
    mds.set("devices", f"{leader},{follower}")
    mds.set("start", 1)

    log.info("Waiting for MDS synchronization ...")
    timeout = 60.0
    t0 = time.monotonic()
    while True:
        status = mds.getInt("status")
        if status == 2:
            break
        if status == -1:
            raise RuntimeError(
                f"MDS synchronization failed (status=-1): {mds.getString('message')}. "
                "Check Ref clock cable, trigger fan-out cabling, and device order."
            )
        if time.monotonic() - t0 > timeout:
            raise RuntimeError(
                f"MDS sync timed out (status={status}): {mds.getString('message')}. "
                "Check Ref clock cable and trigger fan-out cabling."
            )
        time.sleep(0.2)
    log.info("MDS synchronized: leader=%s, follower=%s", leader, follower)


def configure_output(daq: zi.ziDAQServer, cfg: OutputConfig) -> None:
    """
    Set up the leader's Signal Output: AC excitation (via the oscillator/
    mixer channel) summed with a DC offset (the bias), through series_R_ohm
    into the DUT. Output range is fixed once, sized to cover the full bias
    sweep plus the AC amplitude, so it doesn't re-range (and click relays)
    on every bias step — only the offset itself changes during the sweep.
    """
    d = cfg.device

    # The sigouts/N/{amplitudes,enables} node index is a hardware "mixer
    # channel", NOT the demodulator index — it depends on device type and
    # installed options. Ask zhinst.utils to resolve it rather than
    # hardcoding it.
    discovery = zi.ziDiscovery()
    props = discovery.get(discovery.find(d))
    mixer_c = ziutils.default_output_mixer_channel(props, cfg.out_ch)

    required_peak_V = max(abs(cfg.bias_min_V), abs(cfg.bias_max_V)) + cfg.ac_amplitude_V
    output_range_V = max(0.01, required_peak_V * 1.1)  # headroom; device snaps to nearest valid range

    daq.setDouble(f"/{d}/oscs/{cfg.osc_index}/freq",                cfg.frequency_Hz)
    daq.setDouble(f"/{d}/sigouts/{cfg.out_ch}/amplitudes/{mixer_c}", cfg.ac_amplitude_V)
    daq.setDouble(f"/{d}/sigouts/{cfg.out_ch}/range",                output_range_V)
    daq.setDouble(f"/{d}/sigouts/{cfg.out_ch}/offset",               0.0)  # start at zero bias
    daq.setInt(   f"/{d}/sigouts/{cfg.out_ch}/on",                   1)
    daq.setInt(   f"/{d}/sigouts/{cfg.out_ch}/enables/{mixer_c}",    1)
    daq.setInt(   f"/{d}/sigouts/{cfg.out_ch}/imp50",                0)   # High-Z output
    daq.sync()
    log.info(
        "Output: %s  f=%.4f Hz  Vac=%.4f V  R_series=%.2e Ω  bias=[%.3f, %.3f] V  "
        "range=%.3g V  (mixer_c=%d)",
        d, cfg.frequency_Hz, cfg.ac_amplitude_V, cfg.series_R_ohm,
        cfg.bias_min_V, cfg.bias_max_V, output_range_V, mixer_c,
    )


def sync_follower_oscillator(daq: zi.ziDAQServer, out_cfg: OutputConfig,
                              follower: str, follower_osc_index: int = 0) -> None:
    """
    Explicitly copy the leader's excitation frequency onto the follower's
    own local oscillator. Required because MDS (see setup_mds docstring)
    does not do this for you — each device's oscillator is independently
    set. Skipping this step is the single most common reason a two-MFLI
    lock-in measurement silently returns garbage (a slowly beating phasor
    instead of a stable one).
    """
    daq.setDouble(f"/{follower}/oscs/{follower_osc_index}/freq", out_cfg.frequency_Hz)
    daq.sync()
    log.info("Follower %s oscillator %d frequency set to %.4f Hz (matches leader)",
             follower, follower_osc_index, out_cfg.frequency_Hz)


def set_bias(daq: zi.ziDAQServer, cfg: OutputConfig, bias_V: float) -> None:
    """Change only the DC offset on the leader's output — the AC amplitude/range stay fixed."""
    if not (cfg.bias_min_V - 1e-9 <= bias_V <= cfg.bias_max_V + 1e-9):
        raise ValueError(
            f"Requested bias {bias_V:.4f} V is outside the configured sweep "
            f"range [{cfg.bias_min_V}, {cfg.bias_max_V}] V — refusing to set it."
        )
    daq.setDouble(f"/{cfg.device}/sigouts/{cfg.out_ch}/offset", bias_V)
    daq.sync()


def ramp_bias_to_zero(daq: zi.ziDAQServer, cfg: OutputConfig,
                       step_V: float = 0.02, delay_s: float = 0.02) -> None:
    """Step the DC bias back to 0 V gradually rather than jumping — gentler on the DUT."""
    current = daq.getDouble(f"/{cfg.device}/sigouts/{cfg.out_ch}/offset")
    log.info("Ramping bias from %.4f V to 0 V ...", current)
    n_steps = max(1, int(abs(current) / step_V))
    for v in np.linspace(current, 0.0, n_steps + 1)[1:]:
        daq.setDouble(f"/{cfg.device}/sigouts/{cfg.out_ch}/offset", float(v))
        daq.sync()
        time.sleep(delay_s)


def shutdown_output(daq: zi.ziDAQServer, cfg: OutputConfig) -> None:
    """Turn off the leader's Signal Output. Call ramp_bias_to_zero() first."""
    daq.setInt(f"/{cfg.device}/sigouts/{cfg.out_ch}/on", 0)
    daq.sync()
    log.info("Output %s/sigouts/%d disabled", cfg.device, cfg.out_ch)


def configure_demodulator(daq: zi.ziDAQServer, cfg: DemodConfig) -> None:
    """Configure a single demodulator (current-sense on leader, or voltage on follower)."""
    d   = cfg.device
    di  = cfg.demod_index
    flt = cfg.filter

    daq.setInt(   f"/{d}/demods/{di}/oscselect",   cfg.osc_index)
    daq.setInt(   f"/{d}/demods/{di}/harmonic",    cfg.harmonic)

    daq.setDouble(f"/{d}/demods/{di}/timeconstant", flt.time_constant_s)
    daq.setInt(   f"/{d}/demods/{di}/order",         flt.order)
    daq.setInt(   f"/{d}/demods/{di}/sinc",          int(flt.sinc_filter))

    daq.setDouble(f"/{d}/demods/{di}/rate",          cfg.sample_rate_Hz)

    if cfg.use_current_input:
        # adcselect enum: 0 = sigin0 (Signal Input 1), 1 = currin0 (Current
        # Input 1). Only channel 0 is verified here — don't extend to other
        # input_ch values without checking the actual enum in the LabOne
        # node tree browser first.
        if cfg.input_ch != 0:
            raise NotImplementedError(
                "use_current_input is only wired up for input_ch=0 "
                "(Current Input 1) — verify the adcselect enum value for "
                "other channels before using them."
            )
        daq.setInt(   f"/{d}/demods/{di}/adcselect", 1)
        daq.setDouble(f"/{d}/currins/{cfg.input_ch}/range", cfg.input_range)
        daq.setInt(   f"/{d}/currins/{cfg.input_ch}/on",    1)
    else:
        daq.setInt(   f"/{d}/demods/{di}/adcselect",     cfg.input_ch)
        daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/diff",  int(cfg.differential))
        daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/ac",    int(cfg.ac_coupling))
        daq.setDouble(f"/{d}/sigins/{cfg.input_ch}/range", cfg.input_range)
        daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/on",    1)

    daq.setInt(   f"/{d}/demods/{di}/enable",        1)

    daq.sync()
    log.info(
        "Demod %-24s  %s/demod%d  input=%s  harmonic=%df  TC=%.3f s  order=%d  rate=%.1f Sa/s",
        cfg.label, d, di, "currin0" if cfg.use_current_input else f"sigin{cfg.input_ch}",
        cfg.harmonic, flt.time_constant_s, flt.order, cfg.sample_rate_Hz,
    )


def bidirectional_bias_sweep(v_min: float, v_max: float, n_points: int) -> np.ndarray:
    """
    Build a bias sweep that goes v_min → v_max → v_min.

    Sweeping both directions reveals hysteresis (e.g. charge traps,
    self-heating, memristive contacts) — the up and down traces should sit
    on top of each other for a clean, purely resistive contact. The
    turn-around point (v_max) is not duplicated.
    """
    up   = np.linspace(v_min, v_max, n_points)
    down = np.linspace(v_max, v_min, n_points)[1:]
    return np.concatenate([up, down])


# ─────────────────────────────────────────────────────────────────────────────
# Data acquisition
# ─────────────────────────────────────────────────────────────────────────────

def _poll_demod(daq: zi.ziDAQServer, path: str,
                duration_s: float, timeout_ms: int) -> dict:
    """Subscribe, flush, poll for `duration_s`, unsubscribe. Returns arrays for x, y, r, theta_deg."""
    daq.subscribe(path)
    daq.sync()
    data = daq.poll(duration_s, timeout_ms, flat=True)
    daq.unsubscribe(path)

    if path not in data or len(data[path]) == 0:
        raise RuntimeError(f"No data returned for {path}. "
                           "Check demodulator is enabled and sample rate > 0.")

    samples = data[path]
    x = np.atleast_1d(samples["x"])
    y = np.atleast_1d(samples["y"])
    r = np.hypot(x, y)
    theta = np.degrees(np.arctan2(y, x))
    return {"x": x, "y": y, "r": r, "theta_deg": theta}


def acquire_averaged(daq: zi.ziDAQServer, cfg: DemodConfig, n_averages: int) -> dict:
    """Collect at least `n_averages` samples and return their mean ± std."""
    path = f"/{cfg.device}/demods/{cfg.demod_index}/sample".lower()
    duration_s  = max(0.1, (n_averages * 1.5) / cfg.sample_rate_Hz)
    timeout_ms  = int(duration_s * 1000) + 2000

    raw = _poll_demod(daq, path, duration_s, timeout_ms)
    for k in raw:
        raw[k] = raw[k][-n_averages:]

    return {
        "x_mean":     float(np.mean(raw["x"])),
        "y_mean":     float(np.mean(raw["y"])),
        "r_mean":     float(np.mean(raw["r"])),
        "theta_mean": float(np.mean(raw["theta_deg"])),
        "r_std":      float(np.std(raw["r"])),
        "n_samples":  len(raw["r"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    daq:          zi.ziDAQServer,
    out_cfg:      OutputConfig,
    current_cfg:  DemodConfig,     # leader — V across R_series
    voltage_cfg:  DemodConfig,     # follower — V across the DUT
    acq_cfg:      AcquisitionConfig,
    points:       List[BiasPoint],
    stop_event:   Optional[threading.Event] = None,
    on_point:     Optional[Callable[[dict], None]] = None,
) -> pd.DataFrame:
    """
    Iterate over `points`, set each DC bias, acquire the current-sense and
    voltage-sense phasors, compute the differential impedance, log to CSV.

    `stop_event`, if given, is checked before each point — set it to break
    out of the sweep early (e.g. from a UI abort button) while still
    returning the data collected so far, so callers can run their normal
    ramp-down/shutdown path instead of killing the process outright.

    `on_point`, if given, is called with each point's `record` dict right
    after it's appended — lets a caller (e.g. a live TUI) show progress
    without polling the output CSV.
    """
    records: List[dict] = []

    for idx, pt in enumerate(points):
        if stop_event is not None and stop_event.is_set():
            log.info("Measurement aborted after %d / %d points.", idx, len(points))
            break

        log.info("── Point %d / %d   V_bias=%.4f V ──────────────────", idx + 1, len(points), pt.bias_V)

        # ── 1. Apply bias ───────────────────────────────────────────────────
        set_bias(daq, out_cfg, pt.bias_V)

        # ── 2. Settle ────────────────────────────────────────────────────────
        settle = pt.settling_override_s if pt.settling_override_s is not None \
                 else acq_cfg.settling_time_s
        log.info("   Settling %.2f s ...", settle)
        time.sleep(settle)

        # ── 3. Acquire current-sense phasor (leader, Current Input → amps) ────
        i_raw = acquire_averaged(daq, current_cfg, acq_cfg.n_averages)
        if current_cfg.use_current_input:
            I_phasor = complex(i_raw["x_mean"], i_raw["y_mean"])
        else:
            I_phasor = complex(i_raw["x_mean"], i_raw["y_mean"]) / out_cfg.series_R_ohm

        # ── 4. Acquire voltage-sense phasor (follower, across the DUT) ─────────
        v_raw = acquire_averaged(daq, voltage_cfg, acq_cfg.n_averages)
        V_phasor = complex(v_raw["x_mean"], v_raw["y_mean"])

        # ── 5. Differential impedance Z = dV/dI ─────────────────────────────
        if abs(I_phasor) == 0:
            log.warning("   Zero current phasor at this point — check wiring/range. Skipping Z calc.")
            Z = complex("nan")
        else:
            Z = V_phasor / I_phasor

        R_diff   = Z.real
        X_react  = Z.imag
        Z_mag    = abs(Z)
        Z_phase  = np.degrees(np.angle(Z))

        log.info("   I_ac=%.4e A  V_ac(DUT)=%.4e V  R_diff=%.5g Ω  X_react=%.3g Ω  phase=%.2f°",
                 abs(I_phasor), abs(V_phasor), R_diff, X_react, Z_phase)

        record = {
            "point_index":    idx,
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bias_V":         pt.bias_V,
            "I_ac_A":         abs(I_phasor),
            "I_ac_X_A":       I_phasor.real,
            "I_ac_Y_A":       I_phasor.imag,
            "V_dut_ac_V":     abs(V_phasor),
            "V_dut_ac_X_V":   V_phasor.real,
            "V_dut_ac_Y_V":   V_phasor.imag,
            "R_diff_ohm":     R_diff,
            "X_reactive_ohm": X_react,
            "Z_mag_ohm":      Z_mag,
            "Z_phase_deg":    Z_phase,
            "I_r_std_A":      i_raw["r_std"] if current_cfg.use_current_input
                              else i_raw["r_std"] / out_cfg.series_R_ohm,
            "V_r_std_V":      v_raw["r_std"],
        }
        records.append(record)
        if on_point is not None:
            on_point(record)

        # ── 6. Write incrementally (never lose data on a crash) ────────────
        Path(acq_cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(acq_cfg.output_file, index=False)

    log.info("Measurement complete. %d points saved to '%s'.", len(records), acq_cfg.output_file)
    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, out_path: Path) -> None:
    """Three-panel summary: R_diff, reactive component, and |Z|/phase, all vs. bias."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(df["bias_V"], df["R_diff_ohm"], ".-", color="#2E3192")
    axes[0].set_ylabel("R_diff (Ω)")
    axes[0].set_title("Differential resistance vs. DC bias")
    axes[0].grid(alpha=0.4)

    axes[1].plot(df["bias_V"], df["X_reactive_ohm"], ".-", color="#e34948")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("Reactive part (Ω)")
    axes[1].set_title("Should sit near 0 for a clean ohmic contact")
    axes[1].grid(alpha=0.4)

    ax2 = axes[2]
    ax2.plot(df["bias_V"], df["Z_phase_deg"], ".-", color="#00AEEF")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_xlabel("DC bias (V)")
    ax2.set_title("Phase of dV/dI — near 0°/180° indicates a resistive contact")
    ax2.grid(alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info("Saved plot: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── configure your devices and sweep here ──────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Device IDs ──────────────────────────────────────────────────────────
    LEADER   = "dev7885"    # Bias + AC excitation source, current-sense demod
    FOLLOWER = "dev7886"    # Voltage-sense demod across the DUT

    # ── Connect ─────────────────────────────────────────────────────────────
    daq = connect("localhost", 8004)
    connect_device(daq, LEADER,   interface="1GbE")
    connect_device(daq, FOLLOWER, interface="1GbE")

    # ── MDS ─────────────────────────────────────────────────────────────────
    setup_mds(daq, leader=LEADER, follower=FOLLOWER)

    # ── Output: AC excitation + DC bias on the leader ───────────────────────
    out_cfg = OutputConfig(
        device         = LEADER,
        frequency_Hz   = 137.0,      # Hz — away from 50/60 Hz harmonics
        ac_amplitude_V = 5e-3,       # V  — small-signal; see module docstring
        series_R_ohm   = 1e5,        # Ω
        bias_min_V     = -0.5,       # V
        bias_max_V     =  0.5,       # V
    )
    configure_output(daq, out_cfg)
    sync_follower_oscillator(daq, out_cfg, FOLLOWER)   # do NOT skip — see docstring

    # ── Filters ─────────────────────────────────────────────────────────────
    #   Settling rule: settling_time_s ≥ 5 × time_constant_s
    shared_filter = FilterConfig(
        time_constant_s = 0.3,
        order           = 4,
        sinc_filter     = True,
    )

    # ── Current-sense demod (on leader, Current Input 1 — reads amps directly) ──
    current_cfg = DemodConfig(
        device            = LEADER,
        label             = "I (Current Input 1)",
        demod_index       = 0,
        harmonic          = 1,
        input_ch          = 0,
        use_current_input = True,
        input_range       = 1e-6,   # Amps — size this to the actual DUT current
        sample_rate_Hz    = 857.0,
        filter            = shared_filter,
    )
    configure_demodulator(daq, current_cfg)

    # ── Voltage-sense demod (on follower, across the DUT) ────────────────────
    voltage_cfg = DemodConfig(
        device         = FOLLOWER,
        label          = "V (across DUT)",
        demod_index    = 0,
        harmonic       = 1,
        input_range    = 0.1,        # Volts — size this to the actual V across the DUT
        sample_rate_Hz = 857.0,
        filter         = shared_filter,
    )
    configure_demodulator(daq, voltage_cfg)

    # ── Acquisition settings ─────────────────────────────────────────────────
    acq_cfg = AcquisitionConfig(
        settling_time_s = 1.5,       # ≥ 5 × TC = 5 × 0.3 = 1.5 s
        n_averages      = 50,
        output_file     = str(_DATA_DIR / f"diff_resistance_vs_bias_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    # ── Bias sweep points ─────────────────────────────────────────────────────
    # Bidirectional (up then down) so hysteresis is visible.
    bias_values = bidirectional_bias_sweep(out_cfg.bias_min_V, out_cfg.bias_max_V, n_points=41)
    points = [BiasPoint(bias_V=float(v)) for v in bias_values]
    #
    # Example: finer steps near zero bias, coarser further out —
    #   points = [BiasPoint(bias_V=v) for v in np.concatenate([
    #       np.linspace(out_cfg.bias_min_V, -0.05, 10),
    #       np.linspace(-0.05, 0.05, 21),
    #       np.linspace(0.05, out_cfg.bias_max_V, 10),
    #   ])]

    # ── Run ──────────────────────────────────────────────────────────────────
    # Always ramp the bias back to zero and disable the output, even if the
    # measurement raises partway through — don't leave a DC bias sitting on
    # the DUT.
    try:
        df = run_measurement(daq, out_cfg, current_cfg, voltage_cfg, acq_cfg, points)
        print("\n", df.to_string(index=False))
        plot_path = Path(acq_cfg.output_file).with_suffix(".png")
        plot_results(df, plot_path)
        plt.show()
    finally:
        ramp_bias_to_zero(daq, out_cfg)
        shutdown_output(daq, out_cfg)


if __name__ == "__main__":
    main()
