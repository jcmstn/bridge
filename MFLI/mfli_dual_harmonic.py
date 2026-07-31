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

Extensibility:
  Add new sweep variables to MeasurementPoint and a corresponding
  set_action callable — the run_measurement loop handles the rest.

Requirements:
    pip install zhinst-core zhinst-utils numpy pandas pyvisa
"""

import sys
import time
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

import pyvisa

# The Kepco magnet driver lives in a sibling folder whose name contains a
# space ("KEPCO magnet"), so it can't be installed as a normal package —
# add it to sys.path directly.
_KEPCO_DIR = Path(__file__).resolve().parent.parent / "KEPCO magnet"
if str(_KEPCO_DIR) not in sys.path:
    sys.path.insert(0, str(_KEPCO_DIR))
from kepco_magnet import KepkoBOPGL  # noqa: E402

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

@dataclass
class OutputConfig:
    """Voltage source → current source configuration."""
    device: str         = "dev1234"   # MFLI acting as leader + current source
    out_ch: int         = 0           # Signal Output index (0-based)
    osc_index: int      = 0           # Oscillator index
    frequency_Hz: float = 17.777      # Excitation frequency  [Hz]
                                      #   (avoid 50/60 Hz harmonics)
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
                                      #   Rule of thumb: ≥ 5 × TC  (filter settles to >99%)
    n_averages: int        = 50       # Number of independent demod samples to average
    output_file: str       = "lockin_data.csv"


@dataclass
class MagnetConfig:
    """
    Kepco BOP-GL bipolar power supply, used as a current source for an
    electromagnet (see kepco_magnet.KepkoBOPGL).

    Field is assumed proportional to current: B [mT] = field_per_amp_mT * I [A].
    Calibrate field_per_amp_mT for your specific magnet/probe before use.

    current_limit_A / voltage_compliance_V are *software* limits enforced by
    this script — separate from the supply's own hardware range (±50 A /
    ±20 V for a BOP 20-50GL) — and should reflect what your magnet can
    safely handle continuously.
    """
    visa_resource:        str   = "GPIB0::6::INSTR"
    field_per_amp_mT:     float = 10.0    # Magnet calibration  [mT / A]
    current_limit_A:      float = 5.0     # Software current limit  [A]
    voltage_compliance_V: float = 15.0    # CC-mode compliance / OVP limit  [V]
    ramp_step_A:          float = 0.1     # Ramp step size  [A]
    ramp_delay_s:         float = 0.05    # Delay between ramp steps  [s]


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
    Module node paths are relative to the module itself (e.g. "devices",
    not "multiDeviceSyncModule/devices").
    """
    mds = daq.multiDeviceSyncModule()

    mds.set("start", 0)
    mds.set("group", 0)
    mds.execute()   # starts the module's worker thread — without this, "start"
                     # is never actually processed and status sits at 0 forever
    mds.set("devices", f"{leader},{follower}")
    mds.set("start", 1)

    # Poll until synchronization is confirmed (status == 2 → synced,
    # -1 → failed, 0/1 → idle/in progress)
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
    daq.setDouble(f"/{d}/sigins/{cfg.input_ch}/range", cfg.input_range_V)
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/on",    1)

    daq.sync()
    log.info(
        "Demod %s/demod%d  harmonic=%df  TC=%.3f s  order=%d  rate=%.1f Sa/s",
        d, di, cfg.harmonic, flt.time_constant_s, flt.order, cfg.sample_rate_Hz,
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


# ─────────────────────────────────────────────────────────────────────────────
# Magnet control (Kepco BOP-GL current source, via VISA/PyVISA)
# ─────────────────────────────────────────────────────────────────────────────

def connect_magnet(cfg: MagnetConfig) -> KepkoBOPGL:
    """
    Open a VISA session to the Kepco BOP-GL and arm it as a current source.

    Puts the supply in constant-current mode with the configured software
    compliance voltage and current limit, sets the setpoint to 0 A, and
    enables the output. Ramping to nonzero setpoints is done separately via
    set_magnet_current() so every field change goes through the same
    step/delay ramp — important for an inductive (magnet) load.
    """
    rm = pyvisa.ResourceManager()
    psu = KepkoBOPGL(rm.open_resource(cfg.visa_resource))

    psu.reset()
    psu.clear_status()
    psu.mode = "current"
    psu.voltage_limit = cfg.voltage_compliance_V
    psu.current_limit = cfg.current_limit_A
    psu.current = 0.0
    psu.enable_output()

    log.info(
        "Magnet connected: %s  mode=CC  compliance=%.2f V  I_limit=±%.2f A",
        cfg.visa_resource, cfg.voltage_compliance_V, cfg.current_limit_A,
    )
    return psu


def set_magnet_current(psu: KepkoBOPGL, cfg: MagnetConfig, current_A: float) -> None:
    """
    Ramp the magnet current to `current_A`, enforcing the software limit
    in `cfg` (independent of the supply's own ±50 A hardware range).
    """
    if abs(current_A) > cfg.current_limit_A:
        raise ValueError(
            f"Requested current {current_A:.3f} A exceeds configured "
            f"limit ±{cfg.current_limit_A:.3f} A"
        )
    psu.ramp_current(current_A, step=cfg.ramp_step_A, delay=cfg.ramp_delay_s)


def shutdown_magnet(psu: KepkoBOPGL, cfg: MagnetConfig) -> None:
    """Ramp the magnet current safely to zero, disable the output, and close the VISA session."""
    log.info("Ramping magnet to zero and disabling output ...")
    psu.zero_output(ramp=True, step=cfg.ramp_step_A, delay=cfg.ramp_delay_s)
    psu.close()


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

def _poll_demod(daq: zi.ziDAQServer, path: str,
                duration_s: float, timeout_ms: int) -> dict:
    """
    Subscribe, flush, poll for `duration_s`, unsubscribe.
    Returns a dict with arrays for x, y, r, theta_deg.
    """
    daq.subscribe(path)
    daq.sync()
    data = daq.poll(duration_s, timeout_ms, flat=True)
    daq.unsubscribe(path)

    if path not in data or len(data[path]) == 0:
        raise RuntimeError(f"No data returned for {path}. "
                           "Check demodulator is enabled and sample rate > 0.")

    # With flat=True, data[path] is a single dict of field -> numpy array
    # (all samples from the poll window concatenated), not a list of
    # per-sample dicts.
    samples = data[path]
    x = np.atleast_1d(samples["x"])
    y = np.atleast_1d(samples["y"])
    r = np.hypot(x, y)
    theta = np.degrees(np.arctan2(y, x))
    return {"x": x, "y": y, "r": r, "theta_deg": theta}


def acquire_averaged(
    daq: zi.ziDAQServer,
    cfg: DemodConfig,
    n_averages: int,
) -> dict:
    """
    Collect at least `n_averages` samples and return their mean ± std.
    Poll duration is chosen to guarantee enough samples at the configured rate.
    """
    path = f"/{cfg.device}/demods/{cfg.demod_index}/sample".lower()
    # Add a 50 % margin so we comfortably exceed n_averages
    duration_s  = max(0.1, (n_averages * 1.5) / cfg.sample_rate_Hz)
    timeout_ms  = int(duration_s * 1000) + 2000

    raw = _poll_demod(daq, path, duration_s, timeout_ms)

    # Trim to last n_averages samples (freshest data after settling)
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
# Measurement point  ── extend this for sweeping external parameters ──────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPoint:
    """
    One point in the measurement sequence.

    The magnetic field sweep (magnet_current_A / magnet_field_mT below) is
    a worked example of the general pattern for sweeping any external
    parameter:

        1.  Add a plain field here:
                gate_V: float = 0.0

        2.  Supply a set_action that applies it:
                set_action = lambda daq: gate.set_voltage(point.gate_V)

        3.  Add the field to the `record` dict inside run_measurement()
            so it is logged to the CSV.

    The set_action is called first at each point, then the script settles
    and acquires — no other changes are needed.
    """
    # ── Magnetic field sweep (Kepco magnet) ────────────────────────────────
    magnet_current_A: Optional[float] = None   # Setpoint applied to the magnet
    magnet_field_mT:  Optional[float] = None   # Computed from field_per_amp_mT

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
    demod1_cfg: DemodConfig,          # 1f channel
    demod2_cfg: DemodConfig,          # 2f channel
    acq_cfg:    AcquisitionConfig,
    points:     List[MeasurementPoint],
    stop_event: Optional[threading.Event] = None,
    on_point:   Optional[Callable[[dict], None]] = None,
) -> pd.DataFrame:
    """
    Iterate over `points`, acquire 1f and 2f at each, log to CSV.

    Returns a DataFrame of all recorded data.
    The CSV is written after every point so a crash never loses data.

    `stop_event`, if given, is checked before each point — set it to break
    out of the sweep early (e.g. from a UI abort button) while still
    returning the data collected so far, so callers can run their normal
    shutdown/cleanup path instead of killing the process outright.

    `on_point`, if given, is called with each point's `record` dict right
    after it's appended — lets a caller (e.g. a live TUI) show progress
    without polling the output CSV.

    ── Adding more measurements per point ─────────────────────────────────
    Just extend the `record` dict below with any quantity you want to log:
    e.g. a temperature readout, a resistance, or an additional demodulator.
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

        # ── 2. Settle ──────────────────────────────────────────────────────
        settle = pt.settling_override_s if pt.settling_override_s is not None \
                 else acq_cfg.settling_time_s
        log.info("   Settling %.2f s ...", settle)
        time.sleep(settle)

        # ── 3. Acquire 1f ──────────────────────────────────────────────────
        d1 = acquire_averaged(daq, demod1_cfg, acq_cfg.n_averages)
        log.info("   1f  R=%.4e V  θ=%.2f°  σ_R=%.2e V  (n=%d)",
                 d1["r_mean"], d1["theta_mean"], d1["r_std"], d1["n_samples"])

        # ── 4. Acquire 2f ──────────────────────────────────────────────────
        d2 = acquire_averaged(daq, demod2_cfg, acq_cfg.n_averages)
        log.info("   2f  R=%.4e V  θ=%.2f°  σ_R=%.2e V  (n=%d)",
                 d2["r_mean"], d2["theta_mean"], d2["r_std"], d2["n_samples"])

        # ── 5. Build record ────────────────────────────────────────────────
        record: dict = {
            "point_index": idx,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            # ── Magnet sweep ─────────────────────────────────────────────────
            "magnet_current_A": pt.magnet_current_A,
            "magnet_field_mT":  pt.magnet_field_mT,
            # ── Add further external sweep-parameter columns here, e.g.:
            # "gate_V":      pt.gate_V,
            # ── 1f ─────────────────────────────────────────────────────────
            "1f_X_V":      d1["x_mean"],
            "1f_Y_V":      d1["y_mean"],
            "1f_R_V":      d1["r_mean"],
            "1f_theta_deg":d1["theta_mean"],
            "1f_R_std_V":  d1["r_std"],
            # ── 2f ─────────────────────────────────────────────────────────
            "2f_X_V":      d2["x_mean"],
            "2f_Y_V":      d2["y_mean"],
            "2f_R_V":      d2["r_mean"],
            "2f_theta_deg":d2["theta_mean"],
            "2f_R_std_V":  d2["r_std"],
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
    setup_mds(daq, leader=LEADER, follower=FOLLOWER)

    # ── Output (V → I via series resistor) ──────────────────────────────────
    out_cfg = OutputConfig(
        device        = LEADER,
        frequency_Hz  = 17.777,      # Hz  — well away from 50 Hz harmonics
        amplitude_V   = 0.1,         # V
        series_R_ohm  = 10000,         # Ω  → I_exc ≈ 100 nA
    )
    configure_output(daq, out_cfg)
    sync_follower_oscillator(daq, out_cfg, FOLLOWER)   # do NOT skip — see module docstring

    # ── Filters ─────────────────────────────────────────────────────────────
    #   Settling rule: settling_time_s  ≥  5 × time_constant_s
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
        settling_time_s = 15,       # ≥ 5 × TC = 5 × 0.3 = 1.5 s (wait for magnet to settle too)
        n_averages      = 50,
        output_file     = str(_DATA_DIR / f"harmonic_hall_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    # ── Magnet (Kepco BOP-GL current source) ─────────────────────────────────
    magnet_cfg = MagnetConfig(
        visa_resource        = "GPIB0::6::INSTR",
        field_per_amp_mT     = 10.0,   # ← calibrate for your magnet/probe
        current_limit_A      = 35,    # ← safe continuous limit for your magnet
        voltage_compliance_V = 15.0,
        ramp_step_A          = 0.1,
        ramp_delay_s         = 0.05,
    )
    magnet = connect_magnet(magnet_cfg)

    # ── Measurement points ───────────────────────────────────────────────────
    #
    # ① Single acquisition (no sweep):
    #   points = [MeasurementPoint()]
    #
    # ② Sweep the magnet current both directions between two setpoints
    #    (e.g. for a Hall-effect measurement — 1f ≈ longitudinal/MR signal,
    #    2f ≈ transverse/Hall signal, both vs. field, forward and reverse):
    currents_A = bidirectional_current_sweep(i_min=-20.0, i_max=20.0, n_points=21)

    points = [
        MeasurementPoint(
            magnet_current_A = I,
            magnet_field_mT  = I * magnet_cfg.field_per_amp_mT,
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
    #           magnet_field_mT  = I * magnet_cfg.field_per_amp_mT,
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
        df = run_measurement(daq, demod1_cfg, demod2_cfg, acq_cfg, points)
        print("\n", df.to_string(index=False))
    finally:
        shutdown_magnet(magnet, magnet_cfg)
        shutdown_output(daq, out_cfg)


if __name__ == "__main__":
    main()
