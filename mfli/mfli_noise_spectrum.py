#!/usr/bin/env python3
"""
Dual MFLI Voltage-Noise Spectrum Characterization
==================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-30

Records long raw demodulator time series on both MFLIs and turns them into
voltage-noise amplitude spectral densities (ASD, V/√Hz). Use it to find
noise floors, 1/f corners, and mains pickup, and to check whether a
wiring/shielding/grounding change actually helped.

Method (deliberately low-tech):
  These MFLIs don't have the LabOne DAQ Module's spectrum/FFT options
  enabled, so this script does NOT use daq.dataAcquisitionModule() or any
  other paid-option tool. It only uses the demodulator sample *stream*
  (daq.subscribe/poll on "/dev.../demods/N/sample"), which is standard
  base-instrument functionality — the same node mfli_dual_harmonic.py
  already reads. The spectrum itself is computed on the host with
  scipy.signal.welch().

What it measures, per demodulator channel:
  - Excitation ON  : noise with the sample bias applied (real operating
                      condition — includes any excitation-coupled noise).
  - Excitation OFF : baseline instrument + environment noise.
  - (optional)      Any further condition you add to `conditions` in main()
                      — e.g. "inputs shorted" — the script pauses with an
                      on-screen prompt so you can rewire between runs.

For each condition/channel it reports:
  - Voltage noise ASD (X, Y and an averaged X/Y trace) vs. frequency.
  - A white-noise floor estimate and the 1/f corner frequency.
  - RMS noise integrated over the measured band.
  - Mains-pickup peaks (50/100/150 Hz ... by default) flagged automatically.
  - A comparison to the Johnson-Nyquist thermal noise floor of a reference
    resistance (e.g. your series resistor), if you provide one.
  - An approximate input-referred current noise (V_floor / R_ref), if a
    reference resistance is provided.

Results are saved as CSV (one file per condition/channel) plus a summary
plot, and everything of interest is also printed to the console.

IMPORTANT — oscillator frequency is NOT shared automatically by MDS:
MDS synchronizes the sample clock and start trigger across devices, not
the per-device oscillator frequency *value*. Each device's local
oscillator still free-runs at whatever frequency you set it to. If the
follower's demodulator frequency doesn't exactly match the leader's
excitation frequency, its 2f channel will show a slow beat instead of a
stable signal, corrupting the noise spectrum. sync_follower_oscillator()
below sets this explicitly — don't skip it when USE_MDS is True.

Requirements:
    pip install zhinst-core zhinst-utils numpy pandas scipy matplotlib
"""

import time
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import zhinst.core as zi
from scipy import signal
import matplotlib.pyplot as plt

from instruments.mfli_daq import (
    connect,
    connect_device,
    setup_mds,
    check_mds_status,
    sync_follower_oscillator,
)
from instruments.mercury_itc import (
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    shutdown_temperature_controller,
)
from instruments.data_naming import (
    RunContext,
    allocate_run,
    ensure_sample,
    finalize_index_row,
    proc_path,
    write_record,
)

# Data lives outside "bridge" (a sibling of it), same convention as every
# other script in this repo, so nothing generated at runtime ends up in the
# git-tracked source tree.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Locked type code for this measurement (see instruments/data_naming.py) —
# never deviates.
MEASUREMENT_TYPE = "NOISE"

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
    """Voltage source → current source configuration (the excitation to toggle on/off)."""
    device: str         = "dev1234"   # MFLI acting as leader + current source
    out_ch: int         = 0           # Signal Output index (0-based)
    demod_out: int      = 0           # Demodulator that drives the output
    osc_index: int      = 0           # Oscillator index
    frequency_Hz: float = 317.3       # Excitation frequency  [Hz]
                                      #   Track mfli_dual_harmonic.py's real
                                      #   operating point (recommended
                                      #   ~300-1000 Hz band, above 1/f noise).
    amplitude_V: float  = 0.1         # Output amplitude      [V, peak]
    series_R_ohm: float = 1e6         # Series resistor       [Ω]


@dataclass
class NoiseDemodConfig:
    """
    One demodulator channel to characterize.

    Unlike a normal measurement (long time constant, low sample rate — see
    mfli_dual_harmonic.py), a noise-spectrum measurement wants the OPPOSITE:
    a short time constant and a fast sample rate, so the demodulator passes
    as much bandwidth as possible for the host-side FFT/Welch estimate to
    work with. order=1 and sinc off are used deliberately here — higher
    order / sinc rejection narrow the usable bandwidth, which is exactly
    what you don't want while surveying the noise floor.
    """
    device: str                        # Device ID
    label: str                         # Human-readable name, used in logs/plots/filenames
    demod_index: int    = 0            # Demodulator index on that device (0-based)
    harmonic: int        = 1           # Match whatever harmonic this channel uses in real use
    osc_index: int      = 0            # Oscillator to lock to
    input_ch: int       = 0            # Signal Input index (0-based)
    differential: bool  = True         # Enable differential (IN+ / IN−) mode
    ac_coupling: bool   = True         # AC-couple the input
    input_range_V: float = 1.0         # Input range [V]  — try smaller (e.g. 0.01) with the
                                        #   inputs shorted to see how much range affects ADC noise
    sample_rate_Hz: float = 13389.0    # Demod data rate [Sa/s] → Nyquist = rate/2.
                                        #   Raise this (device will clamp to its nearest allowed
                                        #   value) for a wider spectrum, if your device supports it.
    time_constant_s: float = 30e-6     # Short TC → wide noise bandwidth  [s]
    order: int             = 1         # Filter order — kept low on purpose, see class docstring
    sinc_filter: bool      = False


@dataclass
class AcquisitionConfig:
    """Timing and spectral-estimation parameters."""
    duration_s: float        = 60.0    # Raw time series length per condition/channel [s]
                                        #   Sets the lowest usable frequency (~1/duration_s).
    poll_chunk_s: float       = 5.0    # Poll in chunks of this length, purely so progress can
                                        #   be reported while a long recording is running.
    welch_seg_s: float        = 10.0   # Welch segment length [s] → frequency resolution ~1/this
    welch_overlap_frac: float = 0.5    # Fractional overlap between Welch segments
    n_repeats: int             = 1     # Re-record and average this many full `duration_s` runs
                                        #   (reduces variance, mainly at low frequency)
    output_dir: str           = "noise_spectrum_results"


@dataclass
class ReferenceConfig:
    """Reference lines / peak-detection parameters used to interpret the spectrum."""
    mains_freq_Hz: float          = 50.0   # Mains frequency (50 Hz EU / 60 Hz US)
    mains_harmonics: int          = 6      # How many harmonics to check/annotate
    mains_flag_ratio: float       = 2.5    # Flag a bin as pickup if it exceeds this × local median
    thermal_R_ohm: Optional[float] = None  # Reference resistance for Johnson-noise comparison
                                            #   and current-noise conversion (e.g. series_R_ohm)
    thermal_T_K: float            = 293.15 # Temperature for the Johnson-noise line [K]


@dataclass
class Condition:
    """One measurement condition (e.g. excitation on/off, or a manual rewiring step)."""
    label: str
    excitation_on: bool
    prompt: Optional[str] = None   # If set, printed and paused on (input()) before this run —
                                    # use it for steps that need you to physically change wiring.


# ─────────────────────────────────────────────────────────────────────────────
# Instrument setup helpers
# ─────────────────────────────────────────────────────────────────────────────
# connect, connect_device, setup_mds and sync_follower_oscillator are
# imported from instruments/mfli_daq.py above unchanged. configure_output
# (this program's excitation on/off topology) stays local.

def configure_output(daq: zi.ziDAQServer, cfg: OutputConfig) -> None:
    """Set up the voltage output that drives the current through the sample."""
    d = cfg.device
    daq.setDouble(f"/{d}/oscs/{cfg.osc_index}/freq",               cfg.frequency_Hz)
    daq.setDouble(f"/{d}/sigouts/{cfg.out_ch}/amplitudes/{cfg.demod_out}", cfg.amplitude_V)
    daq.setDouble(f"/{d}/sigouts/{cfg.out_ch}/range",              max(0.01, cfg.amplitude_V * 2))
    daq.setInt(   f"/{d}/sigouts/{cfg.out_ch}/enables/{cfg.demod_out}", 1)
    daq.setInt(   f"/{d}/sigouts/{cfg.out_ch}/imp50",              0)   # High-Z output
    daq.setInt(   f"/{d}/sigouts/{cfg.out_ch}/on",                 1)
    daq.sync()
    I_nA = cfg.amplitude_V / cfg.series_R_ohm * 1e9
    log.info(
        "Output configured: %s  f=%.4f Hz  Vpp=%.4f V  R=%.2e Ω  → I≈%.3f nA",
        d, cfg.frequency_Hz, cfg.amplitude_V, cfg.series_R_ohm, I_nA,
    )


def set_output_enabled(daq: zi.ziDAQServer, cfg: OutputConfig, enabled: bool) -> None:
    """Toggle the excitation on/off without touching any other output setting."""
    daq.setInt(f"/{cfg.device}/sigouts/{cfg.out_ch}/on", int(enabled))
    daq.sync()
    log.info("Excitation output %s (%s, f=%.4f Hz, %.4f V)",
              "ENABLED" if enabled else "DISABLED", cfg.device, cfg.frequency_Hz, cfg.amplitude_V)


def configure_noise_demod(daq: zi.ziDAQServer, cfg: NoiseDemodConfig) -> None:
    """Configure a demodulator for wide-bandwidth noise streaming and log the actually-applied values."""
    d, di = cfg.device, cfg.demod_index

    daq.setInt(   f"/{d}/demods/{di}/oscselect",    cfg.osc_index)
    daq.setInt(   f"/{d}/demods/{di}/harmonic",     cfg.harmonic)
    daq.setDouble(f"/{d}/demods/{di}/timeconstant", cfg.time_constant_s)
    daq.setInt(   f"/{d}/demods/{di}/order",        cfg.order)
    daq.setInt(   f"/{d}/demods/{di}/sinc",         int(cfg.sinc_filter))
    daq.setDouble(f"/{d}/demods/{di}/rate",         cfg.sample_rate_Hz)
    daq.setInt(   f"/{d}/demods/{di}/adcselect",    cfg.input_ch)
    daq.setInt(   f"/{d}/demods/{di}/enable",       1)

    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/diff",  int(cfg.differential))
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/ac",    int(cfg.ac_coupling))
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/imp50", 0)   # 10 MΩ, not 50 Ω
    daq.setDouble(f"/{d}/sigins/{cfg.input_ch}/range", cfg.input_range_V)
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/on",    1)
    daq.sync()

    # The API clamps requested rate/TC to the nearest value the device actually
    # supports — read them back so the spectrum's frequency axis is accurate.
    actual_rate = daq.getDouble(f"/{d}/demods/{di}/rate")
    actual_tc   = daq.getDouble(f"/{d}/demods/{di}/timeconstant")
    bw_hz = 1.0 / (2 * np.pi * actual_tc) if actual_tc > 0 else float("inf")
    cfg.sample_rate_Hz = actual_rate

    log.info(
        "%s: demod%d  harmonic=%df  rate=%.1f Sa/s (Nyquist=%.1f Hz)  "
        "TC=%.2e s order=%d (~%.0f Hz -3dB)  range=%.3g V",
        cfg.label, di, cfg.harmonic, actual_rate, actual_rate / 2,
        actual_tc, cfg.order, bw_hz, cfg.input_range_V,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Raw time-series acquisition
# ─────────────────────────────────────────────────────────────────────────────

def acquire_time_series(daq: zi.ziDAQServer, cfg: NoiseDemodConfig,
                         duration_s: float, chunk_s: float, mds=None) -> dict:
    """
    Stream the full raw demodulator sample record (not just averaged points)
    for `duration_s`, polling in `chunk_s` chunks so progress can be logged.

    Also checked once per chunk (cheap, and this recording can run tens of
    seconds to minutes):
      - the Signal Input overload flag — an overloaded input makes the
        whole noise floor meaningless, so this is worth catching before
        trusting a spectrum;
      - `mds` (the module handle setup_mds() returns), if given, via
        check_mds_status() — a mid-recording sync drop would corrupt
        the follower channel's spectrum with no other indication.
    Both are summarized (not spammed per-chunk) in the returned dict.
    """
    d, di = cfg.device, cfg.demod_index
    path = f"/{d}/demods/{di}/sample".lower()
    fs = daq.getDouble(f"/{d}/demods/{di}/rate")

    daq.subscribe(path)
    daq.sync()
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    collected_s = 0.0
    t_start = time.monotonic()
    overload_detected = False
    mds_dropped = False
    try:
        while collected_s < duration_s - 1e-9:
            this_chunk = min(chunk_s, duration_s - collected_s)
            timeout_ms = int(this_chunk * 1000) + 3000
            data = daq.poll(this_chunk, timeout_ms, flat=True)
            if path in data and len(data[path]):
                for s in data[path]:
                    x_parts.append(np.atleast_1d(s["x"]))
                    y_parts.append(np.atleast_1d(s["y"]))
            collected_s += this_chunk
            pct = 100.0 * collected_s / duration_s
            log.info("   %-32s %5.1f%%   (%.1f / %.1f s)",
                      cfg.label + " recording:", pct, collected_s, duration_s)

            try:
                if daq.getInt(f"/{d}/sigins/{cfg.input_ch}/overload"):
                    overload_detected = True
            except Exception:
                pass  # node unavailable — reported as overload_detected=False, not a crash

            if mds is not None and not check_mds_status(mds):
                mds_dropped = True
    finally:
        daq.unsubscribe(path)

    if overload_detected:
        log.warning("   %s: input was OVERLOADED at some point during this "
                     "recording — this spectrum is not trustworthy.", cfg.label)
    if mds_dropped:
        log.error("   %s: MDS sync dropped during this recording — check "
                   "Ref/Trigger cabling before trusting this spectrum.", cfg.label)

    if not x_parts:
        raise RuntimeError(f"No demodulator data received for {path}. "
                           "Check the demodulator is enabled and streaming.")

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    n_expected = int(duration_s * fs)
    log.info("   %s: collected %d samples (~%d expected at %.1f Sa/s), %.1fs wall time",
              cfg.label, len(x), n_expected, fs, time.monotonic() - t_start)
    return {
        "x": x, "y": y, "fs": fs,
        "overload_detected": overload_detected,
        "mds_synced": (not mds_dropped) if mds is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spectral analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_psd(x: np.ndarray, fs: float, seg_s: float, overlap_frac: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density via Welch's method. Returns (freq_Hz, psd_V2_per_Hz)."""
    nperseg = int(min(len(x), max(256, round(fs * seg_s))))
    noverlap = int(nperseg * overlap_frac)
    freq, psd = signal.welch(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        detrend="constant", scaling="density", return_onesided=True,
    )
    return freq, psd


def summarize_asd(freq: np.ndarray, asd: np.ndarray) -> dict:
    """Estimate a white-noise floor (median of the top half-decade of frequency) and 1/f corner."""
    mask_top = freq >= 0.5 * freq[-1]
    white_floor = float(np.median(asd[mask_top])) if mask_top.any() else float(np.median(asd))

    corner = float("nan")
    thresh = white_floor * np.sqrt(2)
    for i in np.where(freq > 0)[0]:
        if asd[i] <= thresh and np.all(asd[i:i + 5] <= thresh * 1.5):
            corner = float(freq[i])
            break

    return {"white_floor_V_rthz": white_floor, "corner_freq_Hz": corner}


def thermal_noise_asd(R_ohm: float, T_K: float) -> float:
    """Johnson-Nyquist voltage noise ASD [V/√Hz] of a resistor R at temperature T."""
    k_B = 1.380649e-23
    return float(np.sqrt(4 * k_B * T_K * R_ohm))


def measure_noise_spectrum(daq: zi.ziDAQServer, cfg: NoiseDemodConfig,
                            acq_cfg: AcquisitionConfig,
                            temp_ctrl: Optional[MercuryITC] = None,
                            temp_cfg: Optional[TemperatureControllerConfig] = None,
                            mds=None) -> dict:
    """
    Record (possibly repeated) time series for one channel/condition and
    return its spectrum + stats.

    `mds`, if given (the module handle setup_mds() returns), is passed down
    to acquire_time_series() so a sync drop during any repeat is caught and
    flagged (`mds_synced` in the returned dict).

    `temp_ctrl`/`temp_cfg`, if given, take one live temperature reading
    (temperature_1_K / temperature_2_K, via the shared MercuryiTC
    controller — see mercury_itc.py) right before the recording starts, and
    include it in the returned dict. A noise recording runs for tens of
    seconds to minutes, far slower than the cryostat drifts, so a single
    reading per condition/channel is enough — this is not a per-point
    sweep. Passing `temp_ctrl=None` (e.g. because the MercuryiTC isn't
    connected) simply leaves those fields empty; it's never a reason to
    stop the measurement.
    """
    temp_1_K, temp_2_K = read_temperature(temp_ctrl, temp_cfg) \
        if temp_cfg is not None else (None, None)

    psd_x_runs, psd_y_runs = [], []
    freq = None
    overload_detected = False
    mds_synced = True if mds is not None else None
    for rep in range(acq_cfg.n_repeats):
        if acq_cfg.n_repeats > 1:
            log.info("   Repeat %d/%d", rep + 1, acq_cfg.n_repeats)
        ts = acquire_time_series(daq, cfg, acq_cfg.duration_s, acq_cfg.poll_chunk_s, mds=mds)
        overload_detected = overload_detected or ts["overload_detected"]
        if mds is not None and not ts["mds_synced"]:
            mds_synced = False
        f, psd_x = compute_psd(ts["x"], ts["fs"], acq_cfg.welch_seg_s, acq_cfg.welch_overlap_frac)
        _, psd_y = compute_psd(ts["y"], ts["fs"], acq_cfg.welch_seg_s, acq_cfg.welch_overlap_frac)
        freq = f
        psd_x_runs.append(psd_x)
        psd_y_runs.append(psd_y)

    psd_x = np.mean(psd_x_runs, axis=0)
    psd_y = np.mean(psd_y_runs, axis=0)
    asd_x = np.sqrt(psd_x)
    asd_y = np.sqrt(psd_y)
    asd_avg = np.sqrt(0.5 * (psd_x + psd_y))

    stats = summarize_asd(freq, asd_avg)
    band = freq > 0
    rms_V = float(np.sqrt(np.trapz(0.5 * (psd_x + psd_y)[band], freq[band])))

    return {
        "freq_Hz": freq,
        "asd_x_V_rthz": asd_x,
        "asd_y_V_rthz": asd_y,
        "asd_avg_V_rthz": asd_avg,
        "nyquist_Hz": float(freq[-1]),
        "rms_V": rms_V,
        "label": cfg.label,
        "temperature_1_K": temp_1_K,
        "temperature_2_K": temp_2_K,
        "overload_detected": overload_detected,
        "mds_synced": mds_synced,
        **stats,
    }


def report_mains_peaks(results: Dict[Tuple[str, str], dict], ref_cfg: ReferenceConfig) -> None:
    """Flag mains-frequency harmonics that stand out above their local baseline."""
    for (cond, label), spec in results.items():
        freq, asd = spec["freq_Hz"], spec["asd_avg_V_rthz"]
        peaks = []
        for n in range(1, ref_cfg.mains_harmonics + 1):
            f0 = n * ref_cfg.mains_freq_Hz
            if f0 > freq[-1]:
                break
            i = int(np.argmin(np.abs(freq - f0)))
            lo, hi = max(0, i - 8), min(len(freq), i + 9)
            neighbor = np.ones(hi - lo, dtype=bool)
            neighbor[i - lo] = False
            baseline = np.median(asd[lo:hi][neighbor]) if neighbor.any() else asd[i]
            if baseline > 0 and asd[i] > ref_cfg.mains_flag_ratio * baseline:
                peaks.append((f0, float(asd[i]), float(asd[i] / baseline)))
        spec["mains_peaks"] = peaks
        if peaks:
            log.warning("Mains pickup detected — %s / %s:", cond, label)
            for f0, val, ratio in peaks:
                log.warning("      %6.1f Hz : %.3e V/√Hz  (%.1fx local baseline)", f0, val, ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Output: CSV + plot
# ─────────────────────────────────────────────────────────────────────────────

def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def save_results(
    results: Dict[Tuple[str, str], dict], *,
    sample: str, device: str, temperature_setpoint_K: Optional[float],
    cooldown: str, series: str, status: str = "completed",
) -> List[RunContext]:
    """
    Write one raw file per (condition, channel) pair via allocate_run() +
    write_record() — single-shot, matching this script's existing
    no-incremental-write behavior (a noise recording is one long acquisition
    per pair, not a point-by-point sweep, so there's nothing to write until
    the whole spectrum for that pair is ready). All pairs share one
    `series` tag so they're recognizable as one session in index.csv.

    `status` should be "completed" only if the whole session ran to
    completion; pass "error" (or similar) when saving a partial result set
    collected before an exception — see main()'s try/except, which calls
    this unconditionally so a failed session never loses already-recorded
    spectra.

    Returns the allocated RunContexts in the order written, so the caller
    can build the combined plot's run-range label from the first/last one.
    """
    contexts: List[RunContext] = []
    for (cond, label), spec in results.items():
        ctx = allocate_run(
            _DATA_DIR, sample, device, MEASUREMENT_TYPE,
            temperature_setpoint_K=temperature_setpoint_K, series=series,
        )
        n = len(spec["freq_Hz"])
        records = pd.DataFrame({
            "frequency_Hz":        spec["freq_Hz"],
            "asd_x_V_per_rtHz":    spec["asd_x_V_rthz"],
            "asd_y_V_per_rtHz":    spec["asd_y_V_rthz"],
            "asd_avg_V_per_rtHz":  spec["asd_avg_V_rthz"],
            # Single reading taken before this condition/channel's recording
            # started (see measure_noise_spectrum) — constant across the
            # spectrum, not re-measured per frequency bin.
            "temperature_1_K":    [spec.get("temperature_1_K")] * n,
            "temperature_2_K":    [spec.get("temperature_2_K")] * n,
            "overload_detected":  [spec.get("overload_detected")] * n,
            "mds_synced":         [spec.get("mds_synced")] * n,
        }).to_dict("records")

        measured = [r["temperature_1_K"] for r in records if r.get("temperature_1_K") is not None]
        T_K = (sum(measured) / len(measured)) if measured else ""
        header_fields = {
            "run": ctx.run_number,
            "timestamp": ctx.timestamp.isoformat(timespec="seconds"),
            "sample": ctx.sample,
            "device": ctx.device,
            "type": MEASUREMENT_TYPE,
            "T_setpoint_K": temperature_setpoint_K,
            "T_K": T_K,
            "cooldown": cooldown,
            "status": status,
            "comment": "",
            "series": series,
            "condition": cond,
            "channel_label": label,
            "white_floor_V_rthz": spec["white_floor_V_rthz"],
            "corner_freq_Hz": spec["corner_freq_Hz"],
            "rms_V": spec["rms_V"],
            "overload_detected": spec["overload_detected"],
            "mds_synced": spec["mds_synced"],
        }
        write_record(ctx.raw_path, records, header_fields)
        finalize_index_row(_DATA_DIR, ctx.sample, ctx.run_number, header_fields)
        contexts.append(ctx)
        log.info("Saved spectrum data: %s", ctx.raw_path)
    return contexts


def plot_results(results: Dict[Tuple[str, str], dict], demod_cfgs: List[NoiseDemodConfig],
                  ref_cfg: ReferenceConfig, out_path: Path) -> Path:
    """One log-log subplot per demodulator channel, one colored trace per condition."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor":   "#fcfcfb",
        "axes.edgecolor":   "#c3c2b7",
        "axes.labelcolor":  "#0b0b0b",
        "text.color":       "#0b0b0b",
        "xtick.color":      "#52514e",
        "ytick.color":      "#52514e",
        "grid.color":       "#e1e0d9",
        "font.size":        10,
    })
    # Fixed categorical assignment: color encodes *condition*, consistent across subplots.
    condition_colors = {
        "Excitation ON":  "#2a78d6",   # blue
        "Excitation OFF": "#eb6834",   # orange
    }
    fallback_colors = ["#1baf7a", "#eda100", "#4a3aa7", "#e34948"]

    n = len(demod_cfgs)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.0), squeeze=False)
    axes = axes[0]

    thermal = None
    if ref_cfg.thermal_R_ohm:
        thermal = thermal_noise_asd(ref_cfg.thermal_R_ohm, ref_cfg.thermal_T_K)

    conditions_seen: List[str] = []
    for ax, cfg in zip(axes, demod_cfgs):
        for (cond, label), spec in results.items():
            if label != cfg.label:
                continue
            if cond not in conditions_seen:
                conditions_seen.append(cond)
            color = condition_colors.get(
                cond, fallback_colors[conditions_seen.index(cond) % len(fallback_colors)]
            )
            ax.loglog(spec["freq_Hz"][1:], spec["asd_avg_V_rthz"][1:],
                       color=color, linewidth=1.5, label=cond)

        if thermal:
            ax.axhline(thermal, color="#898781", linestyle="--", linewidth=1.2)
            ax.text(0.99, thermal, f"Johnson noise ({ref_cfg.thermal_R_ohm:.2g} Ω, "
                                    f"{ref_cfg.thermal_T_K:.0f} K) = {thermal:.2e} V/√Hz",
                    transform=ax.get_yaxis_transform(), va="bottom", ha="right",
                    fontsize=8, color="#898781")

        for k in range(1, ref_cfg.mains_harmonics + 1):
            f0 = k * ref_cfg.mains_freq_Hz
            ax.axvline(f0, color="#e34948", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(ref_cfg.mains_freq_Hz, 0.98, f"{ref_cfg.mains_freq_Hz:.0f} Hz mains",
                transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right",
                fontsize=8, color="#e34948", alpha=0.8)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Voltage noise ASD (V/√Hz)")
        ax.set_title(cfg.label)
        ax.grid(True, which="both", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower left")

    fig.suptitle("Dual-MFLI Voltage Noise Spectral Density", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info("Saved plot: %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Device IDs ──────────────────────────────────────────────────────────
    LEADER   = "dev7885"    # Current source + 1f measurement
    FOLLOWER = "dev7886"    # 2f measurement
    USE_MDS  = True         # Keep True to reproduce the real experiment's clocking condition

    # ── Sample / run identity (see instruments/data_naming.py) ───────────────
    # No TUI/web front end for this script — set these by hand, same place
    # every other hardware parameter for this script already lives.
    SAMPLE = "_test"                  # ← real sample name, or "_test" for a smoke test
    DEVICE = "noise_check"            # ← e.g. HB3, SV2
    COOLDOWN = ""                     # ← optional
    TEMPERATURE_SETPOINT_K: Optional[float] = 293.0
    # create=True only stubs sample.yaml/notes.md/raw//proc//index.csv the
    # first time SAMPLE is used — never overwrites an existing sample.
    ensure_sample(_DATA_DIR, SAMPLE, create=True)

    # ── Connect ─────────────────────────────────────────────────────────────
    daq = connect("localhost", 8004)
    connect_device(daq, LEADER,   interface="1GbE")
    connect_device(daq, FOLLOWER, interface="1GbE")
    mds = setup_mds(daq, leader=LEADER, follower=FOLLOWER) if USE_MDS else None

    # ── Output (the excitation we'll toggle on/off between conditions) ───────
    out_cfg = OutputConfig(
        device        = LEADER,
        frequency_Hz  = 317.3,
        amplitude_V   = 0.1,
        series_R_ohm  = 10000,
    )
    configure_output(daq, out_cfg)
    if USE_MDS:
        sync_follower_oscillator(daq, out_cfg, FOLLOWER)   # do NOT skip — see module docstring

    # ── Demodulator channels to characterize ──────────────────────────────────
    demod_cfgs = [
        NoiseDemodConfig(
            device         = LEADER,
            label          = "MFLI-1 (1f channel)",
            demod_index    = 0,
            harmonic       = 1,
            input_range_V  = 1.0,          # same as the real measurement; try smaller w/ inputs shorted
            sample_rate_Hz = 13389.0,
            time_constant_s= 30e-6,
            order          = 1,
        ),
        NoiseDemodConfig(
            device         = FOLLOWER,
            label          = "MFLI-2 (2f channel)",
            demod_index    = 0,
            harmonic       = 2,
            input_range_V  = 1.0,
            sample_rate_Hz = 13389.0,
            time_constant_s= 30e-6,
            order          = 1,
        ),
    ]
    for cfg in demod_cfgs:
        configure_noise_demod(daq, cfg)

    # ── Temperature (Oxford Instruments MercuryiTC, optional) ────────────────
    # Not every rig has one, and not every MercuryiTC has two probes wired up
    # — connect_temperature_controller() returns None rather than raising if
    # it can't be reached, and the measurement runs fine either way.
    temp_cfg = TemperatureControllerConfig(
        visa_resource = "TCPIP0::192.168.1.5::7020::SOCKET",  # ← set to your iTC's address
        sensor_uids   = ("MB1.T1",),   # ← 1 or 2 board UIDs, e.g. ("MB1.T1", "DB5.T1")
    )
    temp_ctrl = connect_temperature_controller(temp_cfg)

    # ── Acquisition / analysis settings ────────────────────────────────────
    acq_cfg = AcquisitionConfig(
        duration_s        = 60.0,
        poll_chunk_s       = 5.0,
        welch_seg_s        = 10.0,
        welch_overlap_frac = 0.5,
        n_repeats          = 1,
        output_dir         = "noise_spectrum_results",
    )
    ref_cfg = ReferenceConfig(
        mains_freq_Hz    = 50.0,
        mains_harmonics  = 6,
        thermal_R_ohm    = out_cfg.series_R_ohm,   # comparison line + current-noise conversion
        thermal_T_K      = 293.15,
    )

    # ── Conditions to run ───────────────────────────────────────────────────
    # Add more entries here as needed, e.g. a manual rewiring step:
    #   Condition("Inputs shorted (noise floor)", excitation_on=False,
    #             prompt="Disconnect the sample and short/terminate both signal inputs."),
    conditions = [
        Condition("Excitation ON",  excitation_on=True),
        Condition("Excitation OFF", excitation_on=False),
    ]

    results: Dict[Tuple[str, str], dict] = {}
    total_steps = len(conditions) * len(demod_cfgs)
    step = 0
    error: Optional[BaseException] = None

    try:
        for cond in conditions:
            log.info("═" * 78)
            log.info("Condition: %s", cond.label)
            if cond.prompt:
                log.info("ACTION REQUIRED: %s", cond.prompt)
                input(">>> Press Enter when ready to continue ... ")

            set_output_enabled(daq, out_cfg, cond.excitation_on)
            settle_s = 2.0
            log.info("Settling %.1f s ...", settle_s)
            time.sleep(settle_s)

            for cfg in demod_cfgs:
                step += 1
                log.info("[%d/%d] Recording noise spectrum: %s — %s",
                          step, total_steps, cond.label, cfg.label)
                spec = measure_noise_spectrum(daq, cfg, acq_cfg,
                                               temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
                                               mds=mds)
                results[(cond.label, cfg.label)] = spec
                corner_str = f"{spec['corner_freq_Hz']:.2f} Hz" if spec["corner_freq_Hz"] == spec["corner_freq_Hz"] else "n/a"
                log.info("   → white floor %.3e V/√Hz | 1/f corner %s | RMS(%.2f–%.0f Hz) %.3e V",
                          spec["white_floor_V_rthz"], corner_str, spec["freq_Hz"][1],
                          spec["nyquist_Hz"], spec["rms_V"])
    except BaseException as exc:  # noqa: BLE001 — re-raised below, after saving what we have
        log.exception("Noise spectrum measurement failed — saving whatever was collected before re-raising")
        error = exc
    finally:
        # Leave the excitation in its normal ON state regardless of how the loop ended.
        set_output_enabled(daq, out_cfg, True)
        shutdown_temperature_controller(temp_ctrl)

    if not results:
        log.warning("No results collected — nothing to save or plot.")
        if error is not None:
            raise error
        return

    # Save unconditionally — even a partial result set from a failed/aborted
    # session is worth keeping, and this must not be skipped by the
    # exception path above (see the try/except this sits after).
    series = f"{SAMPLE}_{DEVICE}_{MEASUREMENT_TYPE}_{datetime.now():%Y%m%dT%H%M%S}"
    run_contexts = save_results(
        results, sample=SAMPLE, device=DEVICE,
        temperature_setpoint_K=TEMPERATURE_SETPOINT_K, cooldown=COOLDOWN, series=series,
        status="completed" if error is None else "error",
    )
    report_mains_peaks(results, ref_cfg)

    first, last = run_contexts[0], run_contexts[-1]
    run_label = first.run_str if first is last else f"{first.run_str}-{last.run_str}"
    png_path = proc_path(_DATA_DIR, SAMPLE, run_label, DEVICE, MEASUREMENT_TYPE, "combined", combined=True)
    plot_results(results, demod_cfgs, ref_cfg, png_path)

    # ── Console summary ────────────────────────────────────────────────────
    log.info("═" * 78)
    log.info("SUMMARY")
    for (cond, label), spec in results.items():
        corner_str = f"{spec['corner_freq_Hz']:.2f} Hz" if spec["corner_freq_Hz"] == spec["corner_freq_Hz"] else "n/a"
        log.info("  %-16s | %-24s | floor %.3e V/√Hz | corner %-10s | RMS(%.2f-%.0fHz) %.3e V | %d mains peak(s)",
                  cond, label, spec["white_floor_V_rthz"], corner_str,
                  spec["freq_Hz"][1], spec["nyquist_Hz"], spec["rms_V"], len(spec["mains_peaks"]))
        if ref_cfg.thermal_R_ohm:
            i_noise = spec["white_floor_V_rthz"] / ref_cfg.thermal_R_ohm
            log.info("      input-referred current noise (via R=%.2e Ω): %.3e A/√Hz",
                      ref_cfg.thermal_R_ohm, i_noise)
    log.info("Results saved under '%s' (raw CSV per condition/channel in raw/, combined plot in proc/).",
              _DATA_DIR / SAMPLE)

    if error is not None:
        # Already saved what we collected above — re-raise now so the
        # caller/console still sees this run did not complete cleanly.
        raise error

    plt.show()


if __name__ == "__main__":
    main()
