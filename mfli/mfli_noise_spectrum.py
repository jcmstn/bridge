#!/usr/bin/env python3
"""
Dual MFLI Voltage-Noise Floor Estimate — 6221-sourced AC current
==================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-17

A quick nV/√Hz estimate for the mfli_dual_harmonic_6221 program (leader 1f /
follower 2f, Keithley 6221 WAVE excitation, both MFLIs ExtRef-locked to the
6221's phase marker) — NOT a full noise-metrology characterization. Plug the
sample/DUT in exactly as you would for the real measurement, run this, read
the white-noise floor off the plot/console, and use it to size a lock-in
filter's time constant/order.

Wiring — identical to mfli_dual_harmonic_6221.py, nothing extra to cable:
    Keithley 6221 (WAVE, sine, continuous)   HI ──▶ I+ pad ;  LO ──▶ I- pad
    Keithley 6221 TRIGGER LINK phase marker  ──▶ split (BNC T, equal
      lengths) to AUX IN 1 on BOTH MFLIs (leader locks demod-index-1's
      oscillator to it for 1f, follower likewise for 2f — see ExtRefConfig).
    Leader MFLI Signal Input 1 (differential)   ──▶ noise-survey demod
    Follower MFLI Signal Input 1 (differential) ──▶ noise-survey demod

Method (unchanged from the retired plain-MFLI-output version of this
script): stream the raw demodulator sample record (daq.subscribe/poll on
"/dev.../demods/N/sample" — standard base-instrument functionality, no
paid-option DAQ/Sweeper/FFT module) and turn it into a voltage-noise ASD on
the host with scipy.signal.welch().

Two passes, both software-controlled — no manual rewiring:
  Excitation ON  : 6221 armed at the real f_ref/amplitude, both MFLIs
                   ExtRef-locked exactly as the production measurement does
                   — the real operating-point floor.
  Excitation OFF : 6221 output disabled, sample/DUT still connected —
                   baseline with everything else unchanged.

A single reference resistance (the DUT's approximate R, typed in — no
physical resistor swap) draws a Johnson-Nyquist comparison line on the plot,
purely as a sanity anchor; the number you actually act on is the measured
white-noise floor itself.

Results are saved as CSV (one file per condition/channel) plus a summary
plot. See mfli_noise_spectrum_tui.py for a friendlier front end over this
same module — every parameter here matches a field there.

Requirements:
    pip install zhinst-core zhinst-utils numpy pandas scipy matplotlib pyvisa pymeasure
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zhinst.core as zi
from scipy import signal

from instruments.keithley6221 import ACSourceConfig, connect_ac_source, shutdown_ac_source
from instruments.mfli_daq import connect, connect_device, setup_mds, check_mds_status
from mfli.mfli_dual_harmonic_6221 import (
    ExtRefConfig,
    configure_external_reference,
    wait_for_reference_lock,
    disable_sigout,
    _check_ac_safety,
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
# unchanged from the retired plain-MFLI-output version; nothing else needs
# to know the excitation mechanism behind it.
MEASUREMENT_TYPE = "NOISE"

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
class NoiseDemodConfig:
    """
    One demodulator channel to characterize.

    Unlike a normal measurement (long time constant, low sample rate — see
    mfli_dual_harmonic_6221.py), a noise-floor estimate wants the OPPOSITE:
    a short time constant and a fast sample rate, so the demodulator passes
    as much bandwidth as possible for the host-side Welch estimate to work
    with. order=1 and sinc off are used deliberately here — higher order /
    sinc rejection narrow the usable bandwidth, which is exactly what you
    don't want while surveying the noise floor.
    """
    device: str                        # Device ID
    label: str                         # Human-readable name, used in logs/plots/filenames
    demod_index: int    = 0            # Demodulator index on that device (0-based) — the
                                        #   REAL signal demod, distinct from the ExtRef
                                        #   PLL's own dedicated detector demod (see
                                        #   ExtRefConfig.pll_demod_index in mfli_dual_harmonic_6221.py)
    harmonic: int        = 1           # Match whatever harmonic this channel uses in real use
    input_ch: int       = 0            # Signal Input index (0-based)
    differential: bool  = True         # Enable differential (IN+ / IN−) mode
    ac_coupling: bool   = True         # AC-couple the input
    input_range_V: float = 1.0         # Input range [V] — match the real measurement
    sample_rate_Hz: float = 13389.0    # Demod data rate [Sa/s] → Nyquist = rate/2.
                                        #   Raise this (device will clamp to its nearest allowed
                                        #   value) for a wider spectrum, if your device supports it.
    time_constant_s: float = 30e-6     # Short TC → wide noise bandwidth  [s]
    order: int             = 1         # Filter order — kept low on purpose, see class docstring
    sinc_filter: bool      = False


@dataclass
class AcquisitionConfig:
    """Timing and spectral-estimation parameters."""
    duration_s: float        = 30.0    # Raw time series length per condition/channel [s]
                                        #   Sets the lowest usable frequency (~1/duration_s).
                                        #   Kept short by default — this is a quick estimate,
                                        #   not a metrology-grade survey.
    poll_chunk_s: float       = 5.0    # Poll in chunks of this length, purely so progress can
                                        #   be reported while a long recording is running.
    welch_seg_s: float        = 10.0   # Welch segment length [s] → frequency resolution ~1/this
    welch_overlap_frac: float = 0.5    # Fractional overlap between Welch segments


@dataclass
class ReferenceConfig:
    """Reference values used to interpret the spectrum."""
    thermal_R_ohm: Optional[float] = None    # DUT's approximate resistance — Johnson-noise
                                              # comparison line only, no physical resistor swap
    thermal_T_K: float            = 293.0    # Temperature for the Johnson-noise line [K]
    mains_freq_Hz: float          = 50.0     # Mains frequency (50 Hz EU / 60 Hz US)
    mains_harmonics: int          = 6        # How many harmonics to check/annotate
    mains_flag_ratio: float       = 2.5      # Flag a bin as pickup if it exceeds this × local median


# ─────────────────────────────────────────────────────────────────────────────
# Demodulator setup
# ─────────────────────────────────────────────────────────────────────────────

def configure_noise_demod(daq: zi.ziDAQServer, cfg: NoiseDemodConfig) -> None:
    """Configure a demodulator for wide-bandwidth noise streaming and log the actually-applied values."""
    d, di = cfg.device, cfg.demod_index

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
                            acq_cfg: AcquisitionConfig, mds=None) -> dict:
    """Record a time series for one channel/condition and return its spectrum + stats."""
    ts = acquire_time_series(daq, cfg, acq_cfg.duration_s, acq_cfg.poll_chunk_s, mds=mds)
    f, psd_x = compute_psd(ts["x"], ts["fs"], acq_cfg.welch_seg_s, acq_cfg.welch_overlap_frac)
    _, psd_y = compute_psd(ts["y"], ts["fs"], acq_cfg.welch_seg_s, acq_cfg.welch_overlap_frac)
    asd_x = np.sqrt(psd_x)
    asd_y = np.sqrt(psd_y)
    asd_avg = np.sqrt(0.5 * (psd_x + psd_y))

    stats = summarize_asd(f, asd_avg)
    band = f > 0
    rms_V = float(np.sqrt(np.trapz(0.5 * (psd_x + psd_y)[band], f[band])))

    return {
        "freq_Hz": f,
        "asd_x_V_rthz": asd_x,
        "asd_y_V_rthz": asd_y,
        "asd_avg_V_rthz": asd_avg,
        "nyquist_Hz": float(f[-1]),
        "rms_V": rms_V,
        "label": cfg.label,
        "overload_detected": ts["overload_detected"],
        "mds_synced": ts["mds_synced"],
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
# Orchestration  ── the two ON/OFF passes, no manual rewiring
# ─────────────────────────────────────────────────────────────────────────────

def measure_noise_floor(
    daq: zi.ziDAQServer,
    ac_cfg: ACSourceConfig,
    leader_extref_cfg: ExtRefConfig,
    follower_extref_cfg: ExtRefConfig,
    demod_cfgs: List[NoiseDemodConfig],
    acq_cfg: AcquisitionConfig,
    *,
    also_measure_off: bool = True,
    extref_lock_timeout_s: float = 5.0,
    mds=None,
    stop_event=None,
    on_status: Optional[Callable[[str], None]] = None,
    on_result: Optional[Callable[[str, str, dict], None]] = None,
) -> Dict[Tuple[str, str], dict]:
    """
    Run the "Excitation ON" pass (6221 armed at ac_cfg's operating point,
    both MFLIs ExtRef-locked to its phase marker — the real operating-point
    floor) and, unless `also_measure_off` is False, the "Excitation OFF"
    pass (6221 disabled, wiring otherwise untouched — baseline).

    `_check_ac_safety(ac_cfg)` guards against a mistyped exponent before
    arming, same as mfli_dual_harmonic_6221.py's own run path.

    `on_result(condition, label, spec)`, if given, fires right after each
    condition/channel spectrum is computed -- lets a caller (e.g. a TUI)
    drive a progress indicator without duplicating this function's loop.

    The 6221 is always left disabled on return (via `shutdown_ac_source()`
    in `finally`) — this is a diagnostic tool, not a measurement that should
    leave current flowing unattended.
    """
    _check_ac_safety(ac_cfg)
    disable_sigout(daq, leader_extref_cfg.device)
    disable_sigout(daq, follower_extref_cfg.device)

    results: Dict[Tuple[str, str], dict] = {}
    source = None
    try:
        if on_status:
            on_status("Starting 6221 AC current source …")
        source = connect_ac_source(ac_cfg)

        if on_status:
            on_status("Locking MFLI oscillators to the 6221 marker (ExtRef) …")
        configure_external_reference(daq, leader_extref_cfg, ac_cfg.frequency_Hz)
        configure_external_reference(daq, follower_extref_cfg, ac_cfg.frequency_Hz)
        leader_locked = wait_for_reference_lock(daq, leader_extref_cfg,
                                                 extref_lock_timeout_s, stop_event)
        follower_locked = wait_for_reference_lock(daq, follower_extref_cfg,
                                                   extref_lock_timeout_s, stop_event)
        if not leader_locked:
            log.warning("Leader ExtRef PLL did not report locked within %.2g s — "
                        "check the marker cabling before trusting this floor.",
                        extref_lock_timeout_s)
        if not follower_locked:
            log.warning("Follower ExtRef PLL did not report locked within %.2g s — "
                        "check the marker fan-out cabling before trusting this floor.",
                        extref_lock_timeout_s)

        for cfg in demod_cfgs:
            if on_status:
                on_status(f"Recording noise floor: Excitation ON — {cfg.label} …")
            spec = measure_noise_spectrum(daq, cfg, acq_cfg, mds=mds)
            spec["leader_reference_locked"] = leader_locked
            spec["follower_reference_locked"] = follower_locked
            results[("Excitation ON", cfg.label)] = spec
            log.info("   → white floor %.3e V/√Hz | RMS(%.2f–%.0f Hz) %.3e V",
                      spec["white_floor_V_rthz"], spec["freq_Hz"][1],
                      spec["nyquist_Hz"], spec["rms_V"])
            if on_result:
                on_result("Excitation ON", cfg.label, spec)

        if also_measure_off:
            if on_status:
                on_status("Disabling 6221 output for baseline pass …")
            shutdown_ac_source(source)
            source = None
            for cfg in demod_cfgs:
                if on_status:
                    on_status(f"Recording noise floor: Excitation OFF — {cfg.label} …")
                spec = measure_noise_spectrum(daq, cfg, acq_cfg, mds=mds)
                spec["leader_reference_locked"] = None
                spec["follower_reference_locked"] = None
                results[("Excitation OFF", cfg.label)] = spec
                log.info("   → white floor %.3e V/√Hz | RMS(%.2f–%.0f Hz) %.3e V",
                          spec["white_floor_V_rthz"], spec["freq_Hz"][1],
                          spec["nyquist_Hz"], spec["rms_V"])
                if on_result:
                    on_result("Excitation OFF", cfg.label, spec)
    finally:
        if source is not None:
            shutdown_ac_source(source)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output: CSV + plot
# ─────────────────────────────────────────────────────────────────────────────

def _spec_records(spec: dict) -> list[dict]:
    n = len(spec["freq_Hz"])
    return pd.DataFrame({
        "frequency_Hz":        spec["freq_Hz"],
        "asd_x_V_per_rtHz":    spec["asd_x_V_rthz"],
        "asd_y_V_per_rtHz":    spec["asd_y_V_rthz"],
        "asd_avg_V_per_rtHz":  spec["asd_avg_V_rthz"],
        "overload_detected":  [spec.get("overload_detected")] * n,
        "mds_synced":         [spec.get("mds_synced")] * n,
        "leader_reference_locked":   [spec.get("leader_reference_locked")] * n,
        "follower_reference_locked": [spec.get("follower_reference_locked")] * n,
    }).to_dict("records")


def build_header_fields(ctx: RunContext, cond: str, label: str, spec: dict, *,
                         cooldown: str, series: str, status: str, comment: str = "") -> dict:
    """Universal + measurement-specific header/index fields for one
    (condition, channel) pair's run. Used by save_results() for the initial
    write, and reusable afterward (same shape, `status`/`comment` updated)
    for the operator's post-run status/comment prompt -- see
    mfli_noise_spectrum_tui.py's RunScreen._on_status_comment()."""
    return {
        "run": ctx.run_number,
        "timestamp": ctx.timestamp.isoformat(timespec="seconds"),
        "sample": ctx.sample,
        "device": ctx.device,
        "type": MEASUREMENT_TYPE,
        "T_setpoint_K": "",
        "T_K": "",
        "cooldown": cooldown,
        "status": status,
        "comment": comment,
        "series": series,
        "condition": cond,
        "channel_label": label,
        "white_floor_V_rthz": spec["white_floor_V_rthz"],
        "corner_freq_Hz": spec["corner_freq_Hz"],
        "rms_V": spec["rms_V"],
        "overload_detected": spec["overload_detected"],
        "mds_synced": spec["mds_synced"],
        "leader_reference_locked": spec.get("leader_reference_locked"),
        "follower_reference_locked": spec.get("follower_reference_locked"),
    }


def save_results(
    results: Dict[Tuple[str, str], dict], *,
    sample: str, device: str, cooldown: str, series: str, status: str = "completed",
    data_root: Optional[Path] = None,
) -> List[RunContext]:
    """
    Write one raw file per (condition, channel) pair via allocate_run() +
    write_record() — single-shot, matching this suite's existing
    no-incremental-write behavior (a noise recording is one long acquisition
    per pair, not a point-by-point sweep). All pairs share one `series` tag
    so they're recognizable as one session in index.csv.

    `status` should be "completed" only if the whole session ran to
    completion; pass "error" when saving a partial result set collected
    before an exception — see main()'s try/except, which calls this
    unconditionally so a failed session never loses already-recorded spectra.

    `data_root` defaults to this module's own `_DATA_DIR` fallback (the
    plain main() usage); a TUI/web front end must pass its own identity
    bar's "Data root" — see docs/architecture.md's hard rule on this.

    Returns the allocated RunContexts, in the same order as `results` --
    zip them together to recover which context belongs to which
    (condition, channel) pair (see finalize_comment()).
    """
    root = _DATA_DIR if data_root is None else data_root
    contexts: List[RunContext] = []
    for (cond, label), spec in results.items():
        ctx = allocate_run(root, sample, device, MEASUREMENT_TYPE, series=series)
        header_fields = build_header_fields(ctx, cond, label, spec,
                                             cooldown=cooldown, series=series, status=status)
        write_record(ctx.raw_path, _spec_records(spec), header_fields)
        finalize_index_row(root, ctx.sample, ctx.run_number, header_fields)
        contexts.append(ctx)
        log.info("Saved spectrum data: %s", ctx.raw_path)
    return contexts


def finalize_comment(ctx: RunContext, cond: str, label: str, spec: dict, *,
                      cooldown: str, series: str, status: str, comment: str,
                      data_root: Optional[Path] = None) -> None:
    """Re-finalize one already-saved run's index.csv row with the
    operator's real status/comment -- the full row, not just the comment
    field, since finalize_index_row() rewrites the row wholesale."""
    root = _DATA_DIR if data_root is None else data_root
    header_fields = build_header_fields(ctx, cond, label, spec, cooldown=cooldown,
                                         series=series, status=status, comment=comment)
    finalize_index_row(root, ctx.sample, ctx.run_number, header_fields)


def plot_results(results: Dict[Tuple[str, str], dict], demod_cfgs: List[NoiseDemodConfig],
                  ref_cfg: ReferenceConfig, out_path: Path) -> Path:
    """One log-log subplot per demodulator channel, one colored trace per condition.

    matplotlib.pyplot is imported here, not at module top -- a TUI/web
    caller needs to force the Agg backend (matplotlib.use("Agg")) *before*
    pyplot's first import anywhere in the process; importing it eagerly at
    module load time would lock in whatever GUI backend is default first,
    same reason mfli_dual_harmonic_6221.py's own core module never imports
    matplotlib at all."""
    import matplotlib.pyplot as plt
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

    fig.suptitle("Dual-MFLI Noise Floor Estimate (6221-sourced)", fontsize=13)
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
    LEADER   = "dev7885"    # Current source phase marker + 1f measurement
    FOLLOWER = "dev7886"    # 2f measurement
    USE_MDS  = True

    # ── Sample / run identity (see instruments/data_naming.py) ───────────────
    # No TUI/web front end needed to use this module directly — set these by
    # hand. See mfli_noise_spectrum_tui.py for a form-based front end.
    SAMPLE = "_test"                  # ← real sample name, or "_test" for a smoke test
    DEVICE = "noise_check"            # ← e.g. HB3, SV2
    COOLDOWN = ""                     # ← optional
    ensure_sample(_DATA_DIR, SAMPLE, create=True)

    # ── Connect ─────────────────────────────────────────────────────────────
    daq = connect("localhost", 8004)
    connect_device(daq, LEADER,   interface="1GbE")
    connect_device(daq, FOLLOWER, interface="1GbE")
    mds = setup_mds(daq, leader=LEADER, follower=FOLLOWER) if USE_MDS else None

    # ── Excitation (Keithley 6221) — match your real operating point ─────────
    ac_cfg = ACSourceConfig(
        visa_resource = "GPIB0::20::INSTR",
        amplitude_A   = 1e-4,
        frequency_Hz  = 317.3,
        compliance_V  = 2.0,
    )
    leader_extref_cfg = ExtRefConfig(device=LEADER)
    follower_extref_cfg = ExtRefConfig(device=FOLLOWER)

    # ── Demodulator channels to characterize ──────────────────────────────────
    demod_cfgs = [
        NoiseDemodConfig(device=LEADER,   label="MFLI-1 (1f channel)", harmonic=1),
        NoiseDemodConfig(device=FOLLOWER, label="MFLI-2 (2f channel)", harmonic=2),
    ]
    for cfg in demod_cfgs:
        configure_noise_demod(daq, cfg)

    # ── Acquisition / reference settings ──────────────────────────────────────
    acq_cfg = AcquisitionConfig(duration_s=30.0)
    ref_cfg = ReferenceConfig(thermal_R_ohm=10_000, thermal_T_K=293.0)

    error: Optional[BaseException] = None
    results: Dict[Tuple[str, str], dict] = {}
    try:
        results = measure_noise_floor(
            daq, ac_cfg, leader_extref_cfg, follower_extref_cfg, demod_cfgs, acq_cfg,
            mds=mds, on_status=lambda msg: log.info(msg),
        )
    except BaseException as exc:  # noqa: BLE001 — re-raised below, after saving what we have
        log.exception("Noise floor estimate failed — saving whatever was collected before re-raising")
        error = exc

    if not results:
        log.warning("No results collected — nothing to save or plot.")
        if error is not None:
            raise error
        return

    series = f"{SAMPLE}_{DEVICE}_{MEASUREMENT_TYPE}_{datetime.now():%Y%m%dT%H%M%S}"
    run_contexts = save_results(
        results, sample=SAMPLE, device=DEVICE, cooldown=COOLDOWN, series=series,
        status="completed" if error is None else "error",
    )
    report_mains_peaks(results, ref_cfg)

    first, last = run_contexts[0], run_contexts[-1]
    run_label = first.run_str if first is last else f"{first.run_str}-{last.run_str}"
    png_path = proc_path(_DATA_DIR, SAMPLE, run_label, DEVICE, MEASUREMENT_TYPE, "combined", combined=True)
    plot_results(results, demod_cfgs, ref_cfg, png_path)

    log.info("═" * 78)
    log.info("SUMMARY")
    for (cond, label), spec in results.items():
        log.info("  %-16s | %-24s | floor %.3e V/√Hz | RMS(%.2f-%.0fHz) %.3e V | %d mains peak(s)",
                  cond, label, spec["white_floor_V_rthz"],
                  spec["freq_Hz"][1], spec["nyquist_Hz"], spec["rms_V"], len(spec["mains_peaks"]))
        if ref_cfg.thermal_R_ohm:
            i_noise = spec["white_floor_V_rthz"] / ref_cfg.thermal_R_ohm
            log.info("      input-referred current noise (via R=%.2e Ω): %.3e A/√Hz",
                      ref_cfg.thermal_R_ohm, i_noise)
    log.info("Results saved under '%s' (raw CSV per condition/channel in raw/, combined plot in proc/).",
              _DATA_DIR / SAMPLE)

    if error is not None:
        raise error

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
