#!/usr/bin/env python3
"""
SOT pulsed switching, 2nd-harmonic read — 4200A PMU write pulse + delayed
6221 AC / MFLI 2f readout
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-11

Same switching-curve structure as ``sot_pulsed_switching.py`` (one PMU write
pulse per amplitude, delayed read, amplitude swept up and down at a fixed
tilted field) — only the *read* changes: instead of a 6221 DC ±I reversal
read by a 2182, the 6221 sources an AC sine current and a single Zurich
Instruments MFLI locks in on the SECOND HARMONIC of V_xy (the standard
harmonic-Hall SOT technique). The 1st harmonic is recorded alongside it —
it is the resistive AHE/PHE signal, the direct analogue of ``sot_pulsed_
switching.py``'s ``hall_resistance_ohm``, and it is what a phase-alignment
check is done against.

Per amplitude:
  1. 6221 AC wave OFF (``waveform_abort()`` + ``disable_source()``) — never
     pulse into a live current source on the shared pin.
  2. write pulse (PMU, the swept amplitude) — unchanged from
     ``sot_pulsed_switching.py``.
  3. wait ``delay_after_pulse_s``.
  4. 6221 AC wave back ON (re-armed + restarted, a fresh phase-marker edge
     train) → wait for the MFLI's external-reference PLL to report locked
     (bounded by ``lock_timeout_s``, logged-and-continue if it never does —
     see ``wait_for_reference_lock``) → settle → read 1f and 2f.
  5. 6221 AC wave OFF again.

Wiring — only what changes vs. sot_pulsed_switching.py
--------------------------------------------------------
The 4200A PMU / RPM1 / I+ / common-bus wiring, the 6221 HI→I+ / LO→common
connection, and the Kepco + Lake Shore 475 static field are IDENTICAL — see
that module's docstring. Two things are new:

    Keithley 6221 TRIGGER LINK, phase marker on pin ``phasemarker_line``
    (default 1 — NOT the 6221's factory-default Trigger Link pin; confirm
    that default on your own unit before assuming it's free) ──▶
      Zurich Instruments MFLI  AUX IN 1  (BNC)
        One marker edge per excitation cycle. The MFLI's ``extrefs`` module
        locks an internal oscillator to this edge train (see
        ``configure_external_reference`` below) — this is the mechanism the
        Zurich Instruments "external reference" guide describes for locking
        a lock-in to a source that isn't itself a lock-in. Demodulators
        referencing that oscillator with ``harmonic=1`` / ``harmonic=2``
        then read 1f / 2f without the source and the lock-in sharing a
        clock any other way.

    MFLI Signal Input (differential) ──▶ the SAME transverse (Hall) voltage
      arms the 2182 read in the DC version. A bare 4200A SMU still can't do
      this — see sot_pulsed_switching.py's docstring for why.

The 2182 is not used by this program.

Why the 6221 AC amplitude is swapped in and out per pulse, not left running
----------------------------------------------------------------------------
Same shared-bus hazard as the DC version: the 6221 output must be quiet
while the PMU pulses the main channel. ``OUTPUT OFF`` is still the
instrument's whole off-mechanism (see sot_pulsed_switching.py's "Instrument
protection" section) regardless of DC vs. AC/WAVE sourcing mode, but WAVE
mode has its own arm/start/abort state machine layered on top, so the
off/on dance here is ``waveform_abort()`` (stop the wave) +
``disable_source()`` (OUTPUT OFF, belt-and-suspenders) going off, and
``enable_source()`` + ``waveform_arm()`` + ``waveform_start()`` coming back
— see ``instruments/keithley6221.py::connect_ac_source`` for the ordering
this follows (verbatim from the pymeasure driver's own WAVE-mode example:
no explicit OUTPUT ON before arm/start — arming brings the output up
itself). Restarting the wave from scratch every cycle also gives the MFLI's
PLL a clean, unambiguous edge to re-lock to rather than an implicit "was it
still phase-continuous under OUTPUT OFF?" question this code never has to
answer.

Bench-verify before trusting a run
-----------------------------------
``configure_external_reference`` below logs the MFLI's actual
``extrefs/N/*`` node tree (via ``listNodesJSON``) the first time it runs —
only ``extrefs/*/enable`` is confirmed against this rig's zhinst-utils
install; ``adcselect`` (which Aux Input) and the lock-status node name are
this function's best-effort guess at Zurich's own naming convention, not
verified against real firmware. Read that log on the first bench run and
fix the node paths in this file if the device disagrees.

Also confirm on the bench, once: whether your MFLI's ExtRef/PLL module
claims one of its demodulators internally as its own phase detector (true
on some MF/UHF hardware). If so, that demod index is not available for
``demod1_cfg``/``demod2_cfg`` below — the same ``listNodesJSON`` dump should
show a demod-select node under ``extrefs`` if this applies to your unit.

RMS vs. peak
------------
MFLI demodulator X/Y/R are the RMS amplitude of the input's component at
the reference frequency. The 6221's ``waveform_amplitude`` (and this
module's ``ReadConfig.sense_current_A``) is the PEAK current. Both
conventions are recorded explicitly per row (``excitation_current_A_peak``/
``_rms``, ``demod_output_convention``) rather than left for analysis to
assume — see ``mfli/mfli_dual_harmonic.py::build_run_metadata`` for the
convention this is copied from.

Requirements: pymeasure, pyvisa, numpy, pandas, zhinst-core. KXCI enabled on
the 4200A with ``instruments/kult/bridge_sot_pulse.c`` compiled into a KULT
library; a LabOne data server reachable at ``mfli_host``/``mfli_port``.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import zhinst.core as zi
from pymeasure.instruments.keithley import Keithley6221

from instruments.keithley4200a import (
    Keithley4200AConfig,
    PMUPulseConfig,
    configure_pmu_pulse,
    connect_4200a,
    list_user_libraries,
    pulse_once,
    shutdown_4200a,
)
from instruments.keithley6221 import (
    ACSourceConfig,
    connect_ac_source,
    shutdown_ac_source,
)
from instruments.mfli_daq import connect, connect_device, acquire_averaged
from instruments.kepco_magnet import (
    MagnetConfig,
    connect_magnet,
    set_magnet_current,
    shutdown_magnet,
)
from instruments.lakeshore475 import (
    GaussmeterConfig,
    LakeShore475,
    connect_gaussmeter,
    read_field_mT,
    shutdown_gaussmeter,
)
from instruments.mercury_itc import (
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import linear_sweep, safe_shutdown

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Keithley4200AConfig / PMUPulseConfig / ACSourceConfig / MagnetConfig /
# GaussmeterConfig / TemperatureControllerConfig come from instruments/ (see
# imports above). Only the MFLI side (genuinely local to this program, same
# as every mfli/*.py) and the read timing are defined here.

# Same absolute ceilings as sot_pulsed_switching.py — the shared-bus hazard
# doesn't care whether the current is DC or AC peak.
_READ_CURRENT_CEILING_A     = 10e-3
_READ_COMPLIANCE_CEILING_V  = 21.0


@dataclass
class ReadConfig:
    """The 6221 AC + MFLI 1f/2f read, plus the wait before it."""
    sense_current_A: float        = 1e-4    # 6221 AC peak current amplitude for the Hall read [A]
    compliance_V: float           = 2.0
    n_averages: int                = 50      # independent MFLI demod samples averaged per read
    settle_after_enable_s: float  = 1.0      # dwell after the PLL reports locked, before reading [s]
    lock_timeout_s: float         = 5.0      # max wait for the MFLI ExtRef PLL to lock [s]
    delay_after_pulse_s: float    = 1.0      # wait between write-pulse end and the read [s]


def _check_read_safety(read_cfg: ReadConfig) -> None:
    """Same guard as sot_pulsed_switching.py — refuse a read current /
    compliance that has no business on the shared bus. Called before
    connect_ac_source() on every entry path, because connect_ac_source()
    returns with the 6221 already sourcing."""
    if not 0 < read_cfg.sense_current_A <= _READ_CURRENT_CEILING_A:
        raise ValueError(
            f"sense_current_A must be in (0, {_READ_CURRENT_CEILING_A} A]; got "
            f"{read_cfg.sense_current_A} A. The Hall read needs microamps-to-"
            "milliamps — check for a mistyped exponent.")
    if not 0 < read_cfg.compliance_V <= _READ_COMPLIANCE_CEILING_V:
        raise ValueError(
            f"compliance_V must be in (0, {_READ_COMPLIANCE_CEILING_V} V]; got "
            f"{read_cfg.compliance_V} V. On an open contact the 6221 rails to "
            "this across the shared bus.")


@dataclass
class FilterConfig:
    """Lock-in filter parameters, shared shape, set per demodulator — same as
    mfli_dual_harmonic.py's FilterConfig."""
    time_constant_s: float = 0.3
    order: int             = 4
    sinc_filter: bool      = True


@dataclass
class ExtRefConfig:
    """The MFLI external-reference PLL, locked to the 6221's phase marker
    wired into an Aux Input. See the module docstring's "Bench-verify"
    section — the node paths configure_external_reference() writes are a
    best-effort guess, logged and checkable on first connect."""
    device: str        = "dev1234"
    extref_index: int  = 0     # which ExtRef/PLL module (0-based)
    aux_input_ch: int  = 0     # which Aux Input carries the marker (0-based; 0 = Aux In 1)
    osc_index: int     = 0     # oscillator the PLL locks — demods reference this


@dataclass
class DemodConfig:
    """One demodulator on the externally-locked oscillator (1f or 2f) — same
    shape as mfli_dual_harmonic.py's DemodConfig."""
    device: str
    demod_index: int
    harmonic: int
    osc_index: int       = 0
    input_ch: int        = 0
    differential: bool   = True
    ac_coupling: bool    = True
    input_range_V: float = 1.0
    sample_rate_Hz: float = 857.0
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class AmplitudePoint:
    amplitude_V: float


# ─────────────────────────────────────────────────────────────────────────────
# MFLI setup helpers
# ─────────────────────────────────────────────────────────────────────────────
# connect / connect_device / acquire_averaged are imported from
# instruments/mfli_daq.py unchanged — no MDS, no follower: a single MFLI.

def configure_external_reference(daq: "zi.ziDAQServer", cfg: ExtRefConfig,
                                  frequency_Hz: float) -> None:
    """Lock ``cfg.osc_index`` to the 6221's phase marker wired into
    ``cfg.aux_input_ch``. Pre-sets the oscillator to the known 6221
    frequency first — whether the PLL does a full frequency search or a
    phase-only lock, starting close to correct shortens or removes that
    search (same reasoning as sync_follower_oscillator() in
    instruments/mfli_daq.py). Logs the live ``extrefs`` node tree so the
    node names below can be checked against real firmware on first
    connect — see the module docstring's "Bench-verify" section.
    """
    d = cfg.device
    daq.setDouble(f"/{d}/oscs/{cfg.osc_index}/freq", frequency_Hz)
    try:
        nodes = daq.listNodesJSON(f"/{d}/extrefs/{cfg.extref_index}/*")
        log.info("MFLI %s extrefs/%d node tree (verify against this on first "
                 "bench run):\n%s", d, cfg.extref_index, nodes)
    except Exception:
        log.exception("Could not list /%s/extrefs/%d/* — node names below are "
                      "unverified for this device/firmware.", d, cfg.extref_index)
    daq.setInt(f"/{d}/extrefs/{cfg.extref_index}/adcselect", cfg.aux_input_ch)
    daq.setInt(f"/{d}/extrefs/{cfg.extref_index}/enable", 1)
    daq.sync()
    log.info("MFLI %s: oscillator %d locking to Aux In %d via extrefs/%d "
             "(target %.4f Hz)", d, cfg.osc_index, cfg.aux_input_ch + 1,
             cfg.extref_index, frequency_Hz)


def wait_for_reference_lock(daq: "zi.ziDAQServer", cfg: ExtRefConfig,
                             timeout_s: float,
                             stop_event: Optional[threading.Event] = None) -> bool:
    """Poll the ExtRef PLL's lock flag for up to timeout_s. Never raises —
    an unreadable or unknown node degrades to "not locked" (logged once)
    rather than aborting a run that may otherwise be fine; the caller
    records the result per row (see run_measurement's "reference_locked"
    column) instead of trusting it blindly."""
    path = f"/{cfg.device}/extrefs/{cfg.extref_index}/locked"
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if stop_event is not None and stop_event.is_set():
            return False
        try:
            if daq.getInt(path):
                return True
        except Exception:
            log.warning("Could not read ExtRef lock node %s — check the node "
                       "name against configure_external_reference()'s "
                       "listNodesJSON log.", path)
            return False
        time.sleep(0.05)
    return False


def configure_demodulator(daq: "zi.ziDAQServer", cfg: DemodConfig) -> None:
    """Configure one demodulator — identical node set to
    mfli_dual_harmonic.py::configure_demodulator (kept local per that
    module's own convention: each program's demod topology stays with it)."""
    d, di, flt = cfg.device, cfg.demod_index, cfg.filter

    daq.setInt(   f"/{d}/demods/{di}/oscselect",   cfg.osc_index)
    daq.setInt(   f"/{d}/demods/{di}/harmonic",    cfg.harmonic)
    daq.setDouble(f"/{d}/demods/{di}/timeconstant", flt.time_constant_s)
    daq.setInt(   f"/{d}/demods/{di}/order",         flt.order)
    daq.setInt(   f"/{d}/demods/{di}/sinc",          int(flt.sinc_filter))
    daq.setDouble(f"/{d}/demods/{di}/rate",          cfg.sample_rate_Hz)
    daq.setInt(   f"/{d}/demods/{di}/adcselect",     cfg.input_ch)
    daq.setInt(   f"/{d}/demods/{di}/enable",        1)

    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/diff",  int(cfg.differential))
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/ac",    int(cfg.ac_coupling))
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/imp50", 0)
    daq.setDouble(f"/{d}/sigins/{cfg.input_ch}/range", cfg.input_range_V)
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/on",    1)
    daq.sync()
    log.info("Demod %s/demod%d  harmonic=%df  TC=%.3f s  order=%d  rate=%.1f Sa/s",
             d, di, cfg.harmonic, flt.time_constant_s, flt.order, cfg.sample_rate_Hz)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with sot_pulsed_switching.py (kept local — see that
# module's docstring for why each exists)
# ─────────────────────────────────────────────────────────────────────────────

def _interruptible_sleep(seconds: float, stop_event: Optional[threading.Event]) -> None:
    if seconds <= 0:
        return
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if stop_event is not None and stop_event.is_set():
            return
        time.sleep(min(0.2, end - time.monotonic()))


def _pulse_resistance(pinfo: dict) -> Optional[float]:
    v = pinfo.get("pulse_voltage_measured_V")
    i = pinfo.get("pulse_current_measured_A")
    if v is None or i is None or i == 0:
        return None
    return v / i


_MAX_CONSECUTIVE_PULSE_FAILURES = 3


def _pulse_failure_reason(module_return) -> Optional[str]:
    if module_return is None:
        return "no reply from EX"
    s = str(module_return).strip()
    if "ERROR" in s.upper():
        return s
    try:
        code = float(s)
    except ValueError:
        return None
    return None if code == 0 else f"module returned {s}"


def _six221_ac_output_off(source: Keithley6221) -> None:
    """Stop the AC wave and disable the 6221 output — the state it must be
    in whenever the PMU pulses the shared channel pin. waveform_abort()
    stops the wave function; disable_source() (OUTPUT OFF) is the same
    universal off-mechanism the DC read uses (see sot_pulsed_switching.py's
    "Instrument protection" section) — belt-and-suspenders, since ABORt's
    own effect on the output stage isn't spelled out the same explicit way
    OUTPUT OFF is."""
    source.waveform_abort()
    source.disable_source()


def _six221_ac_output_on(source: Keithley6221) -> None:
    """Resume the AC wave + phase marker after a pulse. Re-arms and
    restarts (rather than relying on OUTPUT ON alone to resume a paused
    wave) — this also hands the MFLI's PLL a clean, unambiguous edge to
    re-lock to every time."""
    source.enable_source()
    source.waveform_arm()
    source.waveform_start()


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop  (docs/architecture.md §3 contract)
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    k4200,
    pmu_cfg: PMUPulseConfig,
    source: Keithley6221,
    daq: "zi.ziDAQServer",
    demod1_cfg: DemodConfig,     # 1f — the resistive AHE/PHE anchor
    demod2_cfg: DemodConfig,     # 2f — the harmonic-Hall switching signal
    extref_cfg: ExtRefConfig,
    read_cfg: ReadConfig,
    points: List[AmplitudePoint],
    stop_event: Optional[threading.Event] = None,
    on_point: Optional[Callable[[dict], None]] = None,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg: Optional[GaussmeterConfig] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg: Optional[TemperatureControllerConfig] = None,
    magnet_current_A: Optional[float] = None,
    field_angle_from_oop_deg: Optional[float] = None,
    write_csv: Optional[Callable[[List[dict]], None]] = None,
    output_file: str = "sot_pulsed_switching_2h.csv",
) -> pd.DataFrame:
    """One pulse + delayed 1f/2f read per amplitude in ``points``:
    (6221 AC off) → write pulse → wait → (6221 AC on, wait for PLL lock,
    read 1f + 2f) → (6221 AC off). One row per amplitude; CSV rewritten in
    full every row.

    Same static-field / stop_event / temp_ctrl=None-never-stops semantics
    as sot_pulsed_switching.py.run_measurement — see that docstring.

    ``reference_locked`` is recorded per row (analogous to
    mfli_dual_harmonic.py's ``mds_synced``): a lock-wait timeout is logged
    and the row is tagged, not treated as fatal — a run with a few
    unlocked rows is still diagnosable, a run that aborted at the first
    timeout is not.

    Raises ``RuntimeError`` after ``_MAX_CONSECUTIVE_PULSE_FAILURES`` write
    pulses in a row fail — identical guard to sot_pulsed_switching.py.
    """
    _check_read_safety(read_cfg)

    field_measured_mT = None
    if gaussmeter is not None and gauss_cfg is not None:
        field_measured_mT = read_field_mT(gaussmeter, gauss_cfg)
        log.info("Static field: %.4f mT measured (magnet current %s A)",
                 field_measured_mT, magnet_current_A)

    records: List[dict] = []
    consecutive_pulse_failures = 0

    for a_idx, pt in enumerate(points):
        if stop_event is not None and stop_event.is_set():
            log.info("Aborted after %d / %d amplitudes.", len(records), len(points))
            _six221_ac_output_off(source)
            return pd.DataFrame(records)

        # ── 1. 6221 AC OFF — never pulse into a live current source ─────
        _six221_ac_output_off(source)

        # ── 2. write pulse ────────────────────────────────────────────
        pinfo = pulse_once(k4200, pmu_cfg, amplitude_V=pt.amplitude_V,
                           stop_event=stop_event)
        aborting = stop_event is not None and stop_event.is_set()
        fail = None if aborting else _pulse_failure_reason(pinfo.get("module_return"))
        if fail is not None:
            consecutive_pulse_failures += 1
            log.warning("Write pulse did not fire (amp %.4g V, %d in a row): %s",
                        pt.amplitude_V, consecutive_pulse_failures, fail)
            if consecutive_pulse_failures >= _MAX_CONSECUTIVE_PULSE_FAILURES:
                _six221_ac_output_off(source)
                raise RuntimeError(
                    f"{consecutive_pulse_failures} consecutive pulse failures "
                    f"— last: {fail}. Aborting; check the KXCI log and the pulse "
                    "timing / PMU config.")
        else:
            consecutive_pulse_failures = 0

        # ── 3. wait ─────────────────────────────────────────────────
        _interruptible_sleep(read_cfg.delay_after_pulse_s, stop_event)

        # ── 4. 6221 AC ON, wait for PLL lock, settle, read 1f + 2f ───
        _six221_ac_output_on(source)
        locked = wait_for_reference_lock(daq, extref_cfg, read_cfg.lock_timeout_s,
                                         stop_event)
        if not locked:
            log.warning("MFLI reference PLL did not report locked within %.2g s "
                       "— reading anyway; this row is tagged reference_locked=False.",
                       read_cfg.lock_timeout_s)
        _interruptible_sleep(read_cfg.settle_after_enable_s, stop_event)

        d1 = acquire_averaged(daq, demod1_cfg, read_cfg.n_averages)
        d2 = acquire_averaged(daq, demod2_cfg, read_cfg.n_averages)
        if d1["overload"] or d2["overload"]:
            log.warning("Input overload at amp %.4g V (1f=%s, 2f=%s) — this "
                       "reading is not trustworthy.",
                       pt.amplitude_V, d1["overload"], d2["overload"])

        # ── 5. 6221 AC OFF again ──────────────────────────────────────
        _six221_ac_output_off(source)

        t1_K, t2_K = read_temperature(temp_ctrl, temp_cfg) if temp_cfg is not None else (None, None)

        I_peak_A = read_cfg.sense_current_A
        record = {
            "amplitude_index":  a_idx,
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pulse_amplitude_V": pt.amplitude_V,
            "pulse_width_s":     pmu_cfg.width_s,
            "n_pulses":          pmu_cfg.n_pulses,
            "pulse_voltage_measured_V": pinfo.get("pulse_voltage_measured_V"),
            "pulse_current_measured_A": pinfo.get("pulse_current_measured_A"),
            "pulse_2wire_resistance_ohm": _pulse_resistance(pinfo),
            "pulse_base_voltage_V": pinfo.get("pulse_base_voltage_V"),
            "pulse_base_current_A": pinfo.get("pulse_base_current_A"),
            "reference_locked": locked,
            "excitation_frequency_Hz": daq.getDouble(f"/{extref_cfg.device}/oscs/{extref_cfg.osc_index}/freq"),
            "excitation_current_A_peak": I_peak_A,
            "excitation_current_A_rms":  I_peak_A / 2 ** 0.5,
            "excitation_current_convention": "peak; 6221 waveform_amplitude is peak, not RMS",
            "demod_output_convention": (
                "RMS; ZI demodulator X/Y/R nodes report the RMS amplitude of "
                "the input signal's component at the reference frequency"),
            "1f_X_V":       d1["x_mean"],
            "1f_Y_V":       d1["y_mean"],
            "1f_R_V":       d1["r_mean"],
            "1f_theta_deg": d1["theta_mean"],
            "1f_R_std_V":   d1["r_std"],
            "1f_overload":  d1["overload"],
            "2f_X_V":       d2["x_mean"],
            "2f_Y_V":       d2["y_mean"],
            "2f_R_V":       d2["r_mean"],
            "2f_theta_deg": d2["theta_mean"],
            "2f_R_std_V":   d2["r_std"],
            "2f_overload":  d2["overload"],
            "magnet_current_A":  magnet_current_A,
            "assist_field_measured_mT": field_measured_mT,
            "field_angle_from_oop_deg": field_angle_from_oop_deg,
            "temperature_1_K":   t1_K,
            "temperature_2_K":   t2_K,
        }
        records.append(record)
        if on_point is not None:
            on_point(record)

        if write_csv is not None:
            write_csv(records)
        else:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(output_file, index=False)

        log.info("amp %d/%d  V_pulse=%.4g V  V_1f=%.4e V  V_2f=%.4e V",
                 a_idx + 1, len(points), pt.amplitude_V, d1["r_mean"], d2["r_mean"])

    log.info("Done. %d rows → '%s'", len(records), output_file)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    k_cfg = Keithley4200AConfig(visa_resource="GPIB0::17::INSTR")

    pmu_cfg = PMUPulseConfig(
        pmu_channel=1, pmu_id="PMU1",
        width_s=1e-6, rise_s=20e-9, fall_s=20e-9, period_s=1e-3,
        v_range_V=10.0, i_range_A=0.01,
        dut_res_ohm=1e3,
        v_limit_V=5.0,
    )
    read_cfg = ReadConfig(sense_current_A=1e-4, n_averages=50, delay_after_pulse_s=1.0)
    _check_read_safety(read_cfg)   # before connect_ac_source — connect() leaves the 6221 live

    ac_cfg = ACSourceConfig(visa_resource="GPIB0::20::INSTR",
                            amplitude_A=read_cfg.sense_current_A,
                            frequency_Hz=977.0, compliance_V=read_cfg.compliance_V,
                            phasemarker_line=1)
    magnet_cfg = MagnetConfig(visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
                              voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05)
    gauss_cfg = GaussmeterConfig(visa_resource="GPIB0::12::INSTR", unit="T", n_averages=10)
    temp_cfg = TemperatureControllerConfig(
        visa_resource="TCPIP0::192.168.1.5::7020::SOCKET", sensor_uids=("MB1.T1",))

    MFLI_DEVICE = "dev1234"
    extref_cfg = ExtRefConfig(device=MFLI_DEVICE, aux_input_ch=0, osc_index=0)
    shared_filter = FilterConfig(time_constant_s=0.3, order=4, sinc_filter=True)
    demod1_cfg = DemodConfig(device=MFLI_DEVICE, demod_index=0, harmonic=1,
                             osc_index=0, filter=shared_filter)
    demod2_cfg = DemodConfig(device=MFLI_DEVICE, demod_index=1, harmonic=2,
                             osc_index=0, filter=shared_filter)

    FIELD_ANGLE_FROM_OOP_DEG = 85.0
    STATIC_MAGNET_CURRENT_A = 1.5
    AMPLITUDES_V = list(linear_sweep(0.2, 2.0, 0.1, bidirectional=True))
    OUTPUT_FILE = str(_DATA_DIR / f"sot_pulsed_2h_{datetime.now():%Y%m%d_%H%M%S}.csv")

    k4200 = connect_4200a(k_cfg)
    log.info("Installed user libraries (UL):\n%s", list_user_libraries(k4200))
    configure_pmu_pulse(k4200, pmu_cfg)

    source = connect_ac_source(ac_cfg)
    _six221_ac_output_off(source)

    daq = connect("localhost", 8004)
    connect_device(daq, MFLI_DEVICE, interface="1GbE")
    configure_external_reference(daq, extref_cfg, ac_cfg.frequency_Hz)
    configure_demodulator(daq, demod1_cfg)
    configure_demodulator(daq, demod2_cfg)

    magnet = connect_magnet(magnet_cfg)
    gaussmeter = connect_gaussmeter(gauss_cfg)
    temp_ctrl = connect_temperature_controller(temp_cfg)

    set_magnet_current(magnet, magnet_cfg, STATIC_MAGNET_CURRENT_A, gaussmeter, gauss_cfg)

    points = [AmplitudePoint(amplitude_V=float(v)) for v in AMPLITUDES_V]
    try:
        df = run_measurement(k4200, pmu_cfg, source, daq, demod1_cfg, demod2_cfg,
                             extref_cfg, read_cfg, points,
                             gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
                             temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
                             magnet_current_A=STATIC_MAGNET_CURRENT_A,
                             field_angle_from_oop_deg=FIELD_ANGLE_FROM_OOP_DEG,
                             output_file=OUTPUT_FILE)
        print("\n", df.to_string(index=False))
    finally:
        # 6221 down first (shares the channel pin), then the 4200A, then the
        # magnet — never ramp an inductive field while the DUT carries current.
        safe_shutdown("6221", lambda: shutdown_ac_source(source))
        safe_shutdown("4200A", lambda: shutdown_4200a(k4200, channels=()))
        safe_shutdown("magnet", lambda: shutdown_magnet(magnet, magnet_cfg))
        safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        safe_shutdown("temperature controller", lambda: shutdown_temperature_controller(temp_ctrl))


if __name__ == "__main__":
    main()
