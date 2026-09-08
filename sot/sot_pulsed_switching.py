#!/usr/bin/env python3
"""
SOT pulsed switching — 4200A PMU write pulse + delayed 6221/2182 R_xy readout
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-08

Stage 4 of the SOT plan: probability-of-switching vs. pulse amplitude for an
in-plane ferromagnet, read out anomalous-Hall.

Per cycle:
  1. 6221 output OFF  (never pulse into a live current source on the same channel)
  2. optional reset pulse   (4200A PMU, opposite polarity, fixed amplitude)
  3. write pulse            (4200A PMU, the swept amplitude)
  4. wait ``delay_after_pulse_s``   (e.g. 5 s — let the state settle / any
     transient thermal gradient die away before reading)
  5. 6221 output ON, settle, reversal-averaged R_xy read (6221 + 2182)
  6. 6221 output OFF again — channel left quiet until the next pulse

Repeated ``n_repeats`` times per amplitude for the probability statistics
(Stage 4 wants ~50-100; that also wants the reset pulse ON so each write
starts from a known state).

Field
-----
A single STATIC external field (Kepco magnet, measured by the Lake Shore
475), set once by the caller before the run — held at a slight angle out of
the film plane so the AHE sees a bistable m_z and the two in-plane remanent
states read as different R_xy. ``field_angle_from_oop_deg`` (0° = OOP, 90° =
in-plane) is the mount tilt, recorded on every row, same convention as
``dc/dc_hall_measurement.py``. To check the ±H_z control, re-run at the
opposite field sign — the field is not an axis here.

Wiring
------
    4200A PMU (KXCI / GPIB 17)
      PMU ch → RPM → channel current leads of the Hall bar   (write pulse only)

    Keithley 6221 (current source)   Output → SAME channel current leads
      shares the channel pin with the PMU — hence the OFF/ON dance above

    Keithley 2182 (nanovoltmeter)    Ch 1 → transverse (Hall) voltage leads

    Kepco BOP-GL   ──GPIB──▶ electromagnet (static field, set once)
    Lake Shore 475 ──GPIB──▶ Gaussmeter probe at the sample

    # ponytail: PMU and 6221 sharing the channel pin is assumed. If they are
    # on separate arms and never share a pin, the disable/enable_source() calls
    # in the loop are harmless but unnecessary — leave them, they cost ~ms.

The PMU pulse mechanism (KULT module over KXCI) and why the module name is
config, not a default: see instruments/keithley4200a.py "PMU" section.

Requirements: pymeasure, pyvisa, numpy, pandas. KXCI enabled on the 4200A;
a pulse KULT module identified via ``list_user_libraries()`` (KXCI ``UL``).
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from pymeasure.instruments.keithley import Keithley2182, Keithley6221

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
    SourceConfig,
    acquire_reversal_averaged_voltage,
    connect_source,
    ramp_current_to_zero,
    shutdown_source,
)
from instruments.keithley2182 import VoltmeterConfig, connect_voltmeter
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
# Keithley4200AConfig / PMUPulseConfig come from instruments.keithley4200a;
# SourceConfig / VoltmeterConfig / MagnetConfig / GaussmeterConfig /
# TemperatureControllerConfig from their instruments/ modules (re-exported here
# for the TUI). Only the read timing and the pulse-sequence shape are local.

@dataclass
class ReadConfig:
    """The 6221 + 2182 delayed R_xy read."""
    sense_current_A: float        = 1e-4    # 6221 probe current for the Hall read [A]
    compliance_V: float           = 2.0
    source_delay_s: float         = 0.05    # 6221 settle after each +I/-I flip [s]
    nplc: float                   = 5.0     # 2182 integration
    auto_range: bool              = True
    n_reversals: int              = 5       # +I/-I reversal pairs averaged per read
    settle_after_enable_s: float  = 0.3     # dwell after re-enabling the 6221, before reading [s]


@dataclass
class PulseSequenceConfig:
    delay_after_pulse_s: float  = 5.0      # wait between write-pulse end and the R_xy read [s]
    n_repeats: int              = 50       # write/read cycles per amplitude (probability stats)
    reset_enabled: bool         = False    # opposite-polarity reset pulse before each write
    reset_amplitude_V: float    = 0.0      # reset-pulse amplitude (signed), used iff reset_enabled
    reset_delay_after_s: float  = 0.01     # settle after the reset pulse, before the write pulse [s]
    output_file: str            = "sot_pulsed_switching.csv"


@dataclass
class AmplitudePoint:
    amplitude_V: float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interruptible_sleep(seconds: float, stop_event: Optional[threading.Event]) -> None:
    """time.sleep, but return early if ``stop_event`` fires — so a 5 s
    post-pulse wait doesn't make Abort feel dead."""
    if seconds <= 0:
        return
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if stop_event is not None and stop_event.is_set():
            return
        time.sleep(min(0.2, end - time.monotonic()))


def _six221_output_off(source: Keithley6221) -> None:
    """Zero and disable the 6221 output — the state it must be in whenever the
    PMU pulses the shared channel pin."""
    source.source_current = 0.0
    source.disable_source()


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop  (docs/architecture.md §3 contract)
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    k4200,
    pmu_cfg: PMUPulseConfig,
    source: Keithley6221,
    voltmeter: Keithley2182,
    read_cfg: ReadConfig,
    seq_cfg: PulseSequenceConfig,
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
) -> pd.DataFrame:
    """For each amplitude in ``points`` × ``seq_cfg.n_repeats`` cycles:
    (6221 off) → [reset pulse] → write pulse → wait → (6221 on, read R_xy) →
    (6221 off). One row per cycle; CSV rewritten in full every row.

    The static field is set by the caller before this is called;
    ``magnet_current_A`` is recorded nominal, and if a gaussmeter is passed
    the field is measured once here → ``assist_field_measured_mT``.

    ``stop_event`` is checked before each cycle, inside the post-pulse wait,
    and mid-reversal. ``temp_ctrl=None`` never stops the run.
    """
    field_measured_mT = None
    if gaussmeter is not None and gauss_cfg is not None:
        field_measured_mT = read_field_mT(gaussmeter, gauss_cfg)
        log.info("Static field: %.4f mT measured (magnet current %s A)",
                 field_measured_mT, magnet_current_A)

    records: List[dict] = []
    total = len(points) * seq_cfg.n_repeats

    for a_idx, pt in enumerate(points):
        for rep in range(seq_cfg.n_repeats):
            if stop_event is not None and stop_event.is_set():
                log.info("Aborted after %d / %d cycles.", len(records), total)
                _six221_output_off(source)
                return pd.DataFrame(records)

            # ── 1. 6221 OFF — never pulse into a live current source ────────
            _six221_output_off(source)

            # ── 2. optional reset pulse ───────────────────────────────────
            if seq_cfg.reset_enabled:
                pulse_once(k4200, pmu_cfg, amplitude_V=seq_cfg.reset_amplitude_V,
                           stop_event=stop_event)
                _interruptible_sleep(seq_cfg.reset_delay_after_s, stop_event)

            # ── 3. write pulse ───────────────────────────────────────────
            pinfo = pulse_once(k4200, pmu_cfg, amplitude_V=pt.amplitude_V,
                               stop_event=stop_event)

            # ── 4. wait ─────────────────────────────────────────────────
            _interruptible_sleep(seq_cfg.delay_after_pulse_s, stop_event)

            # ── 5. 6221 ON, settle, reversal-averaged R_xy read ──────────
            source.enable_source()
            _interruptible_sleep(read_cfg.settle_after_enable_s, stop_event)
            rv = acquire_reversal_averaged_voltage(
                source, voltmeter, read_cfg.sense_current_A, read_cfg.n_reversals,
                stop_event, source_delay_s=read_cfg.source_delay_s)
            r_xy = rv["mean"] / read_cfg.sense_current_A

            # ── 6. 6221 OFF again ───────────────────────────────────────
            _six221_output_off(source)

            t1_K, t2_K = read_temperature(temp_ctrl, temp_cfg) if temp_cfg is not None else (None, None)

            record = {
                "amplitude_index":  a_idx,
                "repeat_index":     rep,
                "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
                "pulse_amplitude_V": pt.amplitude_V,
                "pulse_width_s":     pmu_cfg.width_s,
                "pulse_delay_s":     seq_cfg.delay_after_pulse_s,
                "n_pulses":          pmu_cfg.n_pulses,
                "pulse_voltage_measured_V": pinfo.get("pulse_voltage_measured_V"),
                "pulse_current_measured_A": pinfo.get("pulse_current_measured_A"),
                "reset_enabled":     seq_cfg.reset_enabled,
                "reset_amplitude_V": seq_cfg.reset_amplitude_V if seq_cfg.reset_enabled else None,
                "sense_current_A":   read_cfg.sense_current_A,
                "hall_voltage_V":    rv["mean"],
                "hall_voltage_sem_V": rv["sem"],
                "hall_voltage_even_V":     rv["even_mean"],
                "hall_voltage_even_sem_V": rv["even_sem"],
                "hall_resistance_ohm": r_xy,
                "n_reversals":       rv["n_reversals"],
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
                Path(seq_cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(records).to_csv(seq_cfg.output_file, index=False)

            log.info("amp %d/%d  rep %d/%d  V_pulse=%.4g V  R_xy=%.6g Ω",
                     a_idx + 1, len(points), rep + 1, seq_cfg.n_repeats,
                     pt.amplitude_V, r_xy)

    log.info("Done. %d rows → '%s'", len(records), seq_cfg.output_file)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    k_cfg = Keithley4200AConfig(visa_resource="GPIB0::17::INSTR")
    # ── PMU: set library/module/arg_order from `list_user_libraries(k4200)` ──
    pmu_cfg = PMUPulseConfig(
        library="pmu-dut-examples",
        module="",                       # ← REQUIRED: your installed pulse module
        pmu_channel=1,
        width_s=100e-9, rise_s=20e-9, fall_s=20e-9, period_s=1e-3,
        v_limit_V=5.0, i_limit_A=0.2, i_range_A=0.2,
        return_names=(),                 # e.g. ("pulse_voltage_measured_V", "pulse_current_measured_A")
    )
    read_cfg = ReadConfig(sense_current_A=1e-4, n_reversals=5, nplc=5)
    seq_cfg = PulseSequenceConfig(
        delay_after_pulse_s=5.0, n_repeats=50, reset_enabled=True, reset_amplitude_V=-2.0,
        output_file=str(_DATA_DIR / f"sot_pulsed_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    src_cfg = SourceConfig(visa_resource="GPIB0::20::INSTR", sense_current_A=read_cfg.sense_current_A,
                           compliance_V=read_cfg.compliance_V, source_delay_s=read_cfg.source_delay_s)
    volt_cfg = VoltmeterConfig(visa_resource="GPIB0::7::INSTR", nplc=read_cfg.nplc,
                               auto_range=read_cfg.auto_range)
    magnet_cfg = MagnetConfig(visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
                              voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05)
    gauss_cfg = GaussmeterConfig(visa_resource="GPIB0::12::INSTR", unit="T", n_averages=10)
    temp_cfg = TemperatureControllerConfig(
        visa_resource="TCPIP0::192.168.1.5::7020::SOCKET", sensor_uids=("MB1.T1",))

    FIELD_ANGLE_FROM_OOP_DEG = 85.0     # ← SET TO YOUR REAL MOUNT ANGLE (recorded, not measured)
    STATIC_MAGNET_CURRENT_A = 1.5       # ← the static read field; re-run at -1.5 for the ±Hz check
    AMPLITUDES_V = list(linear_sweep(0.2, 2.0, 0.1, bidirectional=False))

    k4200 = connect_4200a(k_cfg)
    log.info("Installed user libraries (UL):\n%s", list_user_libraries(k4200))
    configure_pmu_pulse(k4200, pmu_cfg)

    source = connect_source(src_cfg)
    _six221_output_off(source)
    voltmeter = connect_voltmeter(volt_cfg)
    magnet = connect_magnet(magnet_cfg)
    gaussmeter = connect_gaussmeter(gauss_cfg)
    temp_ctrl = connect_temperature_controller(temp_cfg)

    set_magnet_current(magnet, magnet_cfg, STATIC_MAGNET_CURRENT_A, gaussmeter, gauss_cfg)

    points = [AmplitudePoint(amplitude_V=float(v)) for v in AMPLITUDES_V]
    try:
        df = run_measurement(k4200, pmu_cfg, source, voltmeter, read_cfg, seq_cfg, points,
                             gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
                             temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
                             magnet_current_A=STATIC_MAGNET_CURRENT_A,
                             field_angle_from_oop_deg=FIELD_ANGLE_FROM_OOP_DEG)
        print("\n", df.to_string(index=False))
    finally:
        safe_shutdown("6221 (ramp)", lambda: ramp_current_to_zero(source))
        safe_shutdown("6221", lambda: shutdown_source(source))
        safe_shutdown("4200A", lambda: shutdown_4200a(k4200))
        safe_shutdown("magnet", lambda: shutdown_magnet(magnet, magnet_cfg))
        safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        safe_shutdown("temperature controller", lambda: shutdown_temperature_controller(temp_ctrl))


if __name__ == "__main__":
    main()
