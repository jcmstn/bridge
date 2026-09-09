#!/usr/bin/env python3
"""
SOT pulsed switching — 4200A PMU write pulse + delayed dual-SMU R_xy readout
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-08

Stage 4 of the SOT plan: probability-of-switching vs. pulse amplitude for an
in-plane ferromagnet, read out anomalous-Hall. Everything electrical is on the
4200A — the PMU writes, two SMUs read.

Per cycle:
  1. park SMU1 at 0 A  (relays never switch under load)
  2. optional reset pulse   (PMU, opposite polarity, fixed amplitude)
  3. write pulse            (PMU, the swept amplitude) — the KULT module routes
     RPM1 to the PMU on entry and back to the SMU on exit, so nothing else has
     to sequence the pathway
  4. wait ``delay_after_pulse_s``   (e.g. 5 s — let the state settle and any
     transient thermal gradient die away before reading)
  5. SMU1 forces ±I_read, SMU2 reads V_xy — reversal-averaged
  6. SMU1 back to 0 A — channel quiet until the next pulse

Repeated ``n_repeats`` times per amplitude for the probability statistics
(Stage 4 wants ~50-100; that also wants the reset pulse ON so each write starts
from a known state).

Wiring
------
    4200A PMU1-1 ──▶ RPM1 ─┐
    4200A SMU1   ──▶ RPM1 ─┴──▶ main channel of the Hall cross
      Force and sense share one wire path (2-wire). RPM1 switches which of the
      two reaches the DUT; the KULT module owns that switch.

    4200A SMU2   ──▶ transverse (Hall) arms   (forces 0 A, reads V_xy)
      Direct-wired, no RPM. An RPM is a *current* preamp + pathway switch, so
      it would do nothing for a voltage measurement — not having a second one
      costs this measurement nothing.

    Kepco BOP-GL   ──GPIB──▶ electromagnet (ONE static tilted field, set once)
    Lake Shore 475 ──GPIB──▶ Gaussmeter probe at the sample

Why 2-wire on the main channel is fine
--------------------------------------
R_xy = V_xy / I. V_xy comes from a separate contact pair (SMU2), so no lead
drop enters the numerator, and I is set and measured by SMU1. The series lead +
contact resistance cancels out of the Hall number entirely. What 2-wire does
cost is an *absolute* R_xx: ``channel_voltage_V`` below includes the leads, so
treat it as a relative heating monitor, not a resistance.

Reading the pulse back
----------------------
The PMU forces VOLTAGE, and on a 2-wire path that voltage includes the cable
and contact drop — so ``pulse_current_measured_A`` is the physically meaningful
pulse axis for an I50% fit, not ``pulse_amplitude_V``. Two caveats:

  * With an RPM on the 10 V range the current MEASURE range caps at 10 mA. The
    PMU still sources well past that, but the measurement saturates, and the
    module sets KI_LIM_MODE=KI_VALUE so you get an overflowed number rather
    than an error. Check that column on the first run.
  * ``pulse_2wire_resistance_ohm`` (measured V / measured I) is the in-situ
    heating monitor during the pulse — again 2-wire, so relative only.

Current reversal
----------------
The read current is small, fixed, and independent of the pulse, so ±I_read
reversal IS available here (unlike sot_switching.py, where the current is the
swept axis). It cancels the thermal EMF and SMU2's static offset for 2x the
read time, and is on by default. See docs/current-reversal.md.

    # ponytail: SMU2-as-voltmeter has tens-of-uV-class offset drift, which is
    # this measurement's real noise floor. If the two states' V_xy contrast is
    # not comfortably above it, move V_xy to the 2182 (SMU1 keeps forcing) —
    # instruments/keithley2182.py, ~20 lines here.

Field
-----
A single STATIC external field (Kepco magnet, measured by the Lake Shore 475),
set once by the caller before the run — held at a slight angle out of the film
plane so the AHE sees a bistable m_z and the two in-plane remanent states read
as different R_xy. ``field_angle_from_oop_deg`` (0° = OOP, 90° = in-plane) is
the mount tilt, recorded on every row, same convention as
``dc/dc_hall_measurement.py``. To check the ±H_z control, re-run at the
opposite field sign — the field is not an axis here.

The RPM pathway trap
--------------------
Keithley's own ``PMU_1Chan_Sweep_Example`` routes RPM1 to the PMU and never
routes back. After running it (or any Clarius pulse test built on it), SMU1
cannot reach the DUT — and ``sot_switching.py`` / ``sot_dc_characterization.py``
will read an open circuit with NO error, because they force from SMU1 through
that same RPM. ``instruments/kult/bridge_sot_pulse.c`` always routes back;
running one pulse through it is the quickest way to recover a stuck pathway.

Requirements: pymeasure, pyvisa, numpy, pandas. KXCI enabled on the 4200A, and
``instruments/kult/bridge_sot_pulse.c`` compiled into a KULT library — see
``instruments/kult/README.md``.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from instruments.keithley4200a import (
    Keithley4200AConfig,
    PMUPulseConfig,
    SMUChannelConfig,
    acquire_measurement,
    acquire_reversal_averaged,
    configure_pmu_pulse,
    configure_smu,
    connect_4200a,
    list_user_libraries,
    pulse_once,
    read_measurement,
    set_source_level,
    shutdown_4200a,
)
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
# Keithley4200AConfig / PMUPulseConfig / SMUChannelConfig come from
# instruments.keithley4200a; MagnetConfig / GaussmeterConfig /
# TemperatureControllerConfig from their instruments/ modules (imported above
# and re-exported for the TUI). Only the read timing and the pulse-sequence
# shape are local.

@dataclass
class ReadConfig:
    """The delayed R_xy read: SMU1 forces, SMU2 measures.

    No compliance field here — that lives on the two ``SMUChannelConfig``s,
    which is what ``set_source_level`` actually reads."""
    read_current_A: float       = 1e-4    # SMU1 probe current for the Hall read [A]
    source_delay_s: float       = 0.05    # settle after each +I/-I flip [s]
    n_reversals: int            = 5       # reversal pairs (or plain V samples) per read
    settle_before_read_s: float = 0.3     # extra dwell before the first read [s]
    reversal_enabled: bool      = True    # ±I_read decomposition — see docs/current-reversal.md


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


def _pulse_resistance(pinfo: dict) -> Optional[float]:
    """2-wire resistance seen by the pulse, or None when the module returned no
    measured values (``return_names`` empty) or a zero current. Includes leads
    and contacts — a relative heating monitor, not a channel resistance."""
    v = pinfo.get("pulse_voltage_measured_V")
    i = pinfo.get("pulse_current_measured_A")
    if v is None or i is None or i == 0:
        return None
    return v / i


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop  (docs/architecture.md §3 contract)
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    k4200,
    pmu_cfg: PMUPulseConfig,
    src_cfg: SMUChannelConfig,
    hall_cfg: SMUChannelConfig,
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
    park SMU1 → [reset pulse] → write pulse → wait → SMU1 ±I_read / SMU2 reads
    V_xy → park SMU1. One row per cycle; CSV rewritten in full every row.

    ``src_cfg`` is the SMU that forces the read current through the main
    channel (SMU1, behind RPM1); ``hall_cfg`` is the SMU parked at 0 A across
    the Hall arms (SMU2). The caller parks ``hall_cfg`` once before calling.

    The static field is set by the caller before this is called;
    ``magnet_current_A`` is recorded nominal, and if a gaussmeter is passed the
    field is measured once here → ``assist_field_measured_mT``.

    ``stop_event`` is checked before each cycle, inside every wait, and
    mid-reversal. ``temp_ctrl=None`` never stops the run.
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
                set_source_level(k4200, src_cfg, 0.0)
                return pd.DataFrame(records)

            # ── 1. park SMU1 — relays must not switch under load ────────────
            set_source_level(k4200, src_cfg, 0.0)

            # ── 2. optional reset pulse ───────────────────────────────────
            if seq_cfg.reset_enabled:
                pulse_once(k4200, pmu_cfg, amplitude_V=seq_cfg.reset_amplitude_V,
                           stop_event=stop_event)
                _interruptible_sleep(seq_cfg.reset_delay_after_s, stop_event)

            # ── 3. write pulse (the module owns the RPM pathway) ──────────
            pinfo = pulse_once(k4200, pmu_cfg, amplitude_V=pt.amplitude_V,
                               stop_event=stop_event)

            # ── 4. wait ─────────────────────────────────────────────────
            _interruptible_sleep(seq_cfg.delay_after_pulse_s, stop_event)
            _interruptible_sleep(read_cfg.settle_before_read_s, stop_event)

            # ── 5. read R_xy: SMU1 forces ±I_read, SMU2 measures V_xy ────
            if read_cfg.reversal_enabled:
                rv = acquire_reversal_averaged(
                    k4200, src_cfg, hall_cfg, read_cfg.read_current_A,
                    read_cfg.n_reversals, stop_event,
                    source_delay_s=read_cfg.source_delay_s)
                v_xy, v_xy_sem = rv["mean"], rv["sem"]
                v_even, v_even_sem = rv["even_mean"], rv["even_sem"]
                n_rev_used = rv["n_reversals"]
            else:
                set_source_level(k4200, src_cfg, read_cfg.read_current_A)
                _interruptible_sleep(read_cfg.source_delay_s, stop_event)
                av = acquire_measurement(k4200, hall_cfg, read_cfg.n_reversals, stop_event)
                v_xy, v_xy_sem = av["mean"], av["sem"]
                v_even = v_even_sem = None
                n_rev_used = 0

            r_xy = v_xy / read_cfg.read_current_A
            channel_V = read_measurement(k4200, src_cfg)   # 2-wire heating monitor

            # ── 6. park SMU1 again ──────────────────────────────────────
            # Explicit: acquire_reversal_averaged returns with SMU1 forcing
            # +level, so without this the LAST cycle leaves current in the DUT
            # until shutdown_4200a().
            set_source_level(k4200, src_cfg, 0.0)

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
                "pulse_2wire_resistance_ohm": _pulse_resistance(pinfo),
                "pulse_base_voltage_V": pinfo.get("pulse_base_voltage_V"),
                "pulse_base_current_A": pinfo.get("pulse_base_current_A"),
                "reset_enabled":     seq_cfg.reset_enabled,
                "reset_amplitude_V": seq_cfg.reset_amplitude_V if seq_cfg.reset_enabled else None,
                "read_current_A":    read_cfg.read_current_A,
                "channel_voltage_V": channel_V,
                "hall_voltage_V":    v_xy,
                "hall_voltage_sem_V": v_xy_sem,
                "hall_voltage_even_V":     v_even,
                "hall_voltage_even_sem_V": v_even_sem,
                "hall_resistance_ohm": r_xy,
                "n_reversals":       n_rev_used,
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

    # Defaults target instruments/kult/bridge_sot_pulse.c — compile it in KULT
    # first (instruments/kult/README.md), then only the ranges/timing below
    # need touching.
    pmu_cfg = PMUPulseConfig(
        pmu_channel=1, pmu_id="PMU1",
        width_s=100e-9, rise_s=20e-9, fall_s=20e-9, period_s=1e-3,
        v_range_V=10.0, i_range_A=0.01,   # RPM 10 V range caps the measure range at 10 mA
        dut_res_ohm=1e3,                  # ← set near your real channel R (sot_dc_characterization)
        v_limit_V=5.0,
    )
    # SMU1 forces the read current through RPM1 into the main channel; SMU2 sits
    # at 0 A across the Hall arms. four_wire=False: the RPM1 path is shared
    # 2-wire, and this field is recorded provenance only (KXCI cannot set it).
    src_cfg = SMUChannelConfig(channel=1, source_function="current",
                               compliance_voltage_V=2.0, four_wire=False,
                               source_limit_A=10e-3)
    hall_cfg = SMUChannelConfig(channel=2, source_function="current",  # forces 0 A
                                compliance_voltage_V=2.0, four_wire=False,
                                source_limit_A=1e-9)

    read_cfg = ReadConfig(read_current_A=1e-4, n_reversals=5, reversal_enabled=True)
    seq_cfg = PulseSequenceConfig(
        delay_after_pulse_s=5.0, n_repeats=50, reset_enabled=True, reset_amplitude_V=-2.0,
        output_file=str(_DATA_DIR / f"sot_pulsed_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

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
    configure_smu(k4200, src_cfg)
    configure_smu(k4200, hall_cfg)
    set_source_level(k4200, hall_cfg, 0.0)   # park SMU2 as the voltmeter
    set_source_level(k4200, src_cfg, 0.0)

    magnet = connect_magnet(magnet_cfg)
    gaussmeter = connect_gaussmeter(gauss_cfg)
    temp_ctrl = connect_temperature_controller(temp_cfg)

    set_magnet_current(magnet, magnet_cfg, STATIC_MAGNET_CURRENT_A, gaussmeter, gauss_cfg)

    points = [AmplitudePoint(amplitude_V=float(v)) for v in AMPLITUDES_V]
    try:
        df = run_measurement(k4200, pmu_cfg, src_cfg, hall_cfg, read_cfg, seq_cfg, points,
                             gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
                             temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
                             magnet_current_A=STATIC_MAGNET_CURRENT_A,
                             field_angle_from_oop_deg=FIELD_ANGLE_FROM_OOP_DEG)
        print("\n", df.to_string(index=False))
    finally:
        # SMUs down before the magnet — never ramp an inductive field while the
        # DUT still carries current.
        safe_shutdown("4200A", lambda: shutdown_4200a(k4200))
        safe_shutdown("magnet", lambda: shutdown_magnet(magnet, magnet_cfg))
        safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        safe_shutdown("temperature controller", lambda: shutdown_temperature_controller(temp_ctrl))


if __name__ == "__main__":
    main()
