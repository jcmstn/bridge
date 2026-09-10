#!/usr/bin/env python3
"""
SOT pulsed switching — 4200A PMU write pulse + delayed 6221/2182 R_xy readout
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-08

The switching curve: anomalous-Hall R_xy vs. pulse amplitude, one pulse per
amplitude, amplitude swept up and down at a fixed tilted field. The 4200A ONLY
pulses; a 6221 + 2182 pair does the delayed R_xy read.

Per amplitude:
  1. 6221 output OFF  (never pulse into a live current source on the shared pin)
  2. write pulse      (PMU, the swept amplitude) — the KULT module routes RPM1
     to the PMU on entry and back to the SMU on exit, so nothing else has to
     sequence the pathway
  3. wait ``delay_after_pulse_s``   (let the state settle and any transient
     thermal gradient die away before reading)
  4. 6221 output ON, settle, reversal-averaged R_xy read (6221 forces ±I_read,
     2182 reads V_xy across the transverse arms)
  5. 6221 output OFF again — channel quiet until the next pulse

One row per amplitude. Make the amplitude list a full loop (0 → +max → 0 →
−max → 0) — the sweep itself sets each pulse's starting state, which is what
gives the hysteresis. There is no reset pulse and no per-amplitude repetition:
for switching-*probability* statistics (P(V), I50, thermal stability Δ), re-run
the whole sweep N times and take the switched fraction per amplitude across the
N runs.

Wiring  (2-wire local sense, FORCE triax only)
---------------------------------------------
    4200A PMU1-1 ──▶ RPM1 ──▶ I+ pad of the Hall-cross main channel
      Only the RPM FORCE triax is wired, and the RPM channel is in LOCAL
      (2-wire) sense — its SENSE output is capped. Triax breakout:
        centre = force            → I+ pad
        inner  = guard            → floating (unterminated)
        outer  = circuit COMMON   → the common bus (below)
      The KULT module routes RPM1 onto the pulse pathway for the burst and
      back to the SMU pathway on exit, which lifts the PMU's 50 Ω output out
      of the channel while the 6221 reads.

    COMMON BUS ──▶ I- pad of the Hall cross
      The 4200A circuit common (PMU FORCE outer shell) and the 6221 output LO
      both land here. During the read it is circuit common; during the pulse
      it is the pulse-current return.

    Keithley 6221 (current source)   HI ──▶ I+ pad ;  LO ──▶ common bus
      In parallel with the PMU on the main channel — hence the OFF/ON dance
      (never pulse into a live current source). Confirm the 6221 is actually
      landed on I+ / common before the first run.

    Keithley 2182 (nanovoltmeter)  Ch 1 ──▶ transverse (Hall) voltage arms
      Floating differential input — a bare 4200A SMU could not do this read
      (its LO is bonded to circuit common, so it only ever measures
      arm-to-common, i.e. R_xx with a Hall ripple).

    The two 4200A SMUs and the 4225 / 4200-PA preamps are unused.

    Kepco BOP-GL   ──GPIB──▶ electromagnet (ONE static tilted field, set once)
    Lake Shore 475 ──GPIB──▶ Gaussmeter probe at the sample

Why the readout is clean
------------------------
R_xy = V_xy / I. V_xy comes from the 2182 across a separate contact pair, so no
lead drop enters the numerator; I is set and measured by the 6221. Series lead +
contact resistance cancels out of the Hall number entirely.

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
reversal works: the 6221 flips polarity, the 2182 reads each side, and the
odd/even split cancels the thermal EMF and the 2182's static offset for 2x
the read time. See docs/current-reversal.md.

Instrument protection (the 6221 shares the main-channel pins with the PMU)
------------------------------------------------------------------------
What makes the shared-bus wiring safe is that the 6221 output relay is OPEN
during every pulse — ``_six221_output_off()`` runs before every ``pulse_once``
(and on abort). With the output disabled the 6221 only sees the pulse voltage
across open terminals (<= ``v_limit_V``, 5 V by default; hardware <= 10 V on
the default range), far inside its +/-105 V output isolation. The OFF/ON
ordering is load-bearing and is covered by ``tests/test_sot_run_loops.py``.

Guards against a fat-fingered read setting (they would put a large DC current
or voltage on the shared bus, hence on the 2182 and the disabled PMU/6221):

  * ``run_measurement`` raises if ``sense_current_A`` > 10 mA or
    ``compliance_V`` > 21 V — nothing this measurement can legitimately need,
    so the ceiling catches ``1e-4`` typed as ``1e-1`` etc. The TUI also warns
    softly above 1 mA / 5 V.
  * The PMU pulse amplitude is clamped by ``PMUPulseConfig.v_limit_V`` in
    ``configure_pmu_pulse`` / ``pulse_once``, and on the default 10 V range
    the RPM caps the pulse current near 10 mA in hardware.

The 40 V range
--------------
It exists for higher switching currents; be deliberate about it:

  * DUT (the real risk): the bare PMU can source up to 0.8 A. That current
    flows I+ → channel → I- → common bus, so 0.8 A (or even 100 mA) through
    a narrow Hall channel is Joule heating / electromigration / physical
    destruction. Set ``v_limit_V`` so amplitude / channel-R keeps the pulse
    current where you want it, ramp the amplitude up, and watch
    ``pulse_current_measured_A``.
  * 4225-RPM: it is a 10 V device. The rig pulses through RPM1
    (``KI_RPM_PULSE``); a 40 V pulse there either returns an LPT error (the
    consecutive-failure abort catches it) or the firmware bypasses the RPM,
    leaving you on the bare PMU 40 V current ranges (100 µA / 10 mA / 0.8 A).
    Fire one 40 V pulse and check ``EX`` returns 0 before trusting it.
  * 2182: the pulse puts a common-mode transient of roughly half the channel
    drop on the Hall arms (~10-20 V for a 40 V pulse vs ~5 V at 10 V), plus
    fast-edge spikes. The 2182 CH1 HI-LO damage limit is 120 V and its
    common-mode limit ~±60 V, so a ~10-20 V transient is well inside both,
    and the read is ``delay_after_pulse_s`` later so any input-amp saturation
    has recovered. Keep the 2182 leads short, twisted, and routed away from
    the PMU/RPM triax and the I+/I- pulse leads; the ≥100 ns rise/fall floor
    the 40 V range enforces already cuts the edge spikes.
  * 6221: its output relay is OPEN during every pulse, so it only sees the
    pulse voltage across the open contacts — 40 V vs its ~±105 V output
    isolation. The dV/dt spike through the open relay's few-pF parasitic is
    ~4x the 10 V case but brief (~10 mA, ~10 ns) into a protection network
    built for hot-circuit connection. Confirm the output-off state is NORMAL
    (a ZERO setting keeps the relay closed and puts that transient into the
    connected output amp).

The instrument risk (6221 / 2182) at 40 V is small with output-off = NORMAL,
slow edges, and 2182 leads dressed away from the pulse path. The DUT risk is
inherent and must be managed by ``v_limit_V`` and the measured pulse current.

Two 6221 front-panel settings this code cannot read back — check them once:
  * Output-off state = NORMAL (factory default: the relay opens). If it is
    set to ZERO the relay stays closed and the 6221 output stage eats every
    pulse transient. The "safe" claim above depends on this.
  * Low-terminal earth (OUTPUT LOW) = floating. The external common bus
    already references I- to the 4200A common; a second internal earth is a
    ground loop through that bus.

During ``delay_after_pulse_s`` the 6221 is OFF, so the main channel is
open-circuit for that wait and charge on the Hall arms has no bleed path. If
the first read after the wait looks erratic, that is the suspect — not a
damaged 2182.

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
routes back. After running it (or any Clarius pulse test built on it), an SMU
wired through RPM1 cannot reach the DUT and reads an open circuit with NO
error. ``instruments/kult/bridge_sot_pulse.c`` always routes back; running one
pulse through it is the quickest way to recover a stuck pathway.

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
# TemperatureControllerConfig from their instruments/ modules (imported above
# and re-exported for the TUI). Only the read timing and the post-pulse wait
# are local.

# Absolute ceilings for the delayed read — not tuning knobs. Anything this
# measurement legitimately needs is far below them; the point is to stop a
# mistyped exponent putting a large current/voltage on the shared main-channel
# bus (and thus on the 2182 and the disabled PMU/6221). See the "Instrument
# protection" section of the module docstring.
_READ_CURRENT_CEILING_A     = 10e-3
_READ_COMPLIANCE_CEILING_V  = 21.0


@dataclass
class ReadConfig:
    """The 6221 + 2182 delayed R_xy read, plus the wait before it."""
    sense_current_A: float        = 1e-4    # 6221 probe current for the Hall read [A]
    compliance_V: float           = 2.0
    source_delay_s: float         = 0.05    # 6221 settle after each +I/-I flip [s]
    nplc: float                   = 5.0     # 2182 integration
    auto_range: bool              = True
    n_reversals: int              = 5       # +I/-I reversal pairs averaged per read
    settle_after_enable_s: float  = 0.3     # dwell after re-enabling the 6221, before reading [s]
    delay_after_pulse_s: float    = 1.0     # wait between write-pulse end and the read [s]


def _check_read_safety(read_cfg: ReadConfig) -> None:
    """Refuse a read current / compliance that has no business in this
    measurement — the shared-bus wiring means either would land on the 2182
    and the (disabled) PMU output. Called before connect_source() on every
    entry path (main, the TUI's do_run, and run_measurement itself), because
    connect_source() returns with the 6221 already sourcing sense_current_A."""
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
class AmplitudePoint:
    amplitude_V: float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interruptible_sleep(seconds: float, stop_event: Optional[threading.Event]) -> None:
    """time.sleep, but return early if ``stop_event`` fires — so the post-pulse
    wait doesn't make Abort feel dead."""
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


# The KULT module returns 0 on success, a negative LPT code on a bad pulse
# (e.g. -826 = no flat top, -824 = timing too short), or KXCI itself replies
# with an "EX ERROR: ..." string. Any of those means the pulse did not fire —
# so does an empty reply.
_MAX_CONSECUTIVE_PULSE_FAILURES = 3


def _pulse_failure_reason(module_return) -> Optional[str]:
    """None if the ``EX`` reply looks like a clean success, else a short reason.
    A non-numeric reply that is not an obvious error is let through (some KXCI
    builds echo status text) rather than aborting a run that is really working."""
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
    output_file: str = "sot_pulsed_switching.csv",
) -> pd.DataFrame:
    """One pulse + delayed R_xy read per amplitude in ``points``:
    (6221 off) → write pulse → wait → (6221 on, read R_xy) → (6221 off).
    One row per amplitude; CSV rewritten in full every row.

    The static field is set by the caller before this is called;
    ``magnet_current_A`` is recorded nominal, and if a gaussmeter is passed
    the field is measured once here → ``assist_field_measured_mT``.

    ``stop_event`` is checked before each amplitude, inside the post-pulse
    wait, and mid-reversal. ``temp_ctrl=None`` never stops the run.

    Raises ``RuntimeError`` after ``_MAX_CONSECUTIVE_PULSE_FAILURES`` write
    pulses in a row come back with a non-zero module return or an ``EX ERROR``
    — a systematic config/hardware fault, not worth grinding the whole sweep
    into rows of pure read noise.
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
            _six221_output_off(source)
            return pd.DataFrame(records)

        # ── 1. 6221 OFF — never pulse into a live current source ────────
        _six221_output_off(source)

        # ── 2. write pulse (the module owns the RPM pathway) ──────────
        pinfo = pulse_once(k4200, pmu_cfg, amplitude_V=pt.amplitude_V,
                           stop_event=stop_event)
        aborting = stop_event is not None and stop_event.is_set()
        fail = None if aborting else _pulse_failure_reason(pinfo.get("module_return"))
        if fail is not None:
            consecutive_pulse_failures += 1
            log.warning("Write pulse did not fire (amp %.4g V, %d in a row): %s",
                        pt.amplitude_V, consecutive_pulse_failures, fail)
            if consecutive_pulse_failures >= _MAX_CONSECUTIVE_PULSE_FAILURES:
                _six221_output_off(source)
                raise RuntimeError(
                    f"{consecutive_pulse_failures} consecutive pulse failures "
                    f"— last: {fail}. Aborting; check the KXCI log and the pulse "
                    "timing / PMU config.")
        else:
            consecutive_pulse_failures = 0

        # ── 3. wait ─────────────────────────────────────────────────
        _interruptible_sleep(read_cfg.delay_after_pulse_s, stop_event)

        # ── 4. 6221 ON, settle, reversal-averaged R_xy read ──────────
        source.enable_source()
        _interruptible_sleep(read_cfg.settle_after_enable_s, stop_event)
        rv = acquire_reversal_averaged_voltage(
            source, voltmeter, read_cfg.sense_current_A, read_cfg.n_reversals,
            stop_event, source_delay_s=read_cfg.source_delay_s)
        r_xy = rv["mean"] / read_cfg.sense_current_A

        # ── 5. 6221 OFF again ───────────────────────────────────────
        _six221_output_off(source)

        t1_K, t2_K = read_temperature(temp_ctrl, temp_cfg) if temp_cfg is not None else (None, None)

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
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(output_file, index=False)

        log.info("amp %d/%d  V_pulse=%.4g V  R_xy=%.6g Ω",
                 a_idx + 1, len(points), pt.amplitude_V, r_xy)

    log.info("Done. %d rows → '%s'", len(records), output_file)
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
        width_s=1e-6, rise_s=20e-9, fall_s=20e-9, period_s=1e-3,
        v_range_V=10.0, i_range_A=0.01,   # RPM 10 V range caps the measure range at 10 mA
        dut_res_ohm=1e3,                  # ← set near your real channel R (4-probe it first)
        v_limit_V=5.0,
    )
    read_cfg = ReadConfig(sense_current_A=1e-4, n_reversals=5, nplc=5, delay_after_pulse_s=1.0)
    _check_read_safety(read_cfg)   # before connect_source — connect() leaves the 6221 live

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
    # Full loop: up then down. The sweep itself sets each pulse's starting state.
    AMPLITUDES_V = list(linear_sweep(0.2, 2.0, 0.1, bidirectional=True))
    OUTPUT_FILE = str(_DATA_DIR / f"sot_pulsed_{datetime.now():%Y%m%d_%H%M%S}.csv")

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
        df = run_measurement(k4200, pmu_cfg, source, voltmeter, read_cfg, points,
                             gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
                             temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
                             magnet_current_A=STATIC_MAGNET_CURRENT_A,
                             field_angle_from_oop_deg=FIELD_ANGLE_FROM_OOP_DEG,
                             output_file=OUTPUT_FILE)
        print("\n", df.to_string(index=False))
    finally:
        # 6221 down first (it shares the channel pin), then the 4200A, then the
        # magnet — never ramp an inductive field while the DUT still carries current.
        safe_shutdown("6221 (ramp)", lambda: ramp_current_to_zero(source))
        safe_shutdown("6221", lambda: shutdown_source(source))
        # channels=() — this program never forces the 4200A SMUs, so there is
        # nothing to zero; the KULT module already left the PMU/RPM clean.
        safe_shutdown("4200A", lambda: shutdown_4200a(k4200, channels=()))
        safe_shutdown("magnet", lambda: shutdown_magnet(magnet, magnet_cfg))
        safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        safe_shutdown("temperature controller", lambda: shutdown_temperature_controller(temp_ctrl))


if __name__ == "__main__":
    main()
