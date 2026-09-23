"""
Keithley 4200A-SCS Semiconductor Characterization System — KXCI PMU driver
==========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-07

pymeasure ships no 4200A driver. This module is a hand-written wrapper over
**KXCI** (Keithley External Control Interface) — the command server the 4200A
runs so an outside PC can drive it over GPIB or LAN, the same role KXCI plays
for every remote 4200A script.

Scope of THIS module: the **PMU** (4225-PMU) behind a 4225-RPM, as a
fire-one-pulse-burst primitive — ``PMUPulseConfig`` + ``configure_pmu_pulse``
/ ``pulse_once``, which run the KULT module in
``instruments/kult/bridge_sot_pulse.c``. See the "PMU" section lower in this
file for the KXCI mechanism and for the RPM-pathway trap. (A general SMU
force/measure wrapper for the two SMU cards existed until 2026-09-23 with no
program using it; it is in git history if a 4200A SMU program is ever built.)

The PMU here does not measure the switched state — that is a separate delayed
6221 + 2182 read (6221 forces ±I_read through the shared main-channel pin, 2182
reads V_xy across the Hall arms) in ``sot/sot_pulsed_switching.py``. A bare
4200A SMU cannot do that read: its LO is bonded to circuit common, so it only
ever measures arm-to-common. The PMU's only job is to deliver the write pulse.

Transport
---------
KXCI speaks over either:
  * GPIB   — ``visa_resource="GPIB0::17::INSTR"``
  * LAN    — ``visa_resource="TCPIP0::192.168.0.10::1225::SOCKET"`` (1225 = KXCI
             default port; set it in Clarius → Tools → KXCI Configuration)
Both use ``\\n`` line terminators. ``*IDN?`` (or the KXCI ``ID`` query) over the
chosen transport is the whole smoke test.

Session commands sent by ``connect_4200a`` (KXCI **User Mode**):
  ``US``                             select User Mode
  ``BC`` / ``DR0``                   clear buffer / no data-ready SRQ
  ``IT1|IT2|IT3``                    integration time: fast | normal | quiet

Failure policy: the pulse source is **load-bearing** — ``connect_4200a`` and
``pulse_once`` raise on any failure; callers do not pass ``None`` for the
handle. ``shutdown_4200a`` is a plain call with no internal try/except — wrap
it in ``dc.dc_sweep_utils.safe_shutdown``.

Usage example:
    from instruments.keithley4200a import (
        Keithley4200AConfig, PMUPulseConfig, connect_4200a,
        configure_pmu_pulse, pulse_once, shutdown_4200a)

    k = connect_4200a(Keithley4200AConfig(visa_resource="GPIB0::17::INSTR"))
    try:
        cfg = PMUPulseConfig(width_s=100e-9)
        configure_pmu_pulse(k, cfg)
        info = pulse_once(k, cfg, amplitude_V=1.0)
    finally:
        shutdown_4200a(k)
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from pymeasure.instruments import Instrument

from instruments.run_time import PMU_PULSE_S

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_INTEGRATION = {"fast": 1, "normal": 2, "medium": 2, "quiet": 3, "long": 3}


# ─────────────────────────────────────────────────────────────────────────────
# Private KXCI transport  ── nothing else uses it, so it stays in this file ───
# ─────────────────────────────────────────────────────────────────────────────

class _Keithley4200A_KXCI(Instrument):
    """Thin KXCI line-protocol wrapper. Not SCPI: ``includeSCPI=False`` and
    every command is a KXCI mnemonic (see the module docstring)."""

    def __init__(self, adapter, name="Keithley 4200A-SCS (KXCI)", **kwargs):
        super().__init__(
            adapter, name,
            includeSCPI=False,
            read_termination="\n",
            write_termination="\n",
            **kwargs,
        )

    def command(self, cmd: str) -> None:
        """Send a KXCI command that returns no data (a force/config command)."""
        self.write(cmd)

    def query(self, cmd: str) -> str:
        """Send a KXCI command that returns data and read the reply."""
        return self.ask(cmd).strip()

    def close(self) -> None:
        self.adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Keithley4200AConfig:
    """Connection + global acquisition settings for the 4200A KXCI session."""
    visa_resource: str = "GPIB0::17::INSTR"   # or "TCPIP0::<ip>::1225::SOCKET"
    timeout_s: float   = 30.0                 # VISA I/O timeout (long: SMU measure + KXCI)
    integration: str   = "normal"             # "fast" | "normal" | "quiet"  → IT1/IT2/IT3


# ─────────────────────────────────────────────────────────────────────────────
# connect / shutdown
# ─────────────────────────────────────────────────────────────────────────────

def connect_4200a(cfg: Keithley4200AConfig) -> _Keithley4200A_KXCI:
    """Open the KXCI session, select User Mode, apply global settings, return
    the live handle. Raises on any failure (load-bearing)."""
    if cfg.integration not in _INTEGRATION:
        raise ValueError(f"integration must be one of {sorted(set(_INTEGRATION))}, "
                         f"got {cfg.integration!r}")

    dev = _Keithley4200A_KXCI(cfg.visa_resource, timeout=cfg.timeout_s * 1000)
    dev.command("US")                       # User Mode — immediate command execution
    dev.command("BC")                       # clear the KXCI reading buffer
    dev.command("DR0")                      # read synchronously, no data-ready SRQ
    dev.command(f"IT{_INTEGRATION[cfg.integration]}")

    ident = ""
    for probe in ("*IDN?", "ID"):
        try:
            dev.adapter.connection.timeout = 5000
            ident = dev.query(probe)
            if ident:
                break
        except Exception:
            continue
    try:
        dev.adapter.connection.timeout = cfg.timeout_s * 1000
    except Exception:
        pass

    log.info("Keithley 4200A-SCS connected: %s  integration=%s  id=%s",
             cfg.visa_resource, cfg.integration, ident or "(no id reply)")
    return dev


def shutdown_4200a(dev: _Keithley4200A_KXCI) -> None:
    """Close the KXCI session. Nothing here ever forces the SMU cards, and the
    KULT pulse module leaves the PMU idle and the RPM routed back to the SMU
    on every exit path, so there is no output state to zero first. Plain
    call, no try/except — the caller wraps this in ``safe_shutdown``."""
    dev.close()
    log.info("Keithley 4200A-SCS session closed")


# ─────────────────────────────────────────────────────────────────────────────
# PMU  ── fire one VOLTAGE pulse burst, via a KULT module
# ─────────────────────────────────────────────────────────────────────────────
# Why a KULT module and not native KXCI commands
# ---------------------------------------------------------------------------
# KXCI has no DV/DI-equivalent for the 4225-PMU. The route is to run a KULT
# **user module** over KXCI: ``EX <library> <module>(<args>)`` executes it and
# returns the module's return value; ``GP <n>`` then reads each output
# parameter back by its 1-based position (``GN <name>`` does the same by name
# but needs the KULT grid name verbatim).
#
# KXCI's ``EX`` wants a value for EVERY module parameter positionally, output
# parameters included — they are passed as a placeholder ``0`` and their real
# values come back through ``GP``. So the call sends
# ``len(arg_order) + n_output_params`` args (16 + 4 for bridge_sot_pulse); a
# short count is the ``EX ERROR: invalid number of UTM parameters`` reply. The
# outputs are positions 17-20, so ``pulse_once`` reads ``GP 17`` … ``GP 20``.
#
# The defaults below target ``instruments/kult/bridge_sot_pulse.c`` — written
# for this measurement and tracked in this repo. Compile it on the 4200A in
# KULT (see ``instruments/kult/README.md``) and the defaults are correct as
# they stand. To drive a different module instead, repoint ``library`` /
# ``module`` and match ``arg_order`` / ``return_names`` to its signature: KXCI
# passes arguments POSITIONALLY, so a mismatched ``arg_order`` pulses with the
# wrong numbers rather than erroring.
#
# ``list_user_libraries(dev)`` (KXCI ``UL``) lists what is actually installed.
#
# The PMU forces VOLTAGE at the pin; the RPM gives current ranges + fast
# measure, not current forcing. So the swept axis is volts. On a 2-wire path
# the forced voltage also includes the cable/contact drop, which is why the
# module's measured CURRENT is the physically meaningful pulse axis — never
# back-compute current from an assumed channel resistance.
#
# The RPM pathway
# ---------------
# ``rpm_config()`` is an LPT call, reachable only from inside a KULT module —
# there is no Python-side equivalent, and no KXCI command for it. So the
# pathway is the module's job, and ``bridge_sot_pulse`` routes the RPM to the
# PMU on entry and back to the SMU on EVERY exit path including errors.
# Keithley's own ``PMU_1Chan_Sweep_Example`` does NOT route back: after it
# runs, an SMU wired through that RPM cannot reach the DUT at all and reads an
# open circuit with no error. If a DC read straight after a pulse comes back
# flat or zero, that is the first thing to suspect.

_MAX_PMU_CHANNELS = (1, 2)
_PMU_V_RANGES = (10.0, 40.0)
# Documented 4225-PMU timing floors, per voltage range: (width, rise/fall).
_PMU_TIMING_FLOOR_S = {10.0: (60e-9, 20e-9), 40.0: (60e-9, 100e-9)}


@dataclass
class PMUPulseConfig:
    """One KULT pulse-module invocation. Timing in seconds, levels in volts,
    ranges in volts/amps; fields are substituted into the ``EX`` argument list
    in ``arg_order`` (with ``amplitude_V`` overridden per pulse by
    ``pulse_once``).

    The defaults match ``instruments/kult/bridge_sot_pulse.c``. Point at a
    different module and ``arg_order``/``return_names`` must move with it."""
    library: str = "bridge_sot"             # KULT user-library name — confirm with `UL`
    module: str = "bridge_sot_pulse"        # instruments/kult/bridge_sot_pulse.c
    pmu_channel: int = 1                    # PMU/RPM channel wired to the device channel
    pmu_id: str = "PMU1"                    # PMU card name (lowest-numbered slot = PMU1)
    amplitude_V: float = 0.5                # forced pulse amplitude at the pin [V]
    base_V: float = 0.0                     # quiescent level between pulses [V]
    width_s: float = 100e-9                 # pulse top width (FWHM) [s]
    rise_s: float = 20e-9                   # leading-edge transition time [s]
    fall_s: float = 20e-9                   # trailing-edge transition time [s]
    delay_s: float = 0.0                    # dead time before the rise [s]
    period_s: float = 1e-3                  # full pulse period [s] (≥ delay+rise+width+fall)
    n_pulses: int = 1                       # pulses per burst, meaned into one spot mean
    sample_rate: float = 200e6              # PMU digitiser rate [S/s], 200e6/n
    meas_start_perc: float = 0.75           # spot-mean window start, fraction of the pulse top
    meas_stop_perc: float = 0.90            # spot-mean window stop, fraction of the pulse top
    dut_res_ohm: float = 1e3                # DUT resistance for the 50 Ω load-line correction
    v_range_V: float = 10.0                 # PMU voltage range — 10 or 40
    i_range_A: float = 0.01                 # current MEASURE range [A]; RPM 10 V range caps at 0.01
    v_limit_V: float = 5.0                  # software amplitude guard in pulse_once() [V]
    exec_timeout_s: float = 30.0            # VISA read timeout while the module runs
    arg_order: tuple = (
        "width_s", "rise_s", "fall_s", "period_s", "delay_s", "sample_rate",
        "meas_start_perc", "meas_stop_perc", "n_pulses", "dut_res_ohm",
        "v_range_V", "i_range_A", "amplitude_V", "base_V", "pmu_channel", "pmu_id")
    return_names: tuple = (                 # output params GN fetches, in module order
        "pulse_voltage_measured_V", "pulse_current_measured_A",
        "pulse_base_voltage_V", "pulse_base_current_A")
    n_output_params: int = 4               # module output-param count — passed as
                                           # placeholders in the EX call (KXCI wants
                                           # every param); ≥ len(return_names)


def _fmt_arg(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return f"{value:d}"
    if isinstance(value, str):
        return value
    return f"{float(value):.6E}"


def _parse_gn(reply: str) -> float:
    """One ``GP``/``GN`` output value → float. Takes the FIRST float-parseable
    token (a trailing status flag stays ignored); handles ``,``- or
    space-separated replies without reusing ``_parse_reading`` (GA-style lists
    put status last, not first)."""
    for token in reply.strip().replace(",", " ").replace(";", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    raise ValueError(f"no numeric value in GP reply {reply!r}")


def list_user_libraries(dev: _Keithley4200A_KXCI) -> str:
    """Raw reply to KXCI ``UL`` — the installed user-library list. Discovery
    helper: run this once to find the pulse module to point
    ``PMUPulseConfig.library``/``.module`` at. Some firmware returns several
    lines; read more from ``dev.adapter`` if this looks truncated."""
    return dev.query("UL")


def configure_pmu_pulse(dev: _Keithley4200A_KXCI, cfg: PMUPulseConfig) -> None:
    """Validate ``cfg`` and log the ``EX`` template. No device I/O — the KULT
    module invoked by :func:`pulse_once` is self-contained (KXCI User Mode)."""
    if not cfg.module:
        raise ValueError("PMUPulseConfig.module is empty — set it to a pulse "
                         "module from `list_user_libraries(dev)` (KXCI `UL`); "
                         "the default is 'bridge_sot_pulse' (instruments/kult/).")
    if cfg.pmu_channel not in _MAX_PMU_CHANNELS:
        raise ValueError(f"pmu_channel must be one of {_MAX_PMU_CHANNELS}, got {cfg.pmu_channel!r}")
    if cfg.v_range_V not in _PMU_V_RANGES:
        raise ValueError(f"v_range_V must be one of {_PMU_V_RANGES} V, got {cfg.v_range_V!r}")
    if cfg.width_s <= 0 or cfg.rise_s < 0 or cfg.fall_s < 0:
        raise ValueError("pulse width must be > 0 and rise/fall ≥ 0")

    # Documented 4225-PMU floors. Below these the module returns -824
    # (invalid pulse timing) on the bench — cheaper to catch it here.
    width_min, edge_min = _PMU_TIMING_FLOOR_S[cfg.v_range_V]
    if cfg.width_s < width_min:
        raise ValueError(f"width_s {cfg.width_s:g} s is below the {cfg.v_range_V:g} V range "
                         f"minimum of {width_min:g} s")
    if min(cfg.rise_s, cfg.fall_s) < edge_min:
        raise ValueError(f"rise_s/fall_s must be ≥ {edge_min:g} s on the {cfg.v_range_V:g} V "
                         f"range, got {cfg.rise_s:g}/{cfg.fall_s:g} s")

    if cfg.period_s < cfg.delay_s + cfg.width_s + cfg.rise_s + cfg.fall_s:
        raise ValueError("period_s must be ≥ delay_s + width_s + rise_s + fall_s")
    # PMU width is FWHM (50 % points), so the settled top is
    # width - 0.5*rise - 0.5*fall. If that is not positive the pulse never
    # reaches amplitude and there is nothing for the spot mean — the bench
    # returns -826.
    if cfg.width_s <= 0.5 * (cfg.rise_s + cfg.fall_s):
        raise ValueError(
            f"no flat pulse top: width_s ({cfg.width_s:g} s) must exceed "
            f"0.5*(rise_s+fall_s) = {0.5 * (cfg.rise_s + cfg.fall_s):g} s — "
            "shorten the edges or widen the pulse.")
    if not 0.0 <= cfg.meas_start_perc < cfg.meas_stop_perc <= 1.0:
        raise ValueError("need 0 ≤ meas_start_perc < meas_stop_perc ≤ 1, got "
                         f"{cfg.meas_start_perc!r} / {cfg.meas_stop_perc!r}")
    if cfg.n_pulses < 1:
        raise ValueError("n_pulses must be ≥ 1")
    if abs(cfg.amplitude_V) > cfg.v_limit_V:
        raise ValueError(f"amplitude_V {cfg.amplitude_V:g} exceeds v_limit_V ±{cfg.v_limit_V:g}")
    missing = [n for n in cfg.arg_order if not hasattr(cfg, n)]
    if missing:
        raise ValueError(f"arg_order names not on PMUPulseConfig: {missing}")
    if cfg.n_output_params < len(cfg.return_names):
        raise ValueError(f"n_output_params ({cfg.n_output_params}) < return_names "
                         f"({len(cfg.return_names)}) — can't GN-fetch more outputs "
                         "than the module has")
    template = ", ".join([f"<{n}>" for n in cfg.arg_order] + ["0"] * cfg.n_output_params)
    log.info("4200A PMU pulse via KULT: EX %s %s(%s)  [%s ch %d, %.4g V, %.3g s wide, "
             "%.4g V range, %.4g A measure range]",
             cfg.library, cfg.module or "<unset>", template, cfg.pmu_id,
             cfg.pmu_channel, cfg.amplitude_V, cfg.width_s, cfg.v_range_V, cfg.i_range_A)


def pulse_once(
    dev: _Keithley4200A_KXCI,
    cfg: PMUPulseConfig,
    amplitude_V: Optional[float] = None,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Deliver one pulse burst (``cfg.n_pulses`` pulses) at ``amplitude_V``
    (default ``cfg.amplitude_V``) by running the configured KULT module.

    Returns ``{"module_return": <EX reply>, **{name: float for name in
    cfg.return_names}}`` — a return name whose ``GP`` value can't be parsed is
    set to ``None`` rather than raising, so a partially-cooperating module
    still yields a pulse. Raises on the amplitude guard (load-bearing).
    """
    amp = cfg.amplitude_V if amplitude_V is None else amplitude_V
    if abs(amp) > cfg.v_limit_V:
        raise ValueError(f"Requested pulse amplitude {amp:g} V exceeds v_limit_V "
                         f"±{cfg.v_limit_V:g} V — refusing to pulse.")
    if stop_event is not None and stop_event.is_set():
        return {"module_return": None}

    in_args = [_fmt_arg(amp if name == "amplitude_V" else getattr(cfg, name))
               for name in cfg.arg_order]
    # KXCI EX wants a value for every module parameter — output params too.
    # They ride as placeholder 0 here and are read back by position below.
    args = ", ".join(in_args + ["0"] * cfg.n_output_params)
    cmd = f"EX {cfg.library} {cfg.module}({args})"

    prev_timeout = None
    try:
        prev_timeout = dev.adapter.connection.timeout
        dev.adapter.connection.timeout = cfg.exec_timeout_s * 1000
    except Exception:
        pass
    try:
        ex_reply = dev.query(cmd)
    finally:
        if prev_timeout is not None:
            try:
                dev.adapter.connection.timeout = prev_timeout
            except Exception:
                pass

    out: dict = {"module_return": ex_reply}
    # KXCI reads output params back one at a time. `GN <ParameterName>` needs the
    # exact KULT grid name; `GP <n>` takes the 1-based position. The outputs are
    # the last params, right after the 16 in arg_order, so GP by position is
    # name-independent. A reply that will not parse → None (diagnostic only).
    for i, name in enumerate(cfg.return_names):
        pos = len(cfg.arg_order) + 1 + i
        try:
            out[name] = _parse_gn(dev.query(f"GP {pos}"))
        except Exception:
            out[name] = None
    return out


def pulse_once_s(n_pulses: int, period_s: float) -> float:
    """Modelled wall time of one ``pulse_once()``: run_time.PMU_PULSE_S (module
    setup + EX + the GP read-backs + route-back) plus the pulse burst itself."""
    return PMU_PULSE_S + n_pulses * period_s
