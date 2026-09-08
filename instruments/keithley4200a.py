"""
Keithley 4200A-SCS Semiconductor Characterization System — KXCI SMU driver
==========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-07

pymeasure ships no 4200A driver. This module is a hand-written wrapper over
**KXCI** (Keithley External Control Interface) — the command server the 4200A
runs so an outside PC can drive it over GPIB or LAN, the same role KXCI plays
for every remote 4200A script.

Scope of THIS module:
  * the two 4200A **SMU** cards, as a general force/measure SMU
    (docs/architecture.md §4) — ``SMUChannelConfig`` + ``set_source_level`` /
    ``read_measurement`` / ``acquire_measurement`` / ``acquire_reversal_averaged``.
  * the **PMU** (4225-PMU) + its two RPMs, as a fire-one-pulse primitive —
    ``PMUPulseConfig`` + ``configure_pmu_pulse`` / ``pulse_once``. See the
    "PMU" section lower in this file for the KXCI mechanism and why the KULT
    module name/signature is *your* config, not a hard-coded default.

The PMU here does not measure the switched state — that is a separate
6221 + 2182 delayed R_xy read in ``sot/sot_pulsed_switching.py``. The PMU's
only job is to deliver the write pulse.

Transport
---------
KXCI speaks over either:
  * GPIB   — ``visa_resource="GPIB0::17::INSTR"``
  * LAN    — ``visa_resource="TCPIP0::192.168.0.10::1225::SOCKET"`` (1225 = KXCI
             default port; set it in Clarius → Tools → KXCI Configuration)
Both use ``\\n`` line terminators. ``*IDN?`` (or the KXCI ``ID`` query) over the
chosen transport is the whole smoke test.

Command set used (KXCI **User Mode**, immediate execution):
  ``US``                             select User Mode
  ``BC`` / ``DR0``                   clear buffer / no data-ready SRQ
  ``IT1|IT2|IT3``                    integration time: fast | normal | quiet
  ``DV<ch>, <rng>, <V>, <Icmpl>``    force voltage on SMU <ch>
  ``DI<ch>, <rng>, <I>, <Vcmpl>``    force current on SMU <ch>
  ``TV<ch>`` / ``TI<ch>``            measure V / I on SMU <ch>  (returns ASCII)
``<rng>`` ``0`` = autorange (the default here). A non-zero range value is
passed through verbatim — the numeric range codes are model-specific, so set
one only if you know your card's codes.

VALIDATE AGAINST THE ACTUAL BOX before trusting a run:
  * ``ID`` / ``*IDN?`` round-trips over your transport.
  * one ``DV1,0,0,1e-3`` then ``TV1`` returns a plausible number.
  * Confirm the KXCI/Clarius version — older firmware differs on ``IT`` vs
    ``SP`` for speed and on whether ``*IDN?`` is answered at all.

4-wire (remote/Kelvin) sense
----------------------------
KXCI User Mode has **no** RSENSE toggle. 4-wire is set in Clarius/KCON and by
how the probes are wired. ``SMUChannelConfig.four_wire`` is recorded as run
provenance only — it does not configure the instrument.
# ponytail: KXCI User Mode can't assert sense mode. Upgrade path: KXCI System
# Mode / LPT `setmode(KI_SENSE, ...)` if a run must guarantee it in software.

Failure policy: an SMU sourcing into a device is **load-bearing** —
``connect_4200a`` and ``set_source_level`` raise on any failure; callers do
not pass ``None`` for the handle. ``shutdown_4200a`` is a plain call with no
internal try/except — wrap it in ``dc.dc_sweep_utils.safe_shutdown``.

Usage example (4-probe R: SMU1 forces current, SMU2 is a 0-A voltmeter):
    from instruments.keithley4200a import (
        Keithley4200AConfig, SMUChannelConfig, connect_4200a,
        set_source_level, acquire_reversal_averaged, shutdown_4200a)

    k = connect_4200a(Keithley4200AConfig(visa_resource="GPIB0::17::INSTR"))
    src   = SMUChannelConfig(channel=1, source_function="current",
                             compliance_voltage_V=2.0, source_limit_A=5e-3)
    hallv = SMUChannelConfig(channel=2, source_function="current",  # forces 0 A
                             compliance_voltage_V=2.0, source_limit_A=1e-9)
    try:
        set_source_level(k, hallv, 0.0)                 # park SMU2 as a voltmeter
        rv = acquire_reversal_averaged(k, src, hallv, 100e-6, n_reversals=10)
        R  = rv["mean"] / 100e-6
    finally:
        shutdown_4200a(k)
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymeasure.instruments import Instrument

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_SMU_FUNCS = ("voltage", "current")
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


def _parse_reading(raw: str) -> float:
    """KXCI ``TV``/``TI`` reply → float. Replies are ASCII and may carry a
    leading status letter (``N`` normal, ``C`` in-compliance, …), space-
    separated or not: ``"N +1.2345E-03"`` or ``"1.2345E-03"``."""
    s = raw.strip().replace(",", " ")
    if not s:
        raise ValueError("empty KXCI reading")
    token = s.split()[-1]
    try:
        return float(token)
    except ValueError:
        # No-separator form ("NCV1.23E-3"): drop leading status letters. Safe
        # only because a float never starts with one of these — don't add 'E'.
        return float(token.lstrip("NCXVIT"))


def _range_code(rng: Optional[float]) -> str:
    """``None`` → ``"0"`` (autorange). A given value is passed through verbatim
    — 4200A range codes are card-specific, so only set one deliberately."""
    return "0" if rng is None else f"{rng:g}"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Keithley4200AConfig:
    """Connection + global acquisition settings for the 4200A KXCI session."""
    visa_resource: str = "GPIB0::17::INSTR"   # or "TCPIP0::<ip>::1225::SOCKET"
    timeout_s: float   = 30.0                 # VISA I/O timeout (long: SMU measure + KXCI)
    integration: str   = "normal"             # "fast" | "normal" | "quiet"  → IT1/IT2/IT3


@dataclass
class SMUChannelConfig:
    """One 4200A SMU card as a force/measure channel. Mirrors
    ``instruments.keithley2450.SMUConfig`` so call sites read the same.

    ``source_range`` in the unit of the source function; ``None`` → autorange
    (recommended). The measurement always autoranges (KXCI ``IT`` speed +
    ``TV``/``TI`` — no per-read range knob is wired here)."""
    channel: int              = 1          # SMU card number (1 or 2 on this system)
    source_function: str      = "current"  # "voltage" | "current" — what this SMU drives
    compliance_current_A: float = 1e-3     # limit while sourcing voltage [A]
    compliance_voltage_V: float = 10.0     # limit while sourcing current [V]
    source_range: Optional[float]  = None  # None → autorange the source
    four_wire: bool           = True       # PROVENANCE ONLY over KXCI (see module docstring)
    source_limit_V: float     = 21.0       # set_source_level() refuses |V| beyond this
    source_limit_A: float     = 10e-3      # set_source_level() refuses |I| beyond this


def _resolved_sense(cfg: SMUChannelConfig) -> str:
    """The quantity this channel *measures*: the complement of what it forces
    (force I → read V, force V → read I). Same rule as the 2400/2450 wrappers."""
    return "voltage" if cfg.source_function == "current" else "current"


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


def configure_smu(dev: _Keithley4200A_KXCI, cfg: SMUChannelConfig) -> None:
    """Validate ``cfg`` and log the channel's role. KXCI User Mode has no
    per-channel pre-configuration step — ranges + compliance travel with each
    ``DV``/``DI`` in :func:`set_source_level` — so this only guards the config
    and makes the call site symmetric with the other SMU wrappers."""
    if cfg.source_function not in _SMU_FUNCS:
        raise ValueError(f"source_function must be one of {_SMU_FUNCS}, got {cfg.source_function!r}")
    if cfg.channel not in (1, 2):
        raise ValueError(f"channel must be 1 or 2 on this system, got {cfg.channel!r}")
    log.info("4200A SMU%d: force %s, measure %s, %s (compliance %s)",
             cfg.channel, cfg.source_function, _resolved_sense(cfg),
             "4-wire" if cfg.four_wire else "2-wire",
             f"{cfg.compliance_current_A:g} A" if cfg.source_function == "voltage"
             else f"{cfg.compliance_voltage_V:g} V")


def shutdown_4200a(dev: _Keithley4200A_KXCI, channels=(1, 2)) -> None:
    """Force 0 A on every used SMU (the safe state for a DUT) and close the
    session. Full 4-parameter ``DI`` — 0 A, autorange, 10 V compliance
    headroom — so teardown matches every other force command's arity. Plain
    calls, no try/except — the caller wraps this in ``safe_shutdown``."""
    for ch in channels:
        dev.command(f"DI{ch}, 0, 0.000000E+00, 1.000000E+01")
    dev.close()
    log.info("Keithley 4200A-SCS SMUs zeroed, session closed")


# ─────────────────────────────────────────────────────────────────────────────
# force / measure
# ─────────────────────────────────────────────────────────────────────────────

def set_source_level(dev: _Keithley4200A_KXCI, cfg: SMUChannelConfig, level: float) -> None:
    """Force ``level`` (V or A per ``cfg.source_function``) on ``cfg.channel``,
    refusing to exceed the configured software limit — mirrors
    ``keithley2450.set_source_level``."""
    if cfg.source_function == "voltage":
        if abs(level) > cfg.source_limit_V:
            raise ValueError(f"Requested source voltage {level:.4g} V exceeds "
                             f"source_limit_V ±{cfg.source_limit_V:.4g} V — refusing.")
        dev.command(f"DV{cfg.channel}, {_range_code(cfg.source_range)}, "
                    f"{level:.6E}, {cfg.compliance_current_A:.6E}")
    else:
        if abs(level) > cfg.source_limit_A:
            raise ValueError(f"Requested source current {level:.4g} A exceeds "
                             f"source_limit_A ±{cfg.source_limit_A:.4g} A — refusing.")
        dev.command(f"DI{cfg.channel}, {_range_code(cfg.source_range)}, "
                    f"{level:.6E}, {cfg.compliance_voltage_V:.6E}")


def read_measurement(dev: _Keithley4200A_KXCI, cfg: SMUChannelConfig) -> float:
    """One fresh reading of this channel's *sense* quantity, canonical units
    (V or A). For a current-forcing SMU that is its terminal voltage — i.e. a
    compliance / Joule-heating monitor."""
    cmd = "TV" if _resolved_sense(cfg) == "voltage" else "TI"
    return _parse_reading(dev.query(f"{cmd}{cfg.channel}"))


def acquire_measurement(
    dev: _Keithley4200A_KXCI,
    cfg: SMUChannelConfig,
    n: int,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Average ``n`` fresh readings of ``cfg``'s sense quantity. Returns
    ``{"mean", "sem"}`` with ``sem`` = sample stdev / sqrt(n) (``nan`` for
    n == 1) — same shape as ``keithley2450.acquire_measurement`` and
    ``keithley2182.acquire_averaged_voltage``. ``stop_event`` is checked
    between samples so a UI abort can cut a long average short."""
    sense = _resolved_sense(cfg)
    cmd = "TV" if sense == "voltage" else "TI"
    samples = np.empty(n)
    n_used = 0
    for i in range(n):
        samples[i] = _parse_reading(dev.query(f"{cmd}{cfg.channel}"))
        n_used = i + 1
        if stop_event is not None and stop_event.is_set():
            break
    used = samples[:n_used]
    sem = float(np.std(used, ddof=1) / np.sqrt(n_used)) if n_used >= 2 else float("nan")
    return {"mean": float(np.mean(used)), "sem": sem}


def acquire_reversal_averaged(
    dev: _Keithley4200A_KXCI,
    src_cfg: SMUChannelConfig,
    sense_cfg: SMUChannelConfig,
    level: float,
    n_reversals: int,
    stop_event: Optional[threading.Event] = None,
    source_delay_s: float = 0.0,
) -> dict:
    """Reverse the current forced by ``src_cfg`` (+level / -level) ``n_reversals``
    times and read the voltage on ``sense_cfg`` (a separate SMU parked at
    force-0-A) after each flip, decomposing into odd and even parts in the
    current.

    Same return contract as
    ``instruments.keithley6221.acquire_reversal_averaged_voltage`` — that dict
    shape (``mean``/``sem`` = V_odd, ``even_mean``/``even_sem`` = V_even,
    ``n_reversals``) is what the recorded ``*_even_V`` columns and
    docs/current-reversal.md are keyed to. Different plumbing (two SMUs instead
    of a 6221 + 2182), identical contract.

    Leaves ``src_cfg`` forcing +level on return. If ``stop_event`` fires
    partway, returns the mean/sem of whatever pairs were collected.
    """
    if src_cfg.source_function != "current":
        raise ValueError("acquire_reversal_averaged: src_cfg must force current")
    odd = np.empty(n_reversals)
    even = np.empty(n_reversals)
    n_used = 0
    for i in range(n_reversals):
        set_source_level(dev, src_cfg, level)
        if source_delay_s > 0:
            time.sleep(source_delay_s)
        v_plus = _parse_reading(dev.query(f"TV{sense_cfg.channel}"))

        set_source_level(dev, src_cfg, -level)
        if source_delay_s > 0:
            time.sleep(source_delay_s)
        v_minus = _parse_reading(dev.query(f"TV{sense_cfg.channel}"))

        odd[i] = (v_plus - v_minus) / 2.0
        even[i] = (v_plus + v_minus) / 2.0
        n_used = i + 1
        if stop_event is not None and stop_event.is_set():
            break

    set_source_level(dev, src_cfg, level)

    used_odd, used_even = odd[:n_used], even[:n_used]
    sem_odd = float(np.std(used_odd, ddof=1) / np.sqrt(n_used)) if n_used >= 2 else float("nan")
    sem_even = float(np.std(used_even, ddof=1) / np.sqrt(n_used)) if n_used >= 2 else float("nan")
    return {
        "mean": float(np.mean(used_odd)),
        "sem": sem_odd,
        "even_mean": float(np.mean(used_even)),
        "even_sem": sem_even,
        "n_reversals": n_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PMU  ── fire one current-forcing... no: one VOLTAGE pulse, via a KULT module
# ─────────────────────────────────────────────────────────────────────────────
# Why a KULT module and not native KXCI commands
# ---------------------------------------------------------------------------
# KXCI has no DV/DI-equivalent for the 4225-PMU. The portable route is to run
# a KULT **user module** over KXCI:  ``EX <library> <module>(<args>)`` executes
# it and returns the module's return value; ``GN`` then fetches each output
# parameter in order. WHICH module exists, and its argument order, is specific
# to your 4200A install — so ``PMUPulseConfig.library`` / ``.module`` /
# ``.arg_order`` are YOUR config, and there is deliberately no working default
# module name (a wrong guess would "run" and do nothing, or the wrong thing).
#
# Discover what you have:  ``list_user_libraries(dev)`` sends ``UL``. Point
# ``library``/``module`` at a pulse module (Keithley ships pulse-IV examples
# such as the ``pmu-dut-examples`` library), set ``arg_order`` to match its
# signature, and set ``return_names`` to the output parameters you want back
# (leave empty if the module returns none — then the measured pulse V/I simply
# stay blank in the data, which is fine).
#
# The PMU forces VOLTAGE at the pin; the RPM gives current ranges + fast
# measure, not current forcing. So the swept axis is volts. Record the
# module's spot-mean V/I if it returns them; never back-compute current from
# an assumed channel resistance.

_MAX_PMU_CHANNELS = (1, 2)


@dataclass
class PMUPulseConfig:
    """One KULT pulse-module invocation. Every timing field is in SI seconds /
    volts / amps; they are substituted into the ``EX`` argument list in
    ``arg_order`` (with ``amplitude_V`` overridden per pulse by
    ``pulse_once``). Reorder / trim ``arg_order`` + ``return_names`` to match
    your installed module — see the section comment above."""
    library: str = "pmu-dut-examples"      # KULT user-library name — CONFIRM with `UL`
    module: str = ""                        # KULT module name — REQUIRED, no safe default
    pmu_channel: int = 1                    # PMU/RPM channel wired to the device channel
    amplitude_V: float = 0.5               # forced pulse amplitude at the pin [V]
    base_V: float = 0.0                     # quiescent level between pulses [V]
    width_s: float = 100e-9               # pulse top width [s]
    rise_s: float = 20e-9                 # leading-edge transition time [s]
    fall_s: float = 20e-9                 # trailing-edge transition time [s]
    period_s: float = 1e-3               # full pulse period [s] (≥ width+rise+fall)
    n_pulses: int = 1                      # pulses delivered per pulse_once() call
    i_range_A: float = 0.2               # PMU/RPM current measure range [A]
    v_limit_V: float = 5.0              # PMU voltage limit / pulse_once() amplitude guard [V]
    i_limit_A: float = 0.2             # PMU current limit [A]
    exec_timeout_s: float = 30.0        # VISA read timeout while the module runs
    arg_order: tuple = (
        "pmu_channel", "amplitude_V", "base_V", "width_s", "rise_s", "fall_s",
        "period_s", "n_pulses", "i_range_A", "v_limit_V", "i_limit_A")
    return_names: tuple = ()             # output params GN fetches, in module order,
                                        # e.g. ("pulse_voltage_measured_V", "pulse_current_measured_A")


def _fmt_arg(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return f"{value:d}"
    if isinstance(value, str):
        return value
    return f"{float(value):.6E}"


def _parse_gn(reply: str) -> float:
    """One ``GN`` output value → float. Takes the FIRST float-parseable token
    (a trailing status flag stays ignored); handles ``,``- or space-separated
    replies without reusing ``_parse_reading`` (GA-style lists put status
    last, not first)."""
    for token in reply.strip().replace(",", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    raise ValueError(f"no numeric value in GN reply {reply!r}")


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
                         "module from `list_user_libraries(dev)` (KXCI `UL`).")
    if cfg.pmu_channel not in _MAX_PMU_CHANNELS:
        raise ValueError(f"pmu_channel must be one of {_MAX_PMU_CHANNELS}, got {cfg.pmu_channel!r}")
    if cfg.width_s <= 0 or cfg.rise_s < 0 or cfg.fall_s < 0:
        raise ValueError("pulse width must be > 0 and rise/fall ≥ 0")
    if cfg.period_s < cfg.width_s + cfg.rise_s + cfg.fall_s:
        raise ValueError("period_s must be ≥ width_s + rise_s + fall_s")
    if cfg.n_pulses < 1:
        raise ValueError("n_pulses must be ≥ 1")
    if abs(cfg.amplitude_V) > cfg.v_limit_V:
        raise ValueError(f"amplitude_V {cfg.amplitude_V:g} exceeds v_limit_V ±{cfg.v_limit_V:g}")
    missing = [n for n in cfg.arg_order if not hasattr(cfg, n)]
    if missing:
        raise ValueError(f"arg_order names not on PMUPulseConfig: {missing}")
    template = ", ".join(f"<{n}>" for n in cfg.arg_order)
    log.info("4200A PMU pulse via KULT: EX %s %s(%s)  [ch %d, %.4g V, %.3g s wide]",
             cfg.library, cfg.module or "<unset>", template,
             cfg.pmu_channel, cfg.amplitude_V, cfg.width_s)


def pulse_once(
    dev: _Keithley4200A_KXCI,
    cfg: PMUPulseConfig,
    amplitude_V: Optional[float] = None,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Deliver one pulse burst (``cfg.n_pulses`` pulses) at ``amplitude_V``
    (default ``cfg.amplitude_V``) by running the configured KULT module.

    Returns ``{"module_return": <EX reply>, **{name: float for name in
    cfg.return_names}}`` — a return name whose ``GN`` value can't be parsed is
    set to ``None`` rather than raising, so a partially-cooperating module
    still yields a pulse. Raises on the amplitude guard (load-bearing).
    """
    amp = cfg.amplitude_V if amplitude_V is None else amplitude_V
    if abs(amp) > cfg.v_limit_V:
        raise ValueError(f"Requested pulse amplitude {amp:g} V exceeds v_limit_V "
                         f"±{cfg.v_limit_V:g} V — refusing to pulse.")
    if stop_event is not None and stop_event.is_set():
        return {"module_return": None}

    args = ", ".join(_fmt_arg(amp if name == "amplitude_V" else getattr(cfg, name))
                     for name in cfg.arg_order)
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
    for name in cfg.return_names:
        try:
            out[name] = _parse_gn(dev.query("GN"))
        except Exception:
            out[name] = None
    return out
