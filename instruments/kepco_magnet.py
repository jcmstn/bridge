"""
Kepco BOP 20-50GL Bipolar Power Supply Driver
Built on pymeasure's Instrument architecture (SCPIMixin — this is a
genuinely SCPI/IEEE-488.2 instrument, unlike the LakeShore 475 or
MercuryiTC drivers in this package which use proprietary command sets).

Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-29

Interface: GPIB or RS-232C (9600 baud, 8N1)
Firmware: 3.05+

Usage example:
    from instruments.kepco_magnet import KepkoBOPGL

    psu = KepkoBOPGL("GPIB0::6::INSTR")

    psu.reset()
    psu.raise_range_limits_to_max()  # undo any stale CURR:LIM/VOLT:LIM from
                                     # a previous session -- *RST alone
                                     # will NOT do this (see PAR. 3.3.5)
    psu.mode = "current"
    psu.current = 5.0          # A
    psu.voltage_limit = 20.0   # V (protect limit)
    psu.enable_output()        # verifies the output actually turned on

    print(psu.measure_current)
    print(psu.measure_voltage)

    psu.output_enabled = False
    psu.close()

Usage example (shared controller, used by the DC measurement scripts):
    from instruments.kepco_magnet import MagnetConfig, connect_magnet, set_magnet_current, shutdown_magnet

    magnet_cfg = MagnetConfig(visa_resource="GPIB0::6::INSTR", current_limit_A=35.0)
    magnet = connect_magnet(magnet_cfg)
    set_magnet_current(magnet, magnet_cfg, 5.0)
    ...
    shutdown_magnet(magnet, magnet_cfg)
"""

import logging
import time
from dataclasses import dataclass

from pymeasure.instruments import Instrument, SCPIMixin
from pymeasure.instruments.validators import strict_range

log = logging.getLogger(__name__)


class KepkoBOPGL(SCPIMixin, Instrument):
    """
    Instrument driver for the Kepco BOP 20-50GL (and compatible BOP-GL 1kW series)
    bipolar power supply, optimised for inductive loads (e.g. superconducting magnets).

    Specifications:
        Voltage range : ±20 V DC
        Current range : ±50 A DC
        Interface     : GPIB (IEEE 488.2) or RS-232C
        Command set   : SCPI

    Attributes:
        MAX_VOLTAGE (float): Maximum programmable voltage (V).
        MAX_CURRENT (float): Maximum programmable current (A).
    """

    MAX_VOLTAGE: float = 20.0   # V
    MAX_CURRENT: float = 50.0   # A

    def __init__(self, adapter, name="Kepco BOP-GL Bipolar Power Supply", **kwargs):
        super().__init__(
            adapter,
            name,
            read_termination="\n",
            write_termination="\n",
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    #  IEEE 488.2 common commands                                          #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """
        Reset the instrument's output state (*RST).

        Note: *RST does NOT restore the CURR:LIM / VOLT:LIM setpoint
        ceilings to their factory (full-range) values if they were ever
        narrowed and saved with ``MEM:UPD LIM`` -- that requires a
        hardware Reset Power-up or the password-protected SYST:SEC:IMM
        command (manual PAR. 3.3.2.1 / 3.3.5). If the current or voltage
        setpoint refuses to climb past some fixed value regardless of
        ``current_limit``/``voltage_limit``, call
        :meth:`raise_range_limits_to_max` right after this.
        """
        super().reset()
        time.sleep(0.5)  # Allow unit to reinitialise

    def wait(self) -> None:
        """Block until all pending operations complete (*WAI)."""
        self.write("*WAI")

    def trigger(self) -> None:
        """Send a group execute trigger (*TRG)."""
        self.write("*TRG")

    identification = Instrument.measurement(
        "*IDN?",
        """ Get the instrument identification string (*IDN?). """,
        cast=str,
        maxsplit=0,
    )

    def check_set_errors(self) -> list:
        """
        Drain the error queue after every setpoint write and raise if the
        instrument rejected it.

        Per the BOP-GL manual (PAR. 3.3.5), a value rejected by a software
        limit (VOLT:LIM / CURR:LIM) is *silently disregarded* by the
        instrument -- no exception, no output change, just a queued SCPI
        error (-120 or -222) that would otherwise go unnoticed. Every
        setpoint-programming property below is declared with
        ``check_set_errors=True`` so this runs after each one, turning a
        stale/narrowed limit into a Python exception instead of a
        mysteriously "stuck" output.
        """
        errors = self.check_errors()
        if errors:
            raise RuntimeError(f"Kepco rejected the setpoint: {errors}")
        return errors

    # ------------------------------------------------------------------ #
    #  Output enable / disable                                             #
    # ------------------------------------------------------------------ #

    @property
    def output_enabled(self) -> bool:
        """
        Enable or disable the DC output.

        Returns True if the output is currently enabled.
        """
        return bool(int(self.ask("OUTP?")))

    @output_enabled.setter
    def output_enabled(self, enable: bool) -> None:
        self.write(f"OUTP {'ON' if enable else 'OFF'}")

    def enable_output(self, verify: bool = True) -> None:
        """
        Enable the output and confirm it actually turned on.

        Unlike the ``output_enabled`` setter (a bare ``OUTP ON`` write with no
        feedback), this reads back ``OUTP?`` and drains the error queue to
        catch a fault that would otherwise leave the output silently off.

        Note there is no SCPI command on the BOP-GL to remotely clear a
        latched protection trip -- and no "trip" for the ordinary case,
        either. ``voltage_limit``/``current_limit`` (VOLT:PROT/CURR:PROT)
        are non-latching compliance clamps: they just cap the output and
        release again on their own once the load condition changes (manual
        PAR. 3.3.4). What *does* latch is a hardware-level output-stage
        fault (overvoltage/overcurrent detection, heatsink over-temp, PFC
        fault -- see Table 1-2, "Output Stage Protection"), and the manual
        is explicit that the only recovery is cycling the front-panel POWER
        switch off and on; no digital command reaches it. If ``verify``
        below raises, that physical power-cycle is what's needed -- an
        earlier version of this driver sent a nonexistent ``OUTP:PROT:CLE``
        command here, which did nothing but leave a stale -100 "Command
        error" in the queue for the next checked write to trip over.

        Args:
            verify: Read back the output state and error queue afterward and
                raise RuntimeError if the output did not actually turn on.
        """
        self.output_enabled = True
        if verify:
            time.sleep(0.1)
            if not self.output_enabled:
                errors = self.check_errors()
                detail = f": {errors}" if errors else (
                    " (no error queued — this is a latched hardware-stage fault; "
                    "cycle the front-panel POWER switch off and on to clear it)"
                )
                raise RuntimeError(f"Kepco output failed to enable{detail}")

    # ------------------------------------------------------------------ #
    #  Operating mode (voltage / current)                                  #
    # ------------------------------------------------------------------ #

    @property
    def mode(self) -> str:
        """
        Set or get the operating mode.

        Values:
            ``"voltage"``  — constant-voltage (CV) mode
            ``"current"``  — constant-current (CC) mode
        """
        response = self.ask("FUNC:MODE?").strip().upper()
        if "VOLT" in response:
            return "voltage"
        elif "CURR" in response:
            return "current"
        return response

    @mode.setter
    def mode(self, value: str) -> None:
        value = value.lower()
        if value in ("voltage", "volt", "v"):
            self.write("FUNC:MODE VOLT")
        elif value in ("current", "curr", "i"):
            self.write("FUNC:MODE CURR")
        else:
            raise ValueError(f"Invalid mode '{value}'. Use 'voltage' or 'current'.")

    # ------------------------------------------------------------------ #
    #  Voltage programming and protection                                  #
    # ------------------------------------------------------------------ #

    voltage = Instrument.control(
        "VOLT?", "VOLT %.6f",
        """ Control the programmed voltage setpoint in volts.

        Valid range: –MAX_VOLTAGE to +MAX_VOLTAGE (±20 V). A setpoint the
        instrument rejects (e.g. above a stale, narrowed VOLT:LIM ceiling
        -- see :meth:`raise_range_limits_to_max`) raises RuntimeError
        instead of being silently disregarded. """,
        validator=strict_range,
        values=[-MAX_VOLTAGE, MAX_VOLTAGE],
        check_set_errors=True,
    )

    voltage_limit = Instrument.control(
        "VOLT:PROT:POS?", "VOLT:PROT %.6f",
        """ Control the voltage protection (OVP) limit in volts.

        When operating in current mode this is the compliance voltage limit.
        Valid range: 0 to MAX_VOLTAGE. """,
        validator=strict_range,
        values=[0, MAX_VOLTAGE],
        check_set_errors=True,
    )

    voltage_setpoint_max = Instrument.control(
        "VOLT:LIM:POS?", "VOLT:LIM %.6f",
        """ Control the software ceiling on the ``voltage`` setpoint itself
        (VOLT:LIM), in volts.

        This is a *different* register from ``voltage_limit`` (VOLT:PROT).
        ``voltage_limit`` clamps the compliance voltage while the unit
        regulates current; this one gates what values the ``voltage``
        setter is even allowed to accept. Per the manual (PAR. 3.3.5) it
        defaults to MAX_VOLTAGE, but if it was ever narrowed and saved
        with ``MEM:UPD LIM`` in a previous session, ``*RST`` (and hence
        :meth:`reset`) will NOT restore it -- it silently keeps rejecting
        any higher ``voltage`` setpoint. Call
        :meth:`raise_range_limits_to_max` at the start of a session if you
        suspect this. """,
        validator=strict_range,
        values=[0, MAX_VOLTAGE],
        check_set_errors=True,
    )

    # ------------------------------------------------------------------ #
    #  Current programming and protection                                  #
    # ------------------------------------------------------------------ #

    current = Instrument.control(
        "CURR?", "CURR %.6f",
        """ Control the programmed current setpoint in amperes.

        Valid range: –MAX_CURRENT to +MAX_CURRENT (±50 A). """,
        validator=strict_range,
        values=[-MAX_CURRENT, MAX_CURRENT],
        check_set_errors=True,
    )

    current_limit = Instrument.control(
        "CURR:PROT:POS?", "CURR:PROT %.6f",
        """ Control the current protection limit in amperes.

        When operating in voltage mode this is the compliance current limit.
        Valid range: 0 to MAX_CURRENT. """,
        validator=strict_range,
        values=[0, MAX_CURRENT],
        check_set_errors=True,
    )

    current_setpoint_max = Instrument.control(
        "CURR:LIM:POS?", "CURR:LIM %.6f",
        """ Control the software ceiling on the ``current`` setpoint itself
        (CURR:LIM), in amperes.

        This is a *different* register from ``current_limit`` (CURR:PROT).
        ``current_limit`` clamps the compliance current while the unit
        regulates voltage; this one gates what values the ``current``
        setter is even allowed to accept -- and it is the most common
        reason the output appears stuck at some value well below
        MAX_CURRENT no matter how high ``current_limit`` is raised.
        Per the manual (PAR. 3.3.5) it defaults to MAX_CURRENT, but if it
        was ever narrowed and saved with ``MEM:UPD LIM`` in a previous
        session, ``*RST`` (and hence :meth:`reset`) will NOT restore it --
        the instrument just disregards (SCPI error -120) any higher
        ``current`` setpoint. Call :meth:`raise_range_limits_to_max` at
        the start of a session if you suspect this is why the magnet
        current won't climb past a fixed ceiling. """,
        validator=strict_range,
        values=[0, MAX_CURRENT],
        check_set_errors=True,
    )

    def raise_range_limits_to_max(self, persist: bool = False) -> None:
        """
        Reset the CURR:LIM / VOLT:LIM setpoint ceilings back to the model's
        full rating (MAX_CURRENT / MAX_VOLTAGE).

        These software limits are independent of ``current_limit`` /
        ``voltage_limit`` (the OCP/OVP compliance clamps) and independent
        of ``*RST`` -- once narrowed and saved with ``MEM:UPD LIM``, they
        persist across power cycles and reset commands, silently capping
        every future setpoint (this is the classic "current won't go above
        20 A no matter what I set current_limit to" symptom). Call this
        once at the start of a session whenever you're not certain of the
        unit's history (e.g. it was previously used/calibrated for a
        lower-current test).

        Args:
            persist: If True, also send ``MEM:UPD LIM`` so the restored
                (full-range) limits become the new power-up default.
        """
        self.current_setpoint_max = self.MAX_CURRENT
        self.voltage_setpoint_max = self.MAX_VOLTAGE
        if persist:
            self.write("MEM:UPD LIM")

    # ------------------------------------------------------------------ #
    #  Measurements                                                        #
    # ------------------------------------------------------------------ #

    measure_voltage = Instrument.measurement(
        "MEAS:VOLT?",
        """ Measure and return the actual output voltage in volts. """,
    )

    measure_current = Instrument.measurement(
        "MEAS:CURR?",
        """ Measure and return the actual output current in amperes. """,
    )

    # ------------------------------------------------------------------ #
    #  Memory: save and recall settings                                    #
    # ------------------------------------------------------------------ #

    def select_memory(self, location: int) -> None:
        """
        Select a memory register (1–72 for BOP-GL).

        Args:
            location: Memory slot number.
        """
        self.write(f"MEM:NSEL {location}")

    def save_settings(self, location: int) -> None:
        """
        Save current output settings to a memory register.

        Args:
            location: Memory slot number (1–72).
        """
        self.select_memory(location)
        self.write("MEM:SAVE")

    def recall_settings(self, location: int) -> None:
        """
        Recall output settings from a memory register.

        Args:
            location: Memory slot number (1–72).
        """
        self.select_memory(location)
        self.write("MEM:RCL")

    def update_powerup_memory(self) -> None:
        """
        Save current configuration as power-up default (MEM:UPD).

        Warning: tag the unit with the custom power-up config per manual.
        """
        self.write("MEM:UPD")

    # ------------------------------------------------------------------ #
    #  Waveform / LIST subsystem (basic)                                   #
    # ------------------------------------------------------------------ #

    def abort_waveform(self) -> None:
        """Abort any running waveform sequence (ABOR)."""
        self.write("ABOR")

    def initiate_waveform(self) -> None:
        """Arm the trigger system to accept the next waveform trigger (INIT)."""
        self.write("INIT")

    # ------------------------------------------------------------------ #
    #  Convenience methods for magnet control                              #
    # ------------------------------------------------------------------ #

    def ramp_current(self, target: float, step: float = 0.1,
                     delay: float = 0.05) -> None:
        """
        Ramp the current setpoint from the current value to *target* in
        incremental steps.  Useful for inductive / magnet loads.

        Args:
            target: Target current in amperes.
            step  : Step size in amperes (default 0.1 A).
            delay : Wait time between steps in seconds (default 50 ms).
        """
        start = self.current
        n_steps = max(1, int(abs(target - start) / abs(step)))
        values = [start + (target - start) * i / n_steps
                  for i in range(1, n_steps + 1)]
        for v in values:
            self.current = v
            time.sleep(delay)

    def zero_output(self, ramp: bool = True, step: float = 0.1,
                    delay: float = 0.05) -> None:
        """
        Bring the output setpoint to zero, then disable the output.

        For inductive loads (magnets) a controlled ramp is strongly
        recommended to avoid quench.

        Args:
            ramp : If True, ramp to zero instead of stepping immediately.
            step : Ramp step size in amperes (default 0.1 A).
            delay: Delay between ramp steps in seconds.
        """
        if ramp:
            self.ramp_current(0.0, step=step, delay=delay)
        else:
            if self.mode == "current":
                self.current = 0.0
            else:
                self.voltage = 0.0
        self.output_enabled = False

    # ------------------------------------------------------------------ #
    #  Connection lifecycle                                                #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the underlying VISA session."""
        self.adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Shared controller  ── used directly by the DC measurement scripts ───────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MagnetConfig:
    """
    Kepco BOP-GL bipolar power supply, used as a current source for an
    electromagnet. Shared by every DC program that sweeps or parks a field.
    """
    visa_resource:        str   = "GPIB0::6::INSTR"
    current_limit_A:      float = 50.0    # Software current limit  [A]
    voltage_compliance_V: float = 20.0    # CC-mode compliance / OVP limit  [V]
    ramp_step_A:          float = 0.1     # Ramp step size  [A]
    ramp_delay_s:         float = 0.05    # Delay between ramp steps  [s]


# ─────────────────────────────────────────────────────────────────────────────
# Settle detection  ── used by set_magnet_current() below ─────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ramp_current() only walks the *setpoint*; it returns before the supply's
# analog output has slewed into the inductive magnet, and long before the
# iron-core field has stopped drifting (eddy currents + magnetic after-effect).
# The BOP-GL has no SCPI "output settled / ramp complete" query -- *OPC?/*WAI
# only sync the command parser, and STAT:OPER:COND has no SETTling bit (its
# bits are list running/complete, sample complete, CC/CV mode, transient
# armed/complete, waiting-for-trigger). So set_magnet_current() closes the gap
# itself: poll MEAS:CURR? until the output reaches the setpoint, then (if a
# gaussmeter is supplied) poll the field until it stops drifting.
#
# These constants are deliberately NOT surfaced in the TUI / web forms -- edit
# them here if a particular rig needs it. The one exception is the field
# tolerance, which set_magnet_current() takes as an argument so the TUI/web
# "Advanced" section can override FIELD_SETTLE_TOLERANCE_MT per system.
CURRENT_SETTLE_REL        = 0.05    # readback band as a fraction of |setpoint|
CURRENT_SETTLE_TIMEOUT_S  = 10.0    # give up waiting on MEAS:CURR? after this  [s]
FIELD_SETTLE_TOLERANCE_MT = 0.02    # default window span (max-min) that counts as settled  [mT]
FIELD_SETTLE_WINDOW_N     = 4       # readings that must all fall within tolerance
FIELD_SETTLE_POLL_S       = 0.25    # delay between field readings  [s]  (→ ~1 s look-back)
FIELD_SETTLE_TIMEOUT_S    = 30.0    # give up waiting on the field after this  [s]

_UNIT_TO_MT = {"T": 1e3, "G": 1e-1}   # GaussmeterConfig.unit → mT


def _window_settled(readings: list, tol: float) -> bool:
    """True once the last FIELD_SETTLE_WINDOW_N readings span <= tol (max - min)."""
    if len(readings) < FIELD_SETTLE_WINDOW_N:
        return False
    window = readings[-FIELD_SETTLE_WINDOW_N:]
    return (max(window) - min(window)) <= tol


def connect_magnet(cfg: MagnetConfig) -> "KepkoBOPGL":
    """
    Open a VISA session to the Kepco BOP-GL and arm it as a current source.

    Clears stale CURR:LIM/VOLT:LIM setpoint ceilings back to the supply's
    full rating (independent of, and not restored by, *RST), then puts it
    in constant-current mode with the configured software compliance
    voltage and current limit.
    """
    psu = KepkoBOPGL(cfg.visa_resource, timeout=5000)

    psu.reset()
    psu.clear()
    psu.raise_range_limits_to_max()
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


def set_magnet_current(
    psu: "KepkoBOPGL",
    cfg: MagnetConfig,
    current_A: float,
    gaussmeter=None,
    gauss_cfg=None,
    field_settle_tolerance_mT: "float | None" = None,
    stop_event=None,
) -> dict:
    """
    Ramp the magnet current to `current_A`, then wait until it has actually
    settled before returning.

    Two waits, both bounded by a timeout (a slow rig degrades to the old
    behaviour with a logged warning -- it never hangs the measurement):

      1. Poll ``MEAS:CURR?`` until the supply output is within
         ``max(CURRENT_SETTLE_REL * |current_A|, cfg.ramp_step_A)`` of the
         setpoint. Always runs.

      2. If `gaussmeter` / `gauss_cfg` are given, poll the field until the
         last ``FIELD_SETTLE_WINDOW_N`` readings span no more than
         `field_settle_tolerance_mT` (defaults to
         ``FIELD_SETTLE_TOLERANCE_MT``). This catches the eddy-current /
         after-effect drift the current readback can't see.

    The caller's own post-field settling dwell (``settling_time_s``) still
    belongs *after* this returns -- that one is the physics-equilibration
    wait once the field has reached its final value.

    `stop_event`, if given, is checked between polls and breaks either wait.

    Returns a provenance dict the measurement loops fold into each CSV row::

        {"current_settled": bool,
         "field_settled":    bool | None,   # None → no gaussmeter passed
         "settle_elapsed_s": float,
         "i_measured_A":     float}
    """
    if abs(current_A) > cfg.current_limit_A:
        raise ValueError(
            f"Requested current {current_A:.3f} A exceeds configured "
            f"limit ±{cfg.current_limit_A:.3f} A"
        )
    psu.ramp_current(current_A, step=cfg.ramp_step_A, delay=cfg.ramp_delay_s)

    t0 = time.monotonic()

    # ── 1. Wait for the supply output to reach the setpoint ────────────────
    # Coarse gate only: ramp_current() has already blocked through the whole
    # stepped setpoint walk, so MEAS:CURR? is usually within band on the
    # first read. It catches a supply that fell behind (compliance-limited)
    # during the ramp; the field window below is what actually decides the
    # field has stopped moving.
    band = max(CURRENT_SETTLE_REL * abs(current_A), cfg.ramp_step_A)
    i_meas = float(psu.measure_current)
    while abs(i_meas - current_A) > band:
        if stop_event is not None and stop_event.is_set():
            break
        if time.monotonic() - t0 > CURRENT_SETTLE_TIMEOUT_S:
            log.warning(
                "Magnet current did not settle: I_meas=%.4f A vs set=%.4f A "
                "(band ±%.4f A) after %.1f s", i_meas, current_A, band,
                CURRENT_SETTLE_TIMEOUT_S)
            break
        time.sleep(cfg.ramp_delay_s)
        i_meas = float(psu.measure_current)
    current_settled = abs(i_meas - current_A) <= band

    # ── 2. Wait for the field to stop drifting (gaussmeter) ────────────────
    field_settled = None
    if gaussmeter is not None and gauss_cfg is not None:
        tol = (field_settle_tolerance_mT if field_settle_tolerance_mT is not None
               else FIELD_SETTLE_TOLERANCE_MT)
        to_mT = _UNIT_TO_MT[gauss_cfg.unit]
        readings: list = []
        tf0 = time.monotonic()
        while True:
            readings.append(float(gaussmeter.field) * to_mT)
            if _window_settled(readings, tol):
                field_settled = True
                break
            if stop_event is not None and stop_event.is_set():
                field_settled = _window_settled(readings, tol)
                break
            if time.monotonic() - tf0 > FIELD_SETTLE_TIMEOUT_S:
                w = readings[-FIELD_SETTLE_WINDOW_N:]
                span = (max(w) - min(w)) if len(w) >= FIELD_SETTLE_WINDOW_N else float("nan")
                log.warning(
                    "Field did not settle within %.1f s: window span %.4f mT "
                    "> tol %.4f mT", FIELD_SETTLE_TIMEOUT_S, span, tol)
                field_settled = False
                break
            time.sleep(FIELD_SETTLE_POLL_S)

    return {
        "current_settled":  current_settled,
        "field_settled":    field_settled,
        "settle_elapsed_s": round(time.monotonic() - t0, 3),
        "i_measured_A":     i_meas,
    }


def shutdown_magnet(psu: "KepkoBOPGL", cfg: MagnetConfig) -> None:
    """Ramp the magnet current safely to zero, disable the output, and close the VISA session."""
    log.info("Ramping magnet to zero and disabling output ...")
    psu.zero_output(ramp=True, step=cfg.ramp_step_A, delay=cfg.ramp_delay_s)
    psu.close()


if __name__ == "__main__":
    with KepkoBOPGL("GPIB0::6::INSTR") as psu:
        print(psu.identification)
        psu.reset()
        psu.raise_range_limits_to_max()  # clear any stale CURR:LIM/VOLT:LIM
        psu.mode = "current"
        psu.voltage_limit = 18.0   # compliance voltage
        psu.current = 0.0
        psu.enable_output()

        psu.ramp_current(10.0, step=0.2, delay=0.1)   # ramp up to 10 A
        print(f"V={psu.measure_voltage:.4f} V  I={psu.measure_current:.4f} A")

        psu.zero_output(ramp=True)   # safe ramp-down for inductive load
