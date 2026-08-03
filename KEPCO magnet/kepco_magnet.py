"""
Kepco BOP 20-50GL Bipolar Power Supply Driver
Compatible with pymeasure's Instrument architecture.

Interface: GPIB or RS-232C (9600 baud, 8N1)
Firmware: 3.05+

Usage example:
    from kepco_bop_gl import KepkoBOPGL
    import pyvisa

    rm = pyvisa.ResourceManager()
    psu = KepkoBOPGL(rm.open_resource("GPIB0::6::INSTR"))

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
"""

import time
import pyvisa


class KepkoBOPGL:
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

    def __init__(self, resource, read_termination: str = "\n",
                 write_termination: str = "\n", timeout: int = 5000):
        """
        Initialise the driver.

        Args:
            resource: A pyvisa Resource object (already opened).
            read_termination: Line terminator for read operations.
            write_termination: Line terminator for write operations.
            timeout: Communication timeout in milliseconds.
        """
        self._inst = resource
        self._inst.read_termination = read_termination
        self._inst.write_termination = write_termination
        self._inst.timeout = timeout

    # ------------------------------------------------------------------ #
    #  Low-level communication helpers                                     #
    # ------------------------------------------------------------------ #

    def write(self, command: str) -> None:
        """Send a command string to the instrument."""
        self._inst.write(command)

    def query(self, command: str) -> str:
        """Send a command and return the stripped response string."""
        return self._inst.query(command).strip()

    def query_float(self, command: str) -> float:
        """Send a command and return the response parsed as float."""
        return float(self.query(command))

    def write_checked(self, command: str) -> None:
        """
        Send a command and immediately drain the error queue.

        Per the BOP-GL manual (PAR. 3.3.5), a value rejected by a software
        limit (VOLT:LIM / CURR:LIM) is *silently disregarded* by the
        instrument -- no exception, no output change, just a queued SCPI
        error (-120 or -222) that a plain ``write()`` would never surface.
        Used for any command that programs an output setpoint or limit, so
        a stale/narrowed limit shows up as a Python exception instead of a
        mysteriously "stuck" output.
        """
        self._inst.write(command)
        errors = self.check_errors()
        if errors:
            raise RuntimeError(f"Kepco rejected {command!r}: {'; '.join(errors)}")

    def close(self) -> None:
        """Close the VISA resource."""
        self._inst.close()

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
        self.write("*RST")
        time.sleep(0.5)  # Allow unit to reinitialise

    def clear_status(self) -> None:
        """Clear all status registers (*CLS)."""
        self.write("*CLS")

    def wait(self) -> None:
        """Block until all pending operations complete (*WAI)."""
        self.write("*WAI")

    def trigger(self) -> None:
        """Send a group execute trigger (*TRG)."""
        self.write("*TRG")

    @property
    def identification(self) -> str:
        """Return the instrument identification string (*IDN?)."""
        return self.query("*IDN?")

    @property
    def status_byte(self) -> int:
        """Return the status byte register (*STB?)."""
        return int(self.query("*STB?"))

    @property
    def event_status(self) -> int:
        """Return the standard event status register (*ESR?)."""
        return int(self.query("*ESR?"))

    # ------------------------------------------------------------------ #
    #  Output enable / disable                                             #
    # ------------------------------------------------------------------ #

    @property
    def output_enabled(self) -> bool:
        """
        Enable or disable the DC output.

        Returns True if the output is currently enabled.
        """
        return bool(int(self.query("OUTP?")))

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
        response = self.query("FUNC:MODE?").upper()
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

    @property
    def voltage(self) -> float:
        """
        Programmed voltage setpoint in volts.

        Valid range: –MAX_VOLTAGE to +MAX_VOLTAGE (±20 V).
        """
        return self.query_float("VOLT?")

    @voltage.setter
    def voltage(self, value: float) -> None:
        self._check_range(value, -self.MAX_VOLTAGE, self.MAX_VOLTAGE, "Voltage")
        self.write_checked(f"VOLT {value:.6g}")

    @property
    def voltage_limit(self) -> float:
        """
        Voltage protection (OVP) limit in volts.

        When operating in current mode this is the compliance voltage limit.
        Valid range: 0 to MAX_VOLTAGE.
        """
        return self.query_float("VOLT:PROT:POS?")

    @voltage_limit.setter
    def voltage_limit(self, value: float) -> None:
        self._check_range(value, 0, self.MAX_VOLTAGE, "Voltage limit")
        self.write_checked(f"VOLT:PROT {value:.6g}")

    @property
    def voltage_setpoint_max(self) -> float:
        """
        Software ceiling on the ``voltage`` setpoint itself (VOLT:LIM), in volts.

        This is a *different* register from ``voltage_limit`` (VOLT:PROT).
        ``voltage_limit`` clamps the compliance voltage while the unit
        regulates current; this one gates what values the ``voltage``
        setter is even allowed to accept. Per the manual (PAR. 3.3.5) it
        defaults to MAX_VOLTAGE, but if it was ever narrowed and saved
        with ``MEM:UPD LIM`` in a previous session, ``*RST`` (and hence
        :meth:`reset`) will NOT restore it -- it silently keeps rejecting
        any higher ``voltage`` setpoint. Call
        :meth:`raise_range_limits_to_max` at the start of a session if you
        suspect this.
        """
        return self.query_float("VOLT:LIM:POS?")

    @voltage_setpoint_max.setter
    def voltage_setpoint_max(self, value: float) -> None:
        self._check_range(value, 0, self.MAX_VOLTAGE, "Voltage setpoint limit")
        self.write_checked(f"VOLT:LIM {value:.6g}")

    # ------------------------------------------------------------------ #
    #  Current programming and protection                                  #
    # ------------------------------------------------------------------ #

    @property
    def current(self) -> float:
        """
        Programmed current setpoint in amperes.

        Valid range: –MAX_CURRENT to +MAX_CURRENT (±50 A).
        """
        return self.query_float("CURR?")

    @current.setter
    def current(self, value: float) -> None:
        self._check_range(value, -self.MAX_CURRENT, self.MAX_CURRENT, "Current")
        self.write_checked(f"CURR {value:.6g}")

    @property
    def current_limit(self) -> float:
        """
        Current protection limit in amperes.

        When operating in voltage mode this is the compliance current limit.
        Valid range: 0 to MAX_CURRENT.
        """
        return self.query_float("CURR:PROT:POS?")

    @current_limit.setter
    def current_limit(self, value: float) -> None:
        self._check_range(value, 0, self.MAX_CURRENT, "Current limit")
        self.write_checked(f"CURR:PROT {value:.6g}")

    @property
    def current_setpoint_max(self) -> float:
        """
        Software ceiling on the ``current`` setpoint itself (CURR:LIM), in amperes.

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
        current won't climb past a fixed ceiling.
        """
        return self.query_float("CURR:LIM:POS?")

    @current_setpoint_max.setter
    def current_setpoint_max(self, value: float) -> None:
        self._check_range(value, 0, self.MAX_CURRENT, "Current setpoint limit")
        self.write_checked(f"CURR:LIM {value:.6g}")

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

    @property
    def measure_voltage(self) -> float:
        """Measure and return the actual output voltage in volts."""
        return self.query_float("MEAS:VOLT?")

    @property
    def measure_current(self) -> float:
        """Measure and return the actual output current in amperes."""
        return self.query_float("MEAS:CURR?")

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
    #  Status and error handling                                           #
    # ------------------------------------------------------------------ #

    @property
    def error(self) -> str:
        """
        Return the next error from the error queue.

        Returns ``"0,'No error'"`` when the queue is empty.
        """
        return self.query("SYST:ERR?")

    def check_errors(self) -> list[str]:
        """
        Drain the error queue and return all errors as a list.

        Returns an empty list if no errors are present.
        """
        errors = []
        while True:
            err = self.error
            if err.startswith("0"):
                break
            errors.append(err)
        return errors

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
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_range(value: float, lo: float, hi: float, name: str) -> None:
        if not lo <= value <= hi:
            raise ValueError(
                f"{name} {value} is out of range [{lo}, {hi}]."
            )

    def __repr__(self) -> str:
        return f"KepkoBOPGL(resource={self._inst.resource_name!r})"

    # ------------------------------------------------------------------ #
    #  Context manager support                                             #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

if __name__ == "__main__":
    import pyvisa

    rm = pyvisa.ResourceManager()

    with KepkoBOPGL(rm.open_resource("GPIB0::6::INSTR")) as psu:
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
