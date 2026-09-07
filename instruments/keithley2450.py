#
# This file is part of the PyMeasure package.
#
# Copyright (c) 2013-2026 PyMeasure Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
"""
Keithley 2450 SourceMeter (SMU) — hand-written driver + easy-to-use helpers
==========================================================================
pymeasure ships a ``Keithley2450`` driver, but its buffer support speaks the
2400-emulation SCPI dialect, which a *native* 2450 rejects. This module keeps
a vendored ``Keithley2450`` class whose ``_Keithley2450Buffer`` mixin (just
above the class, nothing else uses it) speaks the native ``:TRACe:…`` dialect
— that is the only reason the class lives here rather than being imported
from pymeasure.

Two layers, use whichever fits:

* ``Keithley2450`` — the full class. Source V or I, measure V/I/R, ranges,
  NPLC, 2/4-wire, filters, front/rear terminals, native ``defbuffer1``
  statistics. Wire it up yourself for anything unusual.

* ``SMUConfig`` + ``connect_smu`` / ``set_source_level`` / ``read_measurement``
  / ``acquire_measurement`` / ``measure_buffered`` / ``shutdown_smu`` — the
  driver-contract wrapper (docs/architecture.md §4). One dataclass describes
  the whole instrument state; the free functions apply it. This is the
  surface a custom script should reach for first.

Failure policy: an SMU that is sourcing into a device is **load-bearing** —
``connect_smu`` and ``set_source_level`` raise on any failure; callers do not
pass ``None`` for the handle.

Usage example (custom script — source current, measure voltage, 4-wire):
    from instruments.keithley2450 import (
        SMUConfig, connect_smu, set_source_level, acquire_measurement, shutdown_smu)

    cfg = SMUConfig(visa_resource="GPIB0::18::INSTR", source_function="current",
                    compliance_voltage_V=2.0, four_wire=True, nplc=1.0,
                    source_limit_A=1e-3)
    smu = connect_smu(cfg)
    try:
        set_source_level(smu, cfg, 100e-6)          # 100 µA
        out = acquire_measurement(smu, cfg, n=10)   # {"mean": V, "sem": V}
    finally:
        shutdown_smu(smu)
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional
from warnings import warn

import numpy as np

from pymeasure.instruments import Instrument, SCPIMixin
from pymeasure.instruments.validators import truncated_range, strict_discrete_set

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class _Keithley2450Buffer:
    """ ``defbuffer1`` acquisition + statistics in the *native* 2450 SCPI
    dialect (``:TRACe:…``), as opposed to pymeasure's ``KeithleyBuffer``
    which uses the 2400-emulation commands a native 2450 rejects. Private
    mixin for ``Keithley2450`` below — nothing else uses it, so it is not a
    separate module. """

    buffer_points = Instrument.control(
        ":TRACe:POINts? \"defbuffer1\"", ":TRACe:POINts %d, \"defbuffer1\"",
        """ Control the number of buffer points in defbuffer1. """,
        validator=truncated_range,
        values=[1, 250000],  # 2450 defbuffer1 max capacity
        cast=int
    )

    def config_buffer(self, points=64, delay=0):
        """ Configure measurement buffer for specified points. """
        self.write("*CLS")
        self.write(":TRACe:CLEar \"defbuffer1\"")
        self.buffer_points = points

        # Native 2450: Set measurement count to match buffer size
        self.write(f":SENSe:COUNt {points}")
        self.check_errors()

    def is_buffer_full(self):
        """ Return True if buffer has reached the requested number of points. """
        # Native 2450: TRACe:ACTual? does NOT take buffer name parameter
        actual = int(self.ask(":TRACe:ACTual?"))
        return actual >= self.buffer_points

    def wait_for_buffer(self, should_stop=lambda: False, timeout=60, interval=0.1):
        """ Wait for full buffer or timeout/stop condition. """
        t = time.time()
        while not self.is_buffer_full():
            time.sleep(interval)
            if should_stop():
                return
            if (time.time() - t) > timeout:
                # Print buffer status for debugging
                actual = int(self.ask(":TRACe:ACTual?"))
                log.error(f"Buffer timeout: {actual}/{self.buffer_points} points")
                raise Exception("Timed out waiting for Keithley 2450 buffer to fill.")

    @property
    def buffer_data(self):
        """ Get numpy array of raw buffer values (only the points actually
        stored, which may be fewer than requested if wait_for_buffer()
        returned early via should_stop). """
        self.write(":FORMat:DATA ASCii")
        # Read what is actually in the buffer, not buffer_points (the
        # configured count) — asking for absent points errors on a 2450.
        actual = int(self.ask(":TRACe:ACTual?"))
        if actual < 1:
            return np.array([], dtype=np.float64)
        # Native 2450: TRACe:DATA? requires start, end, buffername, dataelement
        data = self.values(f':TRACe:DATA? 1, {actual}, "defbuffer1", READ')
        return np.array(data, dtype=np.float64)

    # Native 2450: Statistics are queried without trailing commas
    @property
    def mean_voltage(self): return float(self.ask(":TRACe:STATistics:AVERage? \"defbuffer1\""))
    @property
    def mean_current(self): return float(self.ask(":TRACe:STATistics:AVERage? \"defbuffer1\""))
    @property
    def mean_resistance(self): return float(self.ask(":TRACe:STATistics:AVERage? \"defbuffer1\""))

    @property
    def std_voltage(self): return float(self.ask(":TRACe:STATistics:STDDev? \"defbuffer1\""))
    @property
    def std_current(self): return float(self.ask(":TRACe:STATistics:STDDev? \"defbuffer1\""))
    @property
    def std_resistance(self): return float(self.ask(":TRACe:STATistics:STDDev? \"defbuffer1\""))

    @property
    def max_voltage(self): return float(self.ask(":TRACe:STATistics:MAXimum? \"defbuffer1\""))
    @property
    def max_current(self): return float(self.ask(":TRACe:STATistics:MAXimum? \"defbuffer1\""))
    @property
    def min_voltage(self): return float(self.ask(":TRACe:STATistics:MINimum? \"defbuffer1\""))
    @property
    def min_current(self): return float(self.ask(":TRACe:STATistics:MINimum? \"defbuffer1\""))

    def enable_statistics(self):
        """ The 2450 computes statistics automatically. Kept for backwards compatibility. """
        pass

    def start_buffer(self):
        """ Start buffer measurements natively on 2450. """
        self.write(":TRACe:TRIGger \"defbuffer1\"")

    def reset_buffer(self):
        """ Reset buffer and status. """
        self.write("*CLS")
        self.write(":TRACe:CLEar \"defbuffer1\"")

    def stop_buffer(self):
        """ Abort buffering. """
        self.write(":ABORt")

    def disable_buffer(self):
        """ Restore instrument to single-shot measurement mode. """
        self.write(":SENSe:COUNt 1")


class Keithley2450(_Keithley2450Buffer, SCPIMixin, Instrument):
    """ Represents the Keithley 2450 SourceMeter with native 2450 SCPI support
    (no 2400 emulation required). Includes enhanced buffer statistics via defbuffer1.

    Example:

    .. code-block:: python

        keithley = Keithley2450("GPIB::1")

        keithley.apply_current()                # Sets up to source current
        keithley.source_current_range = 10e-3   # Sets the source current range to 10 mA
        keithley.compliance_voltage = 10        # Sets the compliance voltage to 10 V
        keithley.source_current = 0             # Sets the source current to 0 mA
        keithley.enable_source()                # Enables the source output

        keithley.measure_voltage()              # Sets up to measure voltage
        keithley.config_buffer(10)              # Configure 10-point buffer
        keithley.enable_statistics()            # Enable mean/std calculation
        keithley.start_buffer()                 # Start buffered acquisition
        keithley.wait_for_buffer()              # Wait until full
        print(f"Mean V: {keithley.mean_voltage:.6f}, Std V: {keithley.std_voltage:.6f}")

        keithley.shutdown()                     # Ramps to 0 and disables output
    """

    def __init__(self, adapter, name="Keithley 2450 SourceMeter", **kwargs):
        super().__init__(
            adapter,
            name,
            **kwargs
        )

    source_mode = Instrument.control(
        ":SOUR:FUNC?", ":SOUR:FUNC %s",
        """ Control (string) the source mode, which can
        take the values 'current' or 'voltage'. The convenience methods
        :meth:`~.Keithley2450.apply_current` and :meth:`~.Keithley2450.apply_voltage`
        can also be used. """,
        validator=strict_discrete_set,
        values={'current': 'CURR', 'voltage': 'VOLT'},
        map_values=True
    )

    source_enabled = Instrument.measurement(
        "OUTPUT?",
        """ Get a boolean value that is True if the source is enabled. """,
        cast=bool
    )

    ###############
    # Current (A) #
    ###############

    current = Instrument.measurement(
        ":READ?",
        """ Get the current in Amps, if configured for this reading. """
    )

    current_range = Instrument.control(
        ":SENS:CURR:RANG?", ":SENS:CURR:RANG:AUTO 0;:SENS:CURR:RANG %g",
        """ Control (floating) the measurement current range in Amps,
        which can take values between -1.05 and +1.05 A.
        Auto-range is disabled when this property is set. """,
        validator=truncated_range,
        values=[-1.05, 1.05]
    )

    current_nplc = Instrument.control(
        ":SENS:CURR:NPLC?", ":SENS:CURR:NPLC %g",
        """ Control (floating) the number of power line cycles (NPLC) for
        DC current measurements. Takes values from 0.01 to 10, where 0.1,
        1, and 10 are Fast, Medium, and Slow respectively. """,
        values=[0.01, 10]
    )

    compliance_current = Instrument.control(
        ":SOUR:VOLT:ILIM?", ":SOUR:VOLT:ILIM %g",
        """ Control (floating) the compliance current in Amps. """,
        validator=truncated_range,
        values=[-1.05, 1.05]
    )

    source_current = Instrument.control(
        ":SOUR:CURR?", ":SOUR:CURR:LEV %g",
        """ Control (floating) the source current in Amps. """
    )

    source_current_range = Instrument.control(
        ":SOUR:CURR:RANG?", ":SOUR:CURR:RANG:AUTO 0;:SOUR:CURR:RANG %g",
        """ Control (floating) the source current range in Amps,
        which can take values between -1.05 and +1.05 A.
        Auto-range is disabled when this property is set. """,
        validator=truncated_range,
        values=[-1.05, 1.05]
    )

    source_current_delay = Instrument.control(
        ":SOUR:CURR:DEL?", ":SOUR:CURR:DEL %g",
        """ Control (floating) a manual delay for the source after the output
        is turned on before a measurement is taken. Valid values are between
        0 [seconds] and 999.9999 [seconds]. """,
        validator=truncated_range,
        values=[0, 999.9999],
    )

    source_current_delay_auto = Instrument.control(
        ":SOUR:CURR:DEL:AUTO?", ":SOUR:CURR:DEL:AUTO %d",
        """ Control (bool) auto delay. Valid values are True and False. """,
        values={True: 1, False: 0},
        map_values=True,
    )

    ###############
    # Voltage (V) #
    ###############

    voltage = Instrument.measurement(
        ":READ?",
        """ Get the voltage in Volts, if configured for this reading. """
    )

    voltage_range = Instrument.control(
        ":SENS:VOLT:RANG?", ":SENS:VOLT:RANG:AUTO 0;:SENS:VOLT:RANG %g",
        """ Control (floating) the measurement voltage range in Volts,
        which can take values from -210 to 210 V.
        Auto-range is disabled when this property is set. """,
        validator=truncated_range,
        values=[-210, 210]
    )

    voltage_nplc = Instrument.control(
        ":SENS:VOLT:NPLC?", ":SENS:VOLT:NPLC %g",
        """ Control (floating) the number of power line cycles (NPLC) for
        DC voltage measurements. Takes values from 0.01 to 10, where 0.1,
        1, and 10 are Fast, Medium, and Slow respectively. """
    )

    compliance_voltage = Instrument.control(
        ":SOUR:CURR:VLIM?", ":SOUR:CURR:VLIM %g",
        """ Control (floating) the compliance voltage in Volts. """,
        validator=truncated_range,
        values=[-210, 210]
    )

    source_voltage = Instrument.control(
        ":SOUR:VOLT?", ":SOUR:VOLT:LEV %g",
        """ Control (floating) the source voltage in Volts. """
    )

    source_voltage_range = Instrument.control(
        ":SOUR:VOLT:RANG?", ":SOUR:VOLT:RANG:AUTO 0;:SOUR:VOLT:RANG %g",
        """ Control (floating) the source voltage range in Volts,
        which can take values from -210 to 210 V.
        Auto-range is disabled when this property is set. """,
        validator=truncated_range,
        values=[-210, 210]
    )

    source_voltage_delay = Instrument.control(
        ":SOUR:VOLT:DEL?", ":SOUR:VOLT:DEL %g",
        """ Control (floating) a manual delay for the source after the output
        is turned on before a measurement is taken. Valid values are between
        0 [seconds] and 999.9999 [seconds]. """,
        validator=truncated_range,
        values=[0, 999.9999],
    )

    source_voltage_delay_auto = Instrument.control(
        ":SOUR:VOLT:DEL:AUTO?", ":SOUR:VOLT:DEL:AUTO %d",
        """ Control (bool) auto delay. Valid values are True and False. """,
        values={True: 1, False: 0},
        map_values=True,
    )

    ####################
    # Resistance (Ohm) #
    ####################

    resistance = Instrument.measurement(
        ":READ?",
        """ Get the resistance in Ohms, if configured for this reading. """
    )

    resistance_range = Instrument.control(
        ":SENS:RES:RANG?", ":SENS:RES:RANG:AUTO 0;:SENS:RES:RANG %g",
        """ Control (floating) the resistance range in Ohms,
        which can take values from 0 to 210 MOhms.
        Auto-range is disabled when this property is set. """,
        validator=truncated_range,
        values=[0, 210e6]
    )

    resistance_nplc = Instrument.control(
        ":SENS:RES:NPLC?", ":SENS:RES:NPLC %g",
        """ Control (floating) the number of power line cycles (NPLC) for
        2-wire resistance measurements. Takes values from 0.01 to 10, where
        0.1, 1, and 10 are Fast, Medium, and Slow respectively. """
    )

    wires = Instrument.control(
        ":SENS:RES:RSENSE?", ":SENS:RES:RSENSE %d",
        """ Control (integer) the number of wires in use for resistance
        measurements, which can take the value of 2 or 4. """,
        validator=strict_discrete_set,
        values={4: 1, 2: 0},
        map_values=True
    )

    @property
    def sense_wire_mode(self):
        """ Get 2-wire ('2') or 4-wire ('4') sense mode. """
        val = self.ask(":SENS:CURR:RSENSE?").strip()
        return "4" if val == "1" else "2"

    @sense_wire_mode.setter
    def sense_wire_mode(self, value):
        """ Set 2-wire ('2') or 4-wire ('4') sense mode. """
        if value not in ("2", "4"):
            raise ValueError("sense_wire_mode must be '2' or '4'")
        scpi_val = "1" if value == "4" else "0"
        self.write(f":SENS:CURR:RSENSE {scpi_val}")
        self.write(f":SENS:VOLT:RSENSE {scpi_val}")

    ###########
    # Filters #
    ###########

    current_filter_type = Instrument.control(
        ":SENS:CURR:AVER:TCON?", ":SENS:CURR:AVER:TCON %s",
        """ Control (string) the filter type for current.
        REP: Repeating filter. MOV: Moving filter. """,
        validator=strict_discrete_set,
        values=['REP', 'MOV'],
        map_values=False
    )

    current_filter_count = Instrument.control(
        ":SENS:CURR:AVER:COUNT?", ":SENS:CURR:AVER:COUNT %d",
        """ Control (integer) the number of readings acquired and stored
        in the filter buffer for averaging. """,
        validator=truncated_range,
        values=[1, 100],
        cast=int
    )

    current_filter_state = Instrument.control(
        ":SENS:CURR:AVER?", ":SENS:CURR:AVER %s",
        """ Control (string) if the current filter is active. """,
        validator=strict_discrete_set,
        values=['ON', 'OFF'],
        map_values=False
    )

    voltage_filter_type = Instrument.control(
        ":SENS:VOLT:AVER:TCON?", ":SENS:VOLT:AVER:TCON %s",
        """ Control (string) the filter type for voltage.
        REP: Repeating filter. MOV: Moving filter. """,
        validator=strict_discrete_set,
        values=['REP', 'MOV'],
        map_values=False
    )

    voltage_filter_count = Instrument.control(
        ":SENS:VOLT:AVER:COUNT?", ":SENS:VOLT:AVER:COUNT %d",
        """ Control (integer) the number of readings acquired and stored
        in the filter buffer for averaging. """,
        validator=truncated_range,
        values=[1, 100],
        cast=int
    )

    #####################
    # Output subsystem  #
    #####################

    current_output_off_state = Instrument.control(
        ":OUTP:CURR:SMOD?", ":OUTP:CURR:SMOD %s",
        """ Control SourceMeter current output-off state.
        HIMP: relay open. NORM: V-source 0V. ZERO: V-source 0V with
        programmed compliance. GUAR: I-source 0A. """,
        validator=strict_discrete_set,
        values=['HIMP', 'NORM', 'ZERO', 'GUAR'],
        map_values=False
    )

    voltage_output_off_state = Instrument.control(
        ":OUTP:VOLT:SMOD?", ":OUTP:VOLT:SMOD %s",
        """ Control SourceMeter voltage output-off state.
        HIMP: relay open. NORM: V-source 0V. ZERO: V-source 0V with
        programmed compliance. GUAR: I-source 0A. """,
        validator=strict_discrete_set,
        values=['HIMP', 'NORM', 'ZERO', 'GUAR'],
        map_values=False
    )

    ####################
    # Methods          #
    ####################

    def enable_source(self):
        """ Enables the source of current or voltage depending on the
        configuration of the instrument. """
        self.write("OUTPUT ON")

    def disable_source(self):
        """ Disables the source of current or voltage depending on the
        configuration of the instrument. """
        self.write("OUTPUT OFF")

    def measure_resistance(self, nplc=1, resistance=2.1e5, auto_range=True):
        """ Configures the measurement of resistance.

        :param nplc: Number of power line cycles (NPLC) from 0.01 to 10
        :param resistance: Upper limit of resistance in Ohms
        :param auto_range: Enables auto_range if True, else uses the set resistance
        """
        log.info("%s is measuring resistance.", self.name)
        self.write(":SENS:FUNC 'RES';"
                   ":SENS:RES:NPLC %f;" % nplc)
        if auto_range:
            self.write(":SENS:RES:RANG:AUTO 1;")
        else:
            self.resistance_range = resistance
        self.check_errors()

    def measure_voltage(self, nplc=1, voltage=21.0, auto_range=True):
        """ Configures the measurement of voltage.

        :param nplc: Number of power line cycles (NPLC) from 0.01 to 10
        :param voltage: Upper limit of voltage in Volts
        :param auto_range: Enables auto_range if True, else uses the set voltage
        """
        log.info("%s is measuring voltage.", self.name)
        self.write(":SENS:FUNC 'VOLT';"
                   ":SENS:VOLT:NPLC %f;" % nplc)
        if auto_range:
            self.write(":SENS:VOLT:RANG:AUTO 1;")
        else:
            self.voltage_range = voltage
        self.check_errors()

    def measure_current(self, nplc=1, current=1.05e-4, auto_range=True):
        """ Configures the measurement of current.

        :param nplc: Number of power line cycles (NPLC) from 0.01 to 10
        :param current: Upper limit of current in Amps
        :param auto_range: Enables auto_range if True, else uses the set current
        """
        log.info("%s is measuring current.", self.name)
        self.write(":SENS:FUNC 'CURR';"
                   ":SENS:CURR:NPLC %f;" % nplc)
        if auto_range:
            self.write(":SENS:CURR:RANG:AUTO 1;")
        else:
            self.current_range = current
        self.check_errors()

    def auto_range_source(self):
        """ Configures the source to use an automatic range. """
        if self.source_mode == 'current':
            self.write(":SOUR:CURR:RANG:AUTO 1")
        else:
            self.write(":SOUR:VOLT:RANG:AUTO 1")

    def apply_current(self, current_range=None, compliance_voltage=0.1):
        """ Configures the instrument to apply a source current, and
        uses an auto range unless a current range is specified.

        :param compliance_voltage: A float in the correct range for
                                   :attr:`~.Keithley2450.compliance_voltage`
        :param current_range: A :attr:`~.Keithley2450.current_range` value or None
        """
        log.info("%s is sourcing current.", self.name)
        self.source_mode = 'current'
        if current_range is None:
            self.auto_range_source()
        else:
            self.source_current_range = current_range
        self.compliance_voltage = compliance_voltage
        self.check_errors()

    def apply_voltage(self, voltage_range=None, compliance_current=0.1):
        """ Configures the instrument to apply a source voltage, and
        uses an auto range unless a voltage range is specified.

        :param compliance_current: A float in the correct range for
                                   :attr:`~.Keithley2450.compliance_current`
        :param voltage_range: A :attr:`~.Keithley2450.voltage_range` value or None
        """
        log.info("%s is sourcing voltage.", self.name)
        self.source_mode = 'voltage'
        if voltage_range is None:
            self.auto_range_source()
        else:
            self.source_voltage_range = voltage_range
        self.compliance_current = compliance_current
        self.check_errors()

    def beep(self, frequency, duration):
        """ Sounds a system beep.

        :param frequency: A frequency in Hz between 65 Hz and 2 MHz
        :param duration: A time in seconds between 0 and 7.9 seconds
        """
        self.write(f":SYST:BEEP {frequency:g}, {duration:g}")

    def triad(self, base_frequency, duration):
        """ Sounds a musical triad using the system beep.

        :param base_frequency: A frequency in Hz between 65 Hz and 1.3 MHz
        :param duration: A time in seconds between 0 and 7.9 seconds
        """
        self.beep(base_frequency, duration)
        time.sleep(duration)
        self.beep(base_frequency * 5.0 / 4.0, duration)
        time.sleep(duration)
        self.beep(base_frequency * 6.0 / 4.0, duration)

    @property
    def error(self):
        """ Get the next error from the queue.

        .. deprecated:: 0.15
            Use `next_error` instead.
        """
        warn("Deprecated to use `error`, use `next_error` instead.", FutureWarning)
        return self.next_error

    def reset(self):
        """ Resets the instrument and clears the queue. """
        self.write("*RST;:STAT:PRES;:*CLS;")

    def ramp_to_current(self, target_current, steps=30, pause=20e-3):
        """ Ramps to a target current from the set current value over
        a number of linear steps, each separated by a pause duration.

        :param target_current: A current in Amps
        :param steps: An integer number of steps
        :param pause: A pause duration in seconds to wait between steps
        """
        currents = np.linspace(self.source_current, target_current, steps)
        for current in currents:
            self.source_current = current
            time.sleep(pause)

    def ramp_to_voltage(self, target_voltage, steps=30, pause=20e-3):
        """ Ramps to a target voltage from the set voltage value over
        a number of linear steps, each separated by a pause duration.

        :param target_voltage: A voltage in Volts
        :param steps: An integer number of steps
        :param pause: A pause duration in seconds to wait between steps
        """
        voltages = np.linspace(self.source_voltage, target_voltage, steps)
        for voltage in voltages:
            self.source_voltage = voltage
            time.sleep(pause)

    def trigger(self):
        """ Executes a bus trigger. """
        return self.write("*TRG")

    def use_rear_terminals(self):
        """ Enables the rear terminals for measurement, and
        disables the front terminals. """
        self.write(":ROUT:TERM REAR")

    def use_front_terminals(self):
        """ Enables the front terminals for measurement, and
        disables the rear terminals. """
        self.write(":ROUT:TERM FRON")

    def buffered_measurement(self, points=10, sense_func="VOLT", **measure_kwargs):
        """ Quick buffered measurement with statistics.

        :param points: Number of buffer points.
        :param sense_func: 'VOLT', 'CURR', or 'RES'.
        :param measure_kwargs: Passed to the relevant measure_*() method.
        :return: Dict with 'mean', 'std', 'max', 'min', 'data'.
        """
        _sense_map = {
            'VOLT': 'voltage',
            'CURR': 'current',
            'RES': 'resistance',
        }
        sense_name = _sense_map.get(sense_func.upper(), 'voltage')
        measure_method = getattr(self, f"measure_{sense_name}")
        measure_method(**measure_kwargs)

        self.config_buffer(points)
        self.enable_statistics()
        self.start_buffer()
        self.wait_for_buffer()

        return {
            'mean': getattr(self, f"mean_{sense_name}"),
            'std':  getattr(self, f"std_{sense_name}"),
            'max':  getattr(self, f"max_{sense_name}"),
            'min':  getattr(self, f"min_{sense_name}"),
            'data': self.buffer_data
        }

    def shutdown(self):
        """ Ensures that the current or voltage is turned to zero
        and disables the output. """
        log.info("Shutting down %s.", self.name)
        if self.source_mode == 'current':
            self.ramp_to_current(0.0)
        else:
            self.ramp_to_voltage(0.0)
        self.stop_buffer()
        self.disable_source()
        super().shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Easy-to-use wrapper  ── the driver-contract layer (docs/architecture.md §4) ──
# ─────────────────────────────────────────────────────────────────────────────
# One dataclass describes the whole SMU state; connect_smu() applies it and
# returns a live Keithley2450. A custom script never touches the class unless
# it needs something this surface doesn't cover.

_SMU_FUNCS = ("voltage", "current")
_SENSE_FUNCS = ("voltage", "current", "resistance")
_OFF_STATES = {"himp": "HIMP", "normal": "NORM", "zero": "ZERO", "guard": "GUAR"}


@dataclass
class SMUConfig:
    """Keithley 2450 as a general-purpose SMU. Every field has a safe default;
    override only what your measurement needs.

    ``source_range`` / ``sense_range`` are in the unit of their function
    (V when ``*_function == "voltage"``, A for ``"current"``, Ω for a
    ``"resistance"`` sense); ``None`` means autorange.
    """
    visa_resource: str        = "GPIB0::18::INSTR"
    source_function: str      = "voltage"     # "voltage" | "current" — what the SMU drives
    sense_function: Optional[str] = None      # "voltage"|"current"|"resistance"; None → the other of source_function
    compliance_current_A: float = 1e-3        # limit while sourcing voltage [A]
    compliance_voltage_V: float = 10.0        # limit while sourcing current [V]
    source_range: Optional[float] = None      # None → autorange the source
    sense_range: Optional[float]  = None      # None → autorange the measurement
    nplc: float               = 1.0           # integration time [power-line cycles], 0.01–10
    four_wire: bool           = False         # True → remote (4-wire) sense on both V and I
    terminals: str            = "front"       # "front" | "rear"
    source_delay_s: Optional[float] = None    # None → the 2450's own auto source delay
    output_off_state: str     = "himp"        # state when the output is disabled; "himp" = relay open (safe)
    source_limit_V: float     = 21.0          # set_source_level() refuses |V| beyond this — raise per device
    source_limit_A: float     = 1e-3          # set_source_level() refuses |I| beyond this — raise per device


def _resolved_sense(cfg: SMUConfig) -> str:
    if cfg.sense_function is not None:
        return cfg.sense_function
    return "current" if cfg.source_function == "voltage" else "voltage"


def connect_smu(cfg: SMUConfig) -> Keithley2450:
    """Open the VISA session, reset, apply every field of ``cfg``, enable the
    source, and return the live handle. Raises on any failure (load-bearing)."""
    if cfg.source_function not in _SMU_FUNCS:
        raise ValueError(f"source_function must be one of {_SMU_FUNCS}, got {cfg.source_function!r}")
    sense = _resolved_sense(cfg)
    if sense not in _SENSE_FUNCS:
        raise ValueError(f"sense_function must be one of {_SENSE_FUNCS}, got {sense!r}")
    if cfg.output_off_state.lower() not in _OFF_STATES:
        raise ValueError(f"output_off_state must be one of {tuple(_OFF_STATES)}, got {cfg.output_off_state!r}")

    smu = Keithley2450(cfg.visa_resource)
    smu.reset()

    if cfg.source_function == "voltage":
        smu.apply_voltage(voltage_range=cfg.source_range,
                          compliance_current=cfg.compliance_current_A)
    else:
        smu.apply_current(current_range=cfg.source_range,
                          compliance_voltage=cfg.compliance_voltage_V)

    measure = getattr(smu, f"measure_{sense}")
    if cfg.sense_range is None:
        measure(nplc=cfg.nplc, auto_range=True)
    else:
        measure(nplc=cfg.nplc, auto_range=False, **{sense: cfg.sense_range})

    # 4-wire: sense_wire_mode writes BOTH :SENS:CURR:RSENSE and :SENS:VOLT:RSENSE,
    # unlike `wires` (resistance function only) — so it is correct for a
    # source-I / measure-V four-probe setup.
    smu.sense_wire_mode = "4" if cfg.four_wire else "2"

    smu.use_rear_terminals() if cfg.terminals == "rear" else smu.use_front_terminals()

    if cfg.source_delay_s is not None:
        setattr(smu, f"source_{cfg.source_function}_delay", cfg.source_delay_s)

    setattr(smu, f"{cfg.source_function}_output_off_state", _OFF_STATES[cfg.output_off_state.lower()])

    smu.enable_source()
    log.info(
        "Keithley 2450 SMU connected: %s  source=%s  sense=%s  %s  NPLC=%.3g",
        cfg.visa_resource, cfg.source_function, sense,
        "4-wire" if cfg.four_wire else "2-wire", cfg.nplc,
    )
    return smu


def set_source_level(smu: Keithley2450, cfg: SMUConfig, level: float) -> None:
    """Set the source setpoint (V or A per ``cfg.source_function``), refusing
    to exceed the configured software limit — mirrors ``set_gate_voltage``."""
    if cfg.source_function == "voltage":
        if abs(level) > cfg.source_limit_V:
            raise ValueError(
                f"Requested source voltage {level:.4g} V exceeds source_limit_V "
                f"±{cfg.source_limit_V:.4g} V — refusing to set it."
            )
        smu.source_voltage = level
    else:
        if abs(level) > cfg.source_limit_A:
            raise ValueError(
                f"Requested source current {level:.4g} A exceeds source_limit_A "
                f"±{cfg.source_limit_A:.4g} A — refusing to set it."
            )
        smu.source_current = level


def read_measurement(smu: Keithley2450, cfg: SMUConfig) -> float:
    """One fresh reading of the sense quantity, in canonical units (V/A/Ω)."""
    return float(getattr(smu, _resolved_sense(cfg)))


def acquire_measurement(
    smu: Keithley2450,
    cfg: SMUConfig,
    n: int,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Average ``n`` fresh readings of the sense quantity (slow Python loop,
    one ``:READ?`` per sample). Returns ``{"mean", "sem"}`` where ``sem`` is
    the sample stdev / sqrt(n) (``nan`` for n == 1) — same shape as
    ``keithley2182.acquire_averaged_voltage``. ``stop_event`` is checked
    between samples so a UI abort can cut a long average short."""
    sense = _resolved_sense(cfg)
    samples = np.empty(n)
    n_used = 0
    for i in range(n):
        samples[i] = float(getattr(smu, sense))
        n_used = i + 1
        if stop_event is not None and stop_event.is_set():
            break
    used = samples[:n_used]
    sem = float(np.std(used, ddof=1) / np.sqrt(n_used)) if n_used >= 2 else float("nan")
    return {"mean": float(np.mean(used)), "sem": sem}


def measure_buffered(smu: Keithley2450, n: int, timeout_s: float = 30.0) -> dict:
    """Take ``n`` samples into the native ``defbuffer1`` and return its
    on-instrument statistics ``{"mean", "std", "n"}`` for the active sense
    function — far faster than :func:`acquire_measurement` for large ``n``.
    ``n`` reflects the points actually stored (may be < requested if the
    buffer fill times out).

    Restores single-shot mode (``:SENSe:COUNt 1``) on the way out — without
    that, every later bare ``:READ?`` (read_measurement / acquire_measurement)
    would keep triggering an ``n``-sample sweep."""
    try:
        smu.config_buffer(n)
        smu.start_buffer()
        smu.wait_for_buffer(timeout=timeout_s)
        actual = int(smu.ask(":TRACe:ACTual?"))
        mean = float(smu.ask(':TRACe:STATistics:AVERage? "defbuffer1"'))
        std = float(smu.ask(':TRACe:STATistics:STDDev? "defbuffer1"'))
        return {"mean": mean, "std": std, "n": actual}
    finally:
        smu.disable_buffer()          # :SENSe:COUNt 1


def shutdown_smu(smu: Keithley2450) -> None:
    """Ramp the source to zero, abort any buffer, disable the output, close."""
    smu.shutdown()
    log.info("Keithley 2450 SMU output disabled")
