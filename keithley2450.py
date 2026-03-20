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

import logging
import time
from warnings import warn

import numpy as np

from pymeasure.instruments import Instrument, SCPIMixin
from pymeasure.instruments.validators import truncated_range, strict_discrete_set
from keithley2450Buffer import Keithley2450Buffer

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Keithley2450(Keithley2450Buffer, SCPIMixin, Instrument):
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
        return "4" if val == "ON" else "2"

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
    # Buffer Statistics #
    # (defbuffer1, 2450 native - replaces Instrument.measurement class attrs)
    ####################

    @property
    def means(self):
        """ Get the buffer mean for the configured sense function. """
        return float(self.values(":TRACe:STATistics:AVERage? \"defbuffer1\","))

    @property
    def maximums(self):
        """ Get the buffer maximum for the configured sense function. """
        return float(self.values(":TRACe:STATistics:MAXimum? \"defbuffer1\","))

    @property
    def minimums(self):
        """ Get the buffer minimum for the configured sense function. """
        return float(self.values(":TRACe:STATistics:MINimum? \"defbuffer1\","))

    @property
    def standard_devs(self):
        """ Get the buffer standard deviation for the configured sense function. """
        return float(self.values(":TRACe:STATistics:STDev? \"defbuffer1\","))

    @property
    def mean_voltage(self):
        """ Get the mean voltage from buffer (call measure_voltage() first). """
        return self.means

    @property
    def mean_current(self):
        """ Get the mean current from buffer (call measure_current() first). """
        return self.means

    @property
    def mean_resistance(self):
        """ Get the mean resistance from buffer (call measure_resistance() first). """
        return self.means

    @property
    def max_voltage(self):
        """ Get the maximum voltage from buffer. """
        return self.maximums

    @property
    def max_current(self):
        """ Get the maximum current from buffer. """
        return self.maximums

    @property
    def max_resistance(self):
        """ Get the maximum resistance from buffer. """
        return self.maximums

    @property
    def min_voltage(self):
        """ Get the minimum voltage from buffer. """
        return self.minimums

    @property
    def min_current(self):
        """ Get the minimum current from buffer. """
        return self.minimums

    @property
    def min_resistance(self):
        """ Get the minimum resistance from buffer. """
        return self.minimums

    @property
    def std_voltage(self):
        """ Get the voltage standard deviation from buffer. """
        return self.standard_devs

    @property
    def std_current(self):
        """ Get the current standard deviation from buffer. """
        return self.standard_devs

    @property
    def std_resistance(self):
        """ Get the resistance standard deviation from buffer. """
        return self.standard_devs

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
