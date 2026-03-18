import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from pymeasure.instruments.keithley import Keithley2450, Keithley2400
from pymeasure.experiment import Procedure, Results, Worker
from pymeasure.experiment import FloatParameter, IntegerParameter, Metadata
from time import sleep
import numpy as np
from datetime import datetime


class SVMC18_SC16_Procedure(Procedure):
    '''
    Procedure with two Keithley 2450. One is used to supply gate voltage and measure voltage across the channel. The second one is used to apply bias current.
    '''
    data_points = IntegerParameter('Data points', default=20)
    averages = IntegerParameter('Averages', default=10)
    time = Metadata('Date', default=datetime.now())

    # Gate voltage parameters
    max_vg = FloatParameter('Maximum Vg', units='V', default=10)
    min_vg = FloatParameter('Minimum Vg', units='V', default=-10)

    # Current bias
    current_bias = FloatParameter('Current bias', units='A', default=1e-6)

    DATA_COLUMNS = ['Gate Voltage (V)', 'Current (A)', 'Current Std (A)']

    def startup(self):
        log.info("Connecting and configuring the instruments")

        # Source Gate voltage and measure voltage sourcemeter
        self.gate_measure = Keithley2400("GPIB::18")
        self.gate_measure.reset()
        self.gate_measure.use_front_terminals()
        self.gate_measure.apply_voltage(voltage_range=10, compliance_current=100e-6)
        self.gate_measure.measure_current(nplc=1, current=1e-3, auto_range=True)
        sleep(0.1)
        self.gate_measure.stop_buffer()
        self.gate_measure.disable_buffer()

        log.info("Instrument for gate and voltage configured")

        # Current bias instrument
        self.current_source = Keithley2400("GPIB::16")
        self.current_source.reset()
        self.current_source.use_front_terminals()
        self.current_source.apply_current(current_range=100e-6, compliance_voltage=10)
        sleep(0.1)
        self.current_source.stop_buffer()
        self.current_source.disable_buffer()

        log.info("Bias current instrument configured")

    def execute(self):

        voltages = np.linspace(
            self.min_vg,
            self.max_vg,
            num=self.data_points
        )
        self.gate_measure.enable_source()
        self.current_source.enable_source()

        log.info("Starting gate sweep with bias current measurement")
        for voltage in voltages:
            self.gate_measure.config_buffer(SVMC18_SC16_Procedure.averages.value)

            log.info(f"Setting gate voltage to {voltage} V with bias current {self.current_bias} A")
            self.gate_measure.source_voltage = voltage
            self.current_source.source_current = self.current_bias

            self.gate_measure.start_buffer()
            self.gate_measure.wait_for_buffer()

            data = {
                'Gate Voltage (V)': voltage,
                'Current (A)': self.gate_measure.mean_current,
                'Current Std (A)': self.gate_measure.std_current
            }

            self.emit('results', data)
            sleep(0.1)
            if self.should_stop():
                log.info("User aborted the procedure")
                break

    def shutdown(self):
        self.gate_measure.shutdown()
        self.current_source.shutdown()
        log.info("Finished measuring")


class SVMC18_SV16_Procedure(Procedure):
    '''
    Two 2450 in 2400 mode. GPIB::18 is used as voltage bias source and measuring current. GPIB::16 is used as a gate voltage source.
    '''

    data_points = IntegerParameter('Data points', default=20)
    averages = IntegerParameter('Averages', default=10)
    time = Metadata('Date', default=datetime.now())

    # Gate voltage parameters
    max_vg = FloatParameter('Maximum Vg', units='V', default=10)
    min_vg = FloatParameter('Minimum Vg', units='V', default=-10)

    # Current bias
    voltage_bias = FloatParameter('Voltage bias', units='V', default=1)

    DATA_COLUMNS = ['Gate Voltage (V)', 'Current (A)', 'Voltage Std (V)']

    def startup(self):
        log.info("Connecting and configuring the instruments")

        # Source bias voltage and measure current sourcemeter
        self.bias_measure = Keithley2400("GPIB::18")
        self.bias_measure.reset()
        self.bias_measure.use_front_terminals()
        self.bias_measure.apply_voltage(voltage_range=50, compliance_current=10e-3)
        self.bias_measure.measure_current(nplc=1, current=10e-3, auto_range=True)
        sleep(0.1)
        self.bias_measure.stop_buffer()
        self.bias_measure.disable_buffer()

        log.info("Instrument for bias and current configured")

        # Gate voltage sourcemeter
        self.gate_source = Keithley2400("GPIB::16")
        self.gate_source.reset()
        self.gate_source.use_front_terminals()
        self.gate_source.apply_voltage(voltage_range=60, compliance_current=10e-3)
        sleep(0.1)
        self.gate_source.stop_buffer()
        self.gate_source.disable_buffer()

        log.info("Gate voltage instrument configured")

    def execute(self):

        voltages = np.linspace(
            self.min_vg,
            self.max_vg,
            num=self.data_points
        )
        self.bias_measure.enable_source()
        self.gate_source.enable_source()

        log.info("Starting gate sweep with bias current measurement")
        for voltage in voltages:
            self.bias_measure.config_buffer(SVMV18_SC16_Procedure.averages.value)

            log.info(f"Setting gate voltage to {voltage} V with voltage bias {self.voltage_bias} V")
            self.bias_measure.source_voltage = voltage
            self.gate_source.source_voltage = self.voltage_bias

            self.bias_measure.start_buffer()
            self.bias_measure.wait_for_buffer()

            data = {
                'Gate Voltage (V)': voltage,
                'Current (A)': self.bias_measure.mean_current,
                'Current Std (A)': self.bias_measure.std_current
            }

            self.emit('results', data)
            sleep(0.1)
            if self.should_stop():
                log.info("User aborted the procedure")
                break

    def shutdown(self):
        self.bias_measure.shutdown()
        self.gate_source.shutdown()
        log.info("Finished measuring")


if __name__ == "__main__":
    pass
