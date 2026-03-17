import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from pymeasure.instruments.keithley import Keithley2450
from pymeasure.experiment import Procedure, Results, Worker
from pymeasure.experiment import FloatParameter, IntegerParameter, Metadata
from time import sleep
import numpy as np
from datetime import datetime


class IVgProcedure(Procedure):

    data_points = IntegerParameter('Data points', default=20)
    averages = IntegerParameter('Averages', default=10)
    time = Metadata('Date', default=datetime.now())

    # Gate voltage parameters
    max_vg = FloatParameter('Maximum Vg', units='V', default=10)
    min_vg = FloatParameter('Minimum Vg', units='V', default=-10)

    # Current bias
    current_bias = FloatParameter('Current bias', units='A', default=1e-6)

    DATA_COLUMNS = ['Gate Voltage (V)', 'Voltage (V)', 'Voltage Std (V)']

    def startup(self):
        log.info("Connecting and configuring the instruments")

        # Source Gate voltage and measure voltage sourcemeter
        self.gate_measure = Keithley2450("GPIB::16")
        self.gate_measure.reset()
        self.gate_measure.use_front_terminals()
        self.gate_measure.apply_voltage(voltage_range=10, compliance_current=100e-6)
        self.gate_measure.measure_voltage(nplc=1, voltage=21, auto_range=True)
        sleep(0.1)
        self.gate_measure.stop_buffer()
        self.gate_measure.disable_buffer()

        log.info("Instrument for gate and voltage configured")

        # Current bias instrument
        self.current_source = Keithley2450("GPIB::18")
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
            self.gate_measure.config_buffer(self.averages.value)

            log.info(f"Setting gate voltage to {voltage} V with bias current {self.current_bias} A")
            self.gate_measure.source_voltage = voltage
            self.current_source.source_current = self.current_bias

            self.gate_measure.start_buffer()
            self.gate_measure.wait_for_buffer()

            data = {
                'Gate Voltage (V)': voltage,
                'Voltage (V)': self.gate_measure.mean_voltage,
                'Voltage Std (V)': self.gate_measure.std_voltage
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

if __name__ == "__main__":
    pass
