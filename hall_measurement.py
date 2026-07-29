import logging
import sys
from pathlib import Path
from time import sleep

import numpy as np
import pyvisa

from pymeasure.experiment import Procedure, FloatParameter, IntegerParameter
from pymeasure.instruments.keithley import Keithley2182, Keithley6221

# The Kepco magnet driver lives outside this project (sibling "KEPCO magnet"
# directory) and is a plain pyvisa driver rather than a pymeasure Instrument.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "KEPCO magnet"))
from kepco_magnet import KepkoBOPGL  # noqa: E402

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class HallMeasurement(Procedure):
    """
    Field-dependent Hall voltage measurement.

    Sources a fixed sense current with a Keithley 6221, measures the
    transverse (Hall) voltage with a Keithley 2182 nanovoltmeter, and
    sweeps the applied field via a Kepco bipolar power supply driving an
    electromagnet. At each field point the sense current is reversed and
    the Hall voltage averaged over repeated +/- pairs to cancel thermal
    EMF offsets.
    """

    GPIB_6221 = "GPIB0::12::INSTR"
    GPIB_2182 = "GPIB0::7::INSTR"
    GPIB_KEPCO = "GPIB0::6::INSTR"

    # Internal, non-user-facing constants
    NPLC = 5                       # 2182 integration time (power line cycles)
    SOURCE_DELAY = 0.05            # 6221 settle time after each current step (s)
    MAGNET_RAMP_STEP = 0.05        # magnet current ramp step (A)
    MAGNET_RAMP_DELAY = 0.02       # magnet current ramp step delay (s)
    MAGNET_VOLTAGE_LIMIT = 10.0    # magnet compliance voltage limit (V)

    sense_current = FloatParameter("Sense current", default=1e-3, units="A")
    compliance_voltage = FloatParameter("Compliance voltage", default=2, units="V")

    start_field = FloatParameter("Start field", default=-0.5, units="T")
    end_field = FloatParameter("End field", default=0.5, units="T")
    field_points = IntegerParameter("Field points", default=21)
    field_coefficient = FloatParameter("Field coefficient", default=0.1, units="T/A")

    averages = IntegerParameter("Averages per point", default=5)

    # Configurable, but intentionally left out of the GUI's `inputs` list.
    settling_time = FloatParameter("Settling time", default=1.0, units="s")

    DATA_COLUMNS = [
        "Magnetic Field (T)",
        "Magnet Current (A)",
        "Hall Voltage (V)",
        "Hall Voltage Std (V)",
        "Hall Resistance (ohm)",
    ]

    def startup(self):
        log.info("Connecting to Keithley 6221, Keithley 2182 and Kepco magnet supply")

        self.source = Keithley6221(self.GPIB_6221)
        self.voltmeter = Keithley2182(self.GPIB_2182)

        rm = pyvisa.ResourceManager()
        self._magnet_resource = rm.open_resource(self.GPIB_KEPCO)
        self.magnet = KepkoBOPGL(self._magnet_resource)

        self.source.reset()
        self.source.source_auto_range = True
        self.source.source_compliance = self.compliance_voltage
        self.source.source_delay = self.SOURCE_DELAY
        self.source.source_current = 0.0

        self.voltmeter.reset()
        self.voltmeter.ch_1.setup_voltage(auto_range=True, nplc=self.NPLC)

        self.magnet.reset()
        self.magnet.mode = "current"
        self.magnet.voltage_limit = self.MAGNET_VOLTAGE_LIMIT
        self.magnet.current = 0.0
        self.magnet.output_enabled = True

        self.source.enable_source()
        sleep(0.2)
        log.info("Instruments configured")

    def execute(self):
        fields = np.linspace(self.start_field, self.end_field, num=self.field_points)

        for index, field in enumerate(fields):
            target_current = field / self.field_coefficient
            log.info(f"Field {field:.4f} T -> magnet current {target_current:.4f} A")
            self.magnet.ramp_current(
                target_current, step=self.MAGNET_RAMP_STEP, delay=self.MAGNET_RAMP_DELAY
            )
            sleep(self.settling_time)

            hall_voltage = np.empty(self.averages)
            aborted = False

            for i in range(self.averages):
                self.source.source_current = self.sense_current
                v_plus = self.voltmeter.voltage

                self.source.source_current = -self.sense_current
                v_minus = self.voltmeter.voltage

                hall_voltage[i] = (v_plus - v_minus) / 2.0

                if self.should_stop():
                    aborted = True
                    hall_voltage = hall_voltage[:i + 1]
                    break

            self.source.source_current = self.sense_current

            mean_v = np.mean(hall_voltage)
            std_v = np.std(hall_voltage)

            data = {
                "Magnetic Field (T)": field,
                "Magnet Current (A)": target_current,
                "Hall Voltage (V)": mean_v,
                "Hall Voltage Std (V)": std_v,
                "Hall Resistance (ohm)": mean_v / self.sense_current,
            }
            self.emit("results", data)
            self.emit("progress", 100 * (index + 1) / len(fields))

            if aborted or self.should_stop():
                log.info("User aborted measurement")
                break

    def shutdown(self):
        log.info("Shutting down instruments")
        try:
            self.source.shutdown()
        except Exception as e:
            log.warning(f"Error shutting down Keithley 6221: {e}")

        try:
            self.magnet.zero_output(
                ramp=True, step=self.MAGNET_RAMP_STEP, delay=self.MAGNET_RAMP_DELAY
            )
            self.magnet.close()
        except Exception as e:
            log.warning(f"Error shutting down Kepco magnet supply: {e}")

        log.info("Measurement exited.")
