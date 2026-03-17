import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import sys
import tempfile
import random
from time import sleep
from datetime import datetime
from pymeasure.log import console_log
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import IntegerParameter, FloatParameter, Parameter
from IVmeasurements import IV2450Procedure
from IVgmeasurements import IVgProcedure

class RandomProcedure(Procedure):

    iterations = IntegerParameter('Loop Iterations', default=100)
    delay = FloatParameter('Delay Time', units='s', default=0.2)
    seed = Parameter('Random Seed', default='12345')

    time = datetime.now()

    DATA_COLUMNS = ['Iteration', 'Random Number']

    def startup(self):
        log.info("Setting the seed of the random number generator")
        random.seed(self.seed)

    def execute(self):
        log.info("Starting the loop of %d iterations" % self.iterations)
        for i in range(self.iterations):
            data = {
                'Iteration': i,
                'Random Number': random.random()
            }
            self.emit('results', data)
            log.debug("Emitting results: %s" % data)
            self.emit('progress', 100 * i / self.iterations)
            sleep(self.delay)
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break


class MainWindow(ManagedWindow):

    def __init__(self):
        super().__init__(
            procedure_class=IVgProcedure,
            inputs=['data_points', 'averages', 'max_vg', 'min_vg'],
            displays=['data_points', 'averages', 'max_vg', 'min_vg'],
            x_axis='Gate Voltage (V)',
            y_axis='Voltage (V)',
            sequencer=True,
            # sequence_file='path/to/file'
        )
        self.setWindowTitle('GUI Example')

        self.filename = r'V_vs_Vg_{Current bias:e}A_{Date}'
        self.store_measurement = True
        self.file_input.extensions = ['txt', 'csv']  # First is default
        self.file_input.filename_fixed = False


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
