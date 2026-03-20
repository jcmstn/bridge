import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import sys
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows import ManagedWindow
from gate_sweep import GateSweep
import platform


if platform.system() == "Darwin":
    path_to_lib = '/Library/Frameworks/NI4882.framework/NI4882'
    log.info(f"MacOS environment detected. Looking for NI4882 library in {path_to_lib}")
    log.warning("Only x86 archetecture supported. Make sure you are not running on ARM.")
    import gpib_ctypes
    gpib_ctypes.gpib.gpib._load_lib(path_to_lib)  # On MacOS, to find the NI library


class MainWindow(ManagedWindow):

    def __init__(self):
        super().__init__(
            procedure_class=GateSweep,
            inputs=[
                'use_2400',
                'start_vg',
                'end_vg',
                'voltage_bias',
                'bias',
                'data_points'
            ],
            displays=[
                'use_2400',
                'start_vg',
                'end_vg',
                'voltage_bias',
                'bias',
                'data_points'
            ],
            x_axis='Gate voltage (V)',
            y_axis='Current (A)',
            sequencer=True,
            sequence_file='./sequences/gate_sweep_sequence.txt'
        )
        self.setWindowTitle('Gate Sweep Program')

        self.filename = r'I_vs_Vg'
        self.store_measurement = True
        self.file_input.extensions = ['txt', 'csv']  # First is default
        self.file_input.filename_fixed = False


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
