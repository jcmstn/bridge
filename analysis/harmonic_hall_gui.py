"""PySide6 GUI for batch harmonic-Hall analysis.

Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Point it at a directory of mfli_dual_harmonic.py output CSVs; it steps
through them one at a time. Each file's excitation, filter, phase and Hall
bar metadata is read straight out of the CSV -- see build_run_metadata() in
bridge/mfli/mfli_dual_harmonic.py for what it writes, and
harmonic_hall.apply_run_metadata() for how it's picked up here -- so nothing
needs to be typed in for those. The panel on the right covers only the
handful of sample properties no instrument in that script measures (Ms,
layer thicknesses/resistivities, the NM material, the field's in-plane
azimuth); values typed there stay put as you step to the next file, so you
only need to enter them once per sample.

Unlike ahe_gui.py's AHE analysis, there is no interactive fit range to drag
across -- the harmonic-Hall fits (hard-axis 1f loop, 2f torque slope) are
fully automatic -- so review consists of reading the plot grid and the sanity
report on the left. "Skip" moves on without writing anything; "Save && Next"
writes the PNG/report/JSON that run() would write for a single file (into an
`analyzed/` subdirectory next to the input directory, one file's outputs
sharing that folder with every other file's) and moves on.

Run with: python harmonic_hall_gui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from analysis import harmonic_hall as hh

OUTPUT_SUBDIR_NAME = "analyzed"

# Sample properties mfli_dual_harmonic.py never records, because no
# instrument in that script measures them (see harmonic_hall.py's
# apply_run_metadata() / RUN_METADATA_COLUMNS for everything that *is* read
# from the file automatically) -- the only Config fields worth a manual
# input here. (cfg_field, label, unit-hint-shown-as-placeholder).
MANUAL_METADATA_FIELDS = [
    ("Ms_kA_per_m", "Ms (independent, VSM/SQUID)", "kA/m"),
    ("ni_thickness_nm", "Ni thickness", "nm"),
    ("nm_thickness_nm", "NM thickness", "nm"),
    ("nm_material", "NM material", "e.g. Pt"),
    ("ni_resistivity_uohm_cm", "Ni resistivity", "uOhm*cm"),
    ("nm_resistivity_uohm_cm", "NM resistivity", "uOhm*cm"),
    ("field_azimuth_deg", "Field azimuth vs. current", "deg"),
]


class HarmonicHallWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Harmonic Hall Analysis")
        self.resize(1400, 900)

        self.files: list[Path] = []
        self.index: int = -1
        self.output_dir: Path | None = None
        self.result: hh.AnalysisResult | None = None

        self._build_ui()

    # ---- UI construction ----------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)

        top_bar = QtWidgets.QHBoxLayout()
        self.choose_dir_button = QtWidgets.QPushButton("Choose input directory...")
        self.choose_dir_button.clicked.connect(self.choose_directory)
        self.dir_label = QtWidgets.QLabel("No directory selected")
        top_bar.addWidget(self.choose_dir_button)
        top_bar.addWidget(self.dir_label, stretch=1)
        outer.addLayout(top_bar)

        self.status_label = QtWidgets.QLabel(
            "Choose a directory containing mfli_dual_harmonic.py output CSVs to begin."
        )
        outer.addWidget(self.status_label)

        body = QtWidgets.QHBoxLayout()
        outer.addLayout(body, stretch=1)

        # ---- left: plot grid + sanity report -----------------------------
        left = QtWidgets.QVBoxLayout()
        body.addLayout(left, stretch=3)

        self.figure = Figure(figsize=(14, 11))
        self.canvas = FigureCanvasQTAgg(self.figure)
        left.addWidget(NavigationToolbar2QT(self.canvas, self))
        left.addWidget(self.canvas, stretch=3)

        self.report_text = QtWidgets.QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 11px;")
        left.addWidget(self.report_text, stretch=1)

        # ---- right: manual (not-in-file) sample metadata ------------------
        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, stretch=1)
        note = QtWidgets.QLabel(
            "Sample properties not recorded by the run\n"
            "(kept as you step through files; leave blank to omit)"
        )
        note.setWordWrap(True)
        right.addWidget(note)

        form = QtWidgets.QFormLayout()
        self.metadata_fields: dict[str, QtWidgets.QLineEdit] = {}
        for cfg_field, label, unit in MANUAL_METADATA_FIELDS:
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(unit)
            self.metadata_fields[cfg_field] = edit
            form.addRow(label, edit)
        right.addLayout(form)

        self.reanalyze_button = QtWidgets.QPushButton("Re-analyze with these values")
        self.reanalyze_button.clicked.connect(self.reanalyze_current)
        right.addWidget(self.reanalyze_button)
        right.addStretch(1)

        button_bar = QtWidgets.QHBoxLayout()
        self.skip_button = QtWidgets.QPushButton("Skip")
        self.skip_button.clicked.connect(self.skip_current)
        self.accept_button = QtWidgets.QPushButton("Save && Next")
        self.accept_button.clicked.connect(self.accept_current)
        button_bar.addStretch(1)
        button_bar.addWidget(self.skip_button)
        button_bar.addWidget(self.accept_button)
        outer.addLayout(button_bar)

        self._set_controls_enabled(False)
        self.setCentralWidget(central)

    def _set_controls_enabled(self, enabled: bool):
        self.skip_button.setEnabled(enabled)
        self.accept_button.setEnabled(enabled and self.result is not None)
        self.reanalyze_button.setEnabled(enabled)

    # ---- Directory / file sequencing -----------------------------------
    def choose_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose directory containing harmonic-Hall CSV files"
        )
        if not directory:
            return

        input_dir = Path(directory)
        self.files = hh.find_data_files(input_dir)
        if not self.files:
            QtWidgets.QMessageBox.warning(
                self, "No files found", f"No .csv files found in {input_dir}"
            )
            return

        self.output_dir = input_dir / OUTPUT_SUBDIR_NAME
        self.dir_label.setText(f"{input_dir}  ->  {self.output_dir}")
        self.index = -1
        self.load_next_file()

    def load_next_file(self):
        self.index += 1
        if self.index >= len(self.files):
            self.status_label.setText("All files processed.")
            self.figure.clear()
            self.canvas.draw_idle()
            self.report_text.setPlainText("")
            self.result = None
            self._set_controls_enabled(False)
            return

        self.status_label.setText(
            f"File {self.index + 1} / {len(self.files)}: {self.files[self.index].name}"
        )
        self._set_controls_enabled(True)
        self.run_current()

    # ---- Config / analysis -----------------------------------------------
    def _build_config(self) -> hh.Config:
        path = self.files[self.index]
        cfg = hh.Config(csv_path=str(path), outdir=str(self.output_dir))
        for cfg_field, label, _unit in MANUAL_METADATA_FIELDS:
            text = self.metadata_fields[cfg_field].text().strip()
            if not text:
                continue
            if cfg_field == "nm_material":
                setattr(cfg, cfg_field, text)
                continue
            try:
                setattr(cfg, cfg_field, float(text))
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid value", f"'{text}' is not a number for {label}; ignoring."
                )
        return cfg

    def run_current(self):
        path = self.files[self.index]
        cfg = self._build_config()
        try:
            self.result = hh.run_analysis(cfg, fig=self.figure)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Analysis failed", f"{path.name}: {e}\nSkipping."
            )
            self.load_next_file()
            return

        self.canvas.draw_idle()
        self.report_text.setPlainText(self.result.rep.render())
        self.accept_button.setEnabled(True)

    def reanalyze_current(self):
        if self.index < 0 or self.index >= len(self.files):
            return
        self.run_current()

    # ---- Actions -----------------------------------------------------------
    def skip_current(self):
        self.load_next_file()

    def accept_current(self):
        if self.result is None:
            return
        hh.write_outputs(self.result)
        self.load_next_file()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = HarmonicHallWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
