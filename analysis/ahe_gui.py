"""PySide6 GUI for batch anomalous Hall effect (AHE) analysis.

Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-30

Point it at a directory of raw Hall-sweep files; it steps through them one at
a time. For each file, drag across a single high-field tail of the plot
(e.g. the saturated region on the positive side) to fit the ordinary Hall
(linear) term over that range -- the fit's H=0 intercept is the anomalous
Hall value. Don't drag a span that straddles H=0 or covers both tails: the
antisymmetrized signal is an exactly odd function of H, so mixing both tails
into one fit cancels the anomalous Hall intercept to ~0.  "Accept & Next"
saves the processed outputs into an `analyzed/` subdirectory next to the
input directory and moves on. "Skip" leaves the file untouched and moves on.

Run with: python ahe_gui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PySide6 import QtWidgets

from analysis import ahe_core

OUTPUT_SUBDIR_NAME = "analyzed"


class AHEAnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AHE Analysis")
        self.resize(1000, 750)

        self.files: list[Path] = []
        self.index: int = -1
        self.output_dir: Path | None = None
        self.measurement: ahe_core.Measurement | None = None
        self.antisym_value: np.ndarray | None = None
        self.current_fit: ahe_core.FitResult | None = None
        self._last_fit_range: tuple[float, float] | None = None

        self._build_ui()

    # ---- UI construction ----------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        top_bar = QtWidgets.QHBoxLayout()
        self.choose_dir_button = QtWidgets.QPushButton("Choose input directory...")
        self.choose_dir_button.clicked.connect(self.choose_directory)
        self.dir_label = QtWidgets.QLabel("No directory selected")
        top_bar.addWidget(self.choose_dir_button)
        top_bar.addWidget(self.dir_label, stretch=1)
        layout.addLayout(top_bar)

        self.status_label = QtWidgets.QLabel(
            "Choose a directory containing raw Hall sweep files to begin."
        )
        layout.addWidget(self.status_label)

        self.figure = Figure(figsize=(10, 4.5))
        self.ax = self.figure.add_subplot(121)
        self.ax_loop = self.figure.add_subplot(122)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, stretch=1)

        self.span_selector = SpanSelector(
            self.ax, self.on_span_select, direction="horizontal",
            useblit=True, interactive=True,
            props=dict(alpha=0.2, facecolor="C1"),
        )

        self.result_label = QtWidgets.QLabel("")
        layout.addWidget(self.result_label)

        button_bar = QtWidgets.QHBoxLayout()
        self.skip_button = QtWidgets.QPushButton("Skip")
        self.skip_button.clicked.connect(self.skip_current)
        self.accept_button = QtWidgets.QPushButton("Accept && Next")
        self.accept_button.clicked.connect(self.accept_current)
        button_bar.addStretch(1)
        button_bar.addWidget(self.skip_button)
        button_bar.addWidget(self.accept_button)
        layout.addLayout(button_bar)

        self._set_controls_enabled(False)
        self.setCentralWidget(central)

    def _set_controls_enabled(self, enabled: bool):
        self.skip_button.setEnabled(enabled)
        self.accept_button.setEnabled(enabled and self.current_fit is not None)

    # ---- Directory / file sequencing -----------------------------------
    def choose_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose directory containing raw Hall measurement files"
        )
        if not directory:
            return

        input_dir = Path(directory)
        self.files = ahe_core.find_data_files(input_dir)
        if not self.files:
            QtWidgets.QMessageBox.warning(
                self, "No files found", f"No .csv/.txt files found in {input_dir}"
            )
            return

        self.output_dir = input_dir / OUTPUT_SUBDIR_NAME
        self.dir_label.setText(f"{input_dir}  ->  {self.output_dir}")
        self.index = -1
        self._last_fit_range = None
        self.load_next_file()

    def load_next_file(self):
        self.index += 1
        if self.index >= len(self.files):
            self.status_label.setText("All files processed.")
            self.ax.clear()
            self.ax_loop.clear()
            self.canvas.draw_idle()
            self.current_fit = None
            self._set_controls_enabled(False)
            return

        path = self.files[self.index]
        try:
            self.measurement = ahe_core.load_measurement(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Failed to load file", f"{path.name}: {e}\nSkipping."
            )
            self.load_next_file()
            return

        self.current_fit = None
        self.antisym_value = ahe_core.antisymmetrize(
            self.measurement.field, self.measurement.value
        )

        self.status_label.setText(
            f"File {self.index + 1} / {len(self.files)}: {path.name}  -- "
            "drag across ONE high-field tail (not straddling H=0) to fit"
        )
        self.result_label.setText("")
        self._redraw(fit=None)
        self._set_controls_enabled(True)

        # Convenience: re-apply the previously used fit range to the new
        # file so you don't have to redrag every time it's the same. You can
        # still drag over it to override for this file.
        if self._last_fit_range is not None:
            self._try_fit(*self._last_fit_range)

    # ---- Plotting --------------------------------------------------------
    def _redraw(self, fit: ahe_core.FitResult | None):
        # Shared with save_plot() so the live view and the saved PNG always
        # show the same thing.
        ahe_core.plot_analysis(
            self.figure, self.ax, self.ax_loop, self.measurement, self.antisym_value, fit
        )
        self.canvas.draw_idle()

    # ---- Fitting -----------------------------------------------------------
    def on_span_select(self, xmin: float, xmax: float):
        fit_min, fit_max = sorted((xmin, xmax))
        self._try_fit(fit_min, fit_max)

    def _try_fit(self, fit_min: float, fit_max: float):
        try:
            fit = ahe_core.fit_ordinary_hall(
                self.measurement.field, self.antisym_value, fit_min, fit_max
            )
        except ValueError as e:
            self.result_label.setText(f"Fit failed: {e}")
            self.current_fit = None
            self.accept_button.setEnabled(False)
            return

        self.current_fit = fit
        self._last_fit_range = (fit_min, fit_max)
        self.result_label.setText(
            f"Fit over H in [{fit.fit_min:.4g}, {fit.fit_max:.4g}] T "
            f"({fit.n_points} pts): slope = {fit.slope:.6g}, "
            f"anomalous Hall (H=0 intercept) = {fit.intercept:.6g}, R^2 = {fit.r_squared:.5f}"
        )
        self._redraw(fit=fit)
        self.accept_button.setEnabled(True)

    # ---- Actions -----------------------------------------------------------
    def skip_current(self):
        self.load_next_file()

    def accept_current(self):
        meas = self.measurement
        fit = self.current_fit
        if fit is None:
            return

        # Each of these three outputs is independent -- comment out any call
        # you don't want.
        ahe_core.save_processed_csv(meas, self.antisym_value, fit, self.output_dir)
        ahe_core.save_plot(meas, self.antisym_value, fit, self.output_dir)
        ahe_core.append_summary_row(meas, fit, self.output_dir)

        self.load_next_file()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = AHEAnalysisWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
