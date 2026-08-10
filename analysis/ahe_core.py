"""Core (GUI-free) analysis routines for anomalous Hall effect (AHE) measurements.

Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-30

Loads pymeasure-style measurement files, antisymmetrizes the Hall signal to
remove even-in-field (longitudinal mixing) contamination, fits the ordinary
Hall (linear) term over a user-chosen high-field range, and writes the
processed outputs. Kept independent of the GUI so it can also be driven from
a plain script or notebook.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Preference order for auto-detecting which columns to use. Add your own
# column names here if a file uses different headers.
FIELD_COLUMN_CANDIDATES = ["Magnetic Field (T)"]
VALUE_COLUMN_CANDIDATES = ["Hall Resistance (ohm)", "Hall Voltage (V)"]


@dataclass
class Measurement:
    """A single loaded raw measurement file."""

    path: Path
    header_lines: list[str]  # verbatim leading '#...' lines (pymeasure procedure/parameters)
    data: pd.DataFrame
    field_column: str
    value_column: str

    @property
    def field(self) -> np.ndarray:
        return self.data[self.field_column].to_numpy(dtype=float)

    @property
    def value(self) -> np.ndarray:
        return self.data[self.value_column].to_numpy(dtype=float)


@dataclass
class FitResult:
    fit_min: float  # lower bound of the fitted field range (T)
    fit_max: float  # upper bound of the fitted field range (T)
    slope: float  # ordinary Hall slope (value_column units per T)
    intercept: float  # anomalous Hall value at H=0 (value_column units)
    r_squared: float
    n_points: int


def find_data_files(directory: Path, extensions=(".csv", ".txt")) -> list[Path]:
    """Return measurement files directly inside `directory`, sorted by name."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def load_measurement(path: Path) -> Measurement:
    """Load a pymeasure-style CSV/TXT file: leading '#' comment lines
    (procedure name + parameters) followed by a normal CSV table.
    """
    header_lines = []
    with open(path, "r", newline="") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.rstrip("\n"))
            else:
                break

    data = pd.read_csv(path, comment="#")

    field_column = next((c for c in FIELD_COLUMN_CANDIDATES if c in data.columns), None)
    value_column = next((c for c in VALUE_COLUMN_CANDIDATES if c in data.columns), None)
    if field_column is None or value_column is None:
        raise ValueError(
            f"{path.name}: could not find a known field/value column. "
            f"Found columns: {list(data.columns)}"
        )

    return Measurement(
        path=path, header_lines=header_lines, data=data,
        field_column=field_column, value_column=value_column,
    )


def split_sweep_branches(field: np.ndarray) -> list[np.ndarray]:
    """Split a field sweep into contiguous monotonic runs (e.g. an up-sweep
    followed by a down-sweep, for a full hysteresis loop). Returns a list of
    index arrays covering the whole input in original order; a plain
    one-directional sweep returns a single array covering everything.

    Exact-repeat field values (zero step) are treated as continuing whatever
    direction is already established, so they don't spuriously start a new
    branch.
    """
    n = len(field)
    if n < 3:
        return [np.arange(n)]

    branches = []
    start = 0
    direction = 0  # +1 rising, -1 falling, 0 undetermined
    for i in range(1, n):
        step = field[i] - field[i - 1]
        if step == 0:
            continue
        new_direction = 1 if step > 0 else -1
        if direction == 0:
            direction = new_direction
        elif new_direction != direction:
            branches.append(np.arange(start, i))
            start = i
            direction = new_direction
    branches.append(np.arange(start, n))
    return branches


def _antisymmetrize_branch(field: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Antisymmetrize a single monotonic branch: [f(H) - f(-H)] / 2, with
    f(-H) from linear interpolation onto this branch's own curve.
    """
    order = np.argsort(field)
    field_sorted = field[order]
    value_sorted = value[order]

    lo, hi = field_sorted[0], field_sorted[-1]
    mirror = -field
    in_range = (mirror >= lo) & (mirror <= hi)

    value_at_mirror = np.full_like(value, np.nan, dtype=float)
    value_at_mirror[in_range] = np.interp(mirror[in_range], field_sorted, value_sorted)

    return (value - value_at_mirror) / 2.0


def antisymmetrize(field: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Antisymmetrize value(H) about H=0: [f(H) - f(-H)] / 2.

    f(-H) is obtained by linear interpolation, so the field grid doesn't need
    to be perfectly mirrored. Points whose mirror falls outside the measured
    range come back as NaN rather than being extrapolated.

    If the sweep contains multiple monotonic branches (e.g. a full loop:
    up-sweep then down-sweep), each branch is antisymmetrized against only
    its own points. Mixing branches together would silently pull the mirror
    lookup from whichever branch happens to be nearby in field value -- most
    visibly wrong right at the coercive/switching region, which is exactly
    where you'd want to visually judge data quality.
    """
    result = np.full_like(value, np.nan, dtype=float)
    for branch_idx in split_sweep_branches(field):
        result[branch_idx] = _antisymmetrize_branch(field[branch_idx], value[branch_idx])
    return result


def fit_ordinary_hall(
    field: np.ndarray, antisym_value: np.ndarray, fit_min: float, fit_max: float
) -> FitResult:
    """Fit a straight line to the antisymmetrized signal over field in
    [fit_min, fit_max] -- a single high-field tail (e.g. [0.3, 0.5] T for the
    positive tail), NOT a window applied to |H|. The line's H=0 intercept is
    the anomalous Hall value; its slope is the ordinary Hall coefficient.

    Fitting a single tail matters: because the antisymmetrized signal is an
    exactly odd function of H, folding both the positive and negative tails
    into one fit (e.g. via a mask on |H|) forces the intercept to cancel to
    ~0 by construction, silently discarding the anomalous Hall signal you're
    trying to measure.
    """
    mask = (field >= fit_min) & (field <= fit_max) & np.isfinite(antisym_value)
    n_points = int(mask.sum())
    if n_points < 2:
        raise ValueError("fit range contains fewer than 2 valid points")

    slope, intercept = np.polyfit(field[mask], antisym_value[mask], 1)

    model = slope * field[mask] + intercept
    residuals = antisym_value[mask] - model
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((antisym_value[mask] - np.mean(antisym_value[mask])) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FitResult(
        fit_min=fit_min, fit_max=fit_max, slope=float(slope), intercept=float(intercept),
        r_squared=float(r_squared), n_points=n_points,
    )


def anomalous_hall_loop(field: np.ndarray, antisym_value: np.ndarray, fit: FitResult) -> np.ndarray:
    """Ordinary-Hall-subtracted signal: antisym_value - slope * H.

    Removes just the linear background, leaving the anomalous Hall term
    (which should saturate to +-fit.intercept and trace out any hysteresis
    loop). The intercept itself is deliberately left in, since it's the
    loop's saturation value.
    """
    return antisym_value - fit.slope * field


# Colors used to tell sweep branches apart in antisymmetrized/loop plots --
# kept distinct from the raw curves' grays so the "analysis result" layers
# stand out. Add more entries if you ever expect more than two branches.
BRANCH_COLORS = ["C0", "C3"]


def branch_label(field: np.ndarray, idx: np.ndarray, n_branches: int, branch_number: int, prefix: str) -> str:
    """Human-readable legend label for one sweep branch, e.g. 'raw (up-sweep)'
    or, with more than two branches, 'raw (branch 2, down)'.
    """
    direction = "up" if field[idx[-1]] >= field[idx[0]] else "down"
    if n_branches <= 2:
        return f"{prefix} ({direction}-sweep)"
    return f"{prefix} (branch {branch_number}, {direction})"


def save_processed_csv(
    meas: Measurement, antisym_value: np.ndarray, fit: FitResult, output_dir: Path
) -> Path:
    """Write the original data plus antisymmetrized signal and fit
    model/residual columns to a new CSV in `output_dir`.

    Independent of the other save_* functions below -- comment out the call
    to this one (in the GUI's accept_current) if you don't want it.
    """
    out = meas.data.copy()
    out[f"{meas.value_column} Antisymmetrized"] = antisym_value

    model = fit.slope * meas.field + fit.intercept
    out["Ordinary Hall Fit"] = model
    out["Fit Residual"] = antisym_value - model

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / meas.path.name

    with open(out_path, "w", newline="") as f:
        for line in meas.header_lines:
            f.write(line + "\n")
        f.write(
            f"#Analysis: fit over H in [{fit.fit_min:.6g}, {fit.fit_max:.6g}] T, "
            f"slope={fit.slope:.6g}, intercept={fit.intercept:.6g}, R^2={fit.r_squared:.5f}\n"
        )
        out.to_csv(f, index=False)

    return out_path


def plot_analysis(
    fig, ax_raw, ax_loop, meas: Measurement, antisym_value: np.ndarray, fit: FitResult | None,
):
    """Draw the two-panel analysis view onto existing axes: left panel is
    raw (per branch) + antisymmetrized (per branch) + the fit line; right
    panel is the ordinary-Hall-subtracted anomalous Hall loop. Shared by the
    GUI's live canvas and the saved PNG so they always match.
    """
    field = meas.field
    branches = split_sweep_branches(field)

    ax_raw.clear()
    for i, idx in enumerate(branches):
        label = branch_label(field, idx, len(branches), i + 1, "raw")
        ax_raw.plot(
            field[idx], meas.value[idx], marker=".", linestyle="-",
            linewidth=0.7, color=("0.7" if i % 2 == 0 else "0.45"), label=label,
        )
    for i, idx in enumerate(branches):
        label = branch_label(field, idx, len(branches), i + 1, "antisym")
        ax_raw.plot(
            field[idx], antisym_value[idx], "o", markersize=4,
            color=BRANCH_COLORS[i % len(BRANCH_COLORS)], label=label,
        )

    if fit is not None:
        h_line = np.linspace(field.min(), field.max(), 200)
        ax_raw.plot(
            h_line, fit.slope * h_line + fit.intercept, "--", color="k", linewidth=1,
            label=f"fit: slope={fit.slope:.4g}, R_AHE={fit.intercept:.4g}",
        )
        ax_raw.axvspan(fit.fit_min, fit.fit_max, color="k", alpha=0.08)

    ax_raw.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    ax_raw.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax_raw.set_xlabel(meas.field_column)
    ax_raw.set_ylabel(meas.value_column)
    ax_raw.set_title(meas.path.name)
    ax_raw.legend(fontsize=7, loc="best")

    ax_loop.clear()
    if fit is not None:
        loop_value = anomalous_hall_loop(field, antisym_value, fit)
        for i, idx in enumerate(branches):
            label = branch_label(field, idx, len(branches), i + 1, "loop")
            ax_loop.plot(
                field[idx], loop_value[idx], marker=".", linestyle="-",
                color=BRANCH_COLORS[i % len(BRANCH_COLORS)], label=label,
            )
        ax_loop.axhline(fit.intercept, color="0.6", linestyle=":", linewidth=1)
        ax_loop.axhline(-fit.intercept, color="0.6", linestyle=":", linewidth=1)
        ax_loop.legend(fontsize=7, loc="best")
        ax_loop.set_title("Anomalous Hall loop (ordinary Hall subtracted)")
    else:
        ax_loop.text(
            0.5, 0.5, "Select a fit range\nto see the AHE loop", ha="center", va="center",
            transform=ax_loop.transAxes, color="0.5",
        )
        ax_loop.set_title("Anomalous Hall loop")

    ax_loop.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    ax_loop.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax_loop.set_xlabel(meas.field_column)
    ax_loop.set_ylabel(f"{meas.value_column} (background subtracted)")

    fig.tight_layout()


def save_plot(
    meas: Measurement, antisym_value: np.ndarray, fit: FitResult, output_dir: Path
) -> Path:
    """Save a PNG with two panels: raw + antisymmetrized data with the fit
    line, and the resulting anomalous Hall loop (ordinary Hall subtracted).

    Independent of the other save_* functions -- comment out the call to
    this one (in the GUI's accept_current) if you don't want per-file plots.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless backend for saving, independent of the GUI's own canvas
    import matplotlib.pyplot as plt

    fig, (ax_raw, ax_loop) = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_analysis(fig, ax_raw, ax_loop, meas, antisym_value, fit)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{meas.path.stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


SUMMARY_COLUMNS = [
    "file", "field_column", "value_column", "fit_min_T", "fit_max_T",
    "slope", "intercept_anomalous_hall", "r_squared", "n_fit_points",
]


def append_summary_row(
    meas: Measurement, fit: FitResult, output_dir: Path, summary_name: str = "summary.csv"
) -> Path:
    """Append one row (per accepted file) to a batch-wide summary CSV.

    Independent of the other save_* functions -- comment out the call to
    this one (in the GUI's accept_current) if you don't want the summary file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / summary_name
    is_new = not summary_path.exists()

    row = {
        "file": meas.path.name,
        "field_column": meas.field_column,
        "value_column": meas.value_column,
        "fit_min_T": fit.fit_min,
        "fit_max_T": fit.fit_max,
        "slope": fit.slope,
        "intercept_anomalous_hall": fit.intercept,
        "r_squared": fit.r_squared,
        "n_fit_points": fit.n_points,
    }

    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    return summary_path
