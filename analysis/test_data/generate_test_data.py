"""Generate synthetic Hall-sweep files for exercising ahe_gui.py / ahe_core.py.

Writes a handful of pymeasure-style CSVs into this directory, each with a
different combination of ordinary Hall slope (R0), anomalous Hall resistance
(R_AHE), even-in-field contamination (simulating contact misalignment), and
noise -- including one deliberately unusable file to test the GUI's "Skip"
button. Re-run this script any time to regenerate them.
"""
from pathlib import Path

import numpy as np

HEADER_TEMPLATE = """\
#Procedure: <hall_measurement.HallMeasurement>
#Parameters:
#\tAverages per point: 5
#\tCompliance voltage: 2 V
#\tEnd field: {end_field} T
#\tField coefficient: 0.1 T/A
#\tField points: {n_points}
#\tSense current: {sense_current} A
#\tSettling time: 1 s
#\tStart field: {start_field} T
#Data:
"""

COLUMNS = (
    "Magnetic Field (T),Magnet Current (A),Hall Voltage (V),"
    "Hall Voltage Std (V),Hall Resistance (ohm)"
)


def write_sweep(
    path: Path, field, rxy, sense_current: float = 1e-3, std_fraction: float = 0.02,
    rng: np.random.Generator = None,
):
    rng = rng or np.random.default_rng(0)
    field_coefficient = 0.1
    current = field / field_coefficient
    voltage = rxy * sense_current
    std = np.abs(voltage) * std_fraction + 1e-7 * rng.random(field.size)

    header = HEADER_TEMPLATE.format(
        end_field=field.max(), n_points=field.size,
        sense_current=sense_current, start_field=field.min(),
    )
    with open(path, "w", newline="") as f:
        f.write(header)
        f.write(COLUMNS + "\n")
        for h, i, v, s, r in zip(field, current, voltage, std, rxy):
            f.write(f"{h},{i},{v},{s},{r}\n")


def main():
    out_dir = Path(__file__).parent

    # 1) Clean sweep, moderate anomalous Hall signal.
    rng = np.random.default_rng(1)
    field = np.linspace(-0.5, 0.5, 41)
    rxy = 120.0 * field + 3.5 * np.tanh(field / 0.05) + rng.normal(0, 0.05, field.size)
    write_sweep(out_dir / "sample_01_clean.csv", field, rxy, rng=rng)

    # 2) Clean sweep, wider field range, small anomalous Hall signal.
    rng = np.random.default_rng(2)
    field = np.linspace(-1.0, 1.0, 51)
    rxy = 80.0 * field + 1.2 * np.tanh(field / 0.08) + rng.normal(0, 0.03, field.size)
    write_sweep(out_dir / "sample_02_small_ahe.csv", field, rxy, rng=rng)

    # 3) Sweep with visible even-in-field contamination (contact misalignment),
    #    to show off what antisymmetrization removes.
    rng = np.random.default_rng(3)
    field = np.linspace(-0.6, 0.6, 41)
    rxy_true = 150.0 * field + 6.0 * np.tanh(field / 0.06)
    even_contamination = 8.0 + 4.0 * field ** 2  # even in H -- should vanish after antisymmetrizing
    rxy = rxy_true + even_contamination + rng.normal(0, 0.05, field.size)
    write_sweep(out_dir / "sample_03_misaligned_contacts.csv", field, rxy, rng=rng)

    # 4) Noisy sweep -- still fittable, but a good test of dragging a wide
    #    enough span to average out the noise.
    rng = np.random.default_rng(4)
    field = np.linspace(-0.5, 0.5, 41)
    rxy = 100.0 * field + 2.5 * np.tanh(field / 0.05) + rng.normal(0, 0.6, field.size)
    write_sweep(out_dir / "sample_04_noisy.csv", field, rxy, std_fraction=0.15, rng=rng)

    # 5) Garbage/bad measurement (no real field dependence, pure noise) --
    #    meant to be Skipped rather than fit.
    rng = np.random.default_rng(5)
    field = np.linspace(-0.5, 0.5, 41)
    rxy = rng.normal(0, 5.0, field.size)
    write_sweep(out_dir / "sample_05_bad_skip_me.csv", field, rxy, std_fraction=0.5, rng=rng)

    # 6) Full loop: up-sweep then down-sweep, with a hysteretic switching
    #    field that differs between branches (a soft-ferromagnet-like AHE
    #    loop). Both branches converge at saturation, but the loop shows
    #    visible hysteresis near the coercive field -- tests that the two
    #    sweep directions get told apart and plotted/antisymmetrized
    #    correctly rather than being scrambled together.
    rng = np.random.default_rng(6)
    up_field = np.linspace(-0.5, 0.5, 41)
    down_field = np.linspace(0.5, -0.5, 41)

    def rxy_of(field, switch_field):
        return 130.0 * field + 4.0 * np.tanh((field - switch_field) / 0.01)

    up_rxy = rxy_of(up_field, -0.05) + rng.normal(0, 0.05, up_field.size)
    down_rxy = rxy_of(down_field, 0.05) + rng.normal(0, 0.05, down_field.size)
    field = np.concatenate([up_field, down_field])
    rxy = np.concatenate([up_rxy, down_rxy])
    write_sweep(out_dir / "sample_06_full_loop.csv", field, rxy, rng=rng)

    print(f"Wrote 6 synthetic sweeps to {out_dir}")


if __name__ == "__main__":
    main()
