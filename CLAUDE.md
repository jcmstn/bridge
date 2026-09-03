# CLAUDE.md

`bridge` — instrument control for DC transport + MFLI lock-in measurements
in a spintronics lab (Textual TUI and NiceGUI web front ends).

## Read first

- [`docs/architecture.md`](docs/architecture.md) — layout, the three-layer
  pattern, the `run_measurement()` contract, "change X → touch these files".
- [`docs/data_convention.md`](docs/data_convention.md) — run naming/saving +
  the `instruments/data_naming.py` API.
- [`docs/current-reversal.md`](docs/current-reversal.md) — V_odd / V_even.
- Every module's top-of-file docstring is its API reference (wiring diagram
  + usage example). Prefer reading those over guessing.

## Hard rules

- **Import direction is one-way:** `web/{suite}/{name}.py` →
  `{suite}/{name}_tui.py` → `{suite}/{name}.py` → `instruments/*.py`.
  Never the other way. A measurement script must not import Textual; a TUI
  must not import `nicegui`. The web page reuses the TUI module's pure
  helpers (`DEFAULTS`, `*_FIELDS`, `MeasurementPlan`, `build_summary`, …) —
  don't re-implement the parameter surface.
- **Read raw run files with `data_naming.read_raw(path)`**, never
  `pd.read_csv(path, comment="#")` (that eats the units sub-header row).
- **One `allocate_run()` per output file.** Multi-file sweeps call it once
  per loop iteration with a fresh writer closure — never reuse a
  `RunContext` across iterations.
- **Never invent a type code or key-axis `kind`** outside the locked tables
  in `instruments/data_naming.py` (and `docs/data_convention.md`).
- **`finalize_index_row()` unconditionally** with the outcome status the
  instant a run ends, before any optional user status/comment prompt.
- MercuryiTC is optional by design — `temp_ctrl=None` is never an error.
  Kepco magnet + Lake Shore 475 are load-bearing in a field sweep.
- A run's data root comes from the identity bar "Data root" field
  (`MeasurementPlan.data_root`); `_DEFAULT_DATA_DIR` (`<repo>/../data`) is
  only the fallback.

## Tests

`uv run pytest` — pure logic only (parse/plan/summary, `data_naming`, sweep
utils, kepco settle). No hardware or VISA layer is exercised.
