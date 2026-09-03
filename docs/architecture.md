# Architecture & code map

The cross-cutting picture of `bridge`: how the layers fit together, what
contract each layer exposes, and which files to touch for a given change.
Read this once before working anywhere in the tree.

Companion documents (this file does **not** repeat them):

- [`../README.md`](../README.md) — install, run commands, `BRIDGE_WEB_PORT`,
  Windows WinNAT troubleshooting.
- [`data_convention.md`](data_convention.md) — how runs are named and saved,
  and the full `instruments/data_naming.py` API.
- [`current-reversal.md`](current-reversal.md) — the V_odd / V_even
  current-reversal decomposition every DC program uses.

Every module also carries a long module-level docstring with a wiring
diagram and a runnable usage example — those **are** the per-module API
reference. This file is the map, not a second copy of them.


## 1. Entry points

| Command | Opens |
|---------|-------|
| `uv run python dc/dc_tui.py`   | DC suite picker (Hall, I–V, gate sweep, spin-valve) — Textual |
| `uv run python mfli/mfli_tui.py` | MFLI suite picker (dual-harmonic, diff-resistance, phase calibration) — Textual |
| `uv run python web/app.py`     | Browser front end, same 7 measurements — NiceGUI, `http://localhost:8080` |
| `uv run python tools/curate_sample.py <sample>` | Post-hoc curation TUI: mark runs `paper_include` / `figure_ref` |

Each measurement's `*_tui.py` is also runnable on its own
(`uv run python dc/dc_iv_curve_tui.py`). `tools/` is scripts only — it is
not a package and nothing imports from it.

See the README for `uv sync` and prerequisites.


## 2. The three-layer pattern (the important part)

Every measurement exists as **three files** with a strict one-way import
direction:

```
web/{suite}/{name}.py        NiceGUI page          ─┐
      imports pure helpers from ↓                    │ imports
{suite}/{name}_tui.py         Textual TUI            ─┤ never
      imports run_measurement + configs from ↓       │ upward
{suite}/{name}.py             hardware + physics    ─┤
      imports drivers from ↓                         │
instruments/*.py             shared drivers        ─┘
```

**`web → tui → measurement → instruments`, never backwards.** The web page
does not re-implement the parameter surface; it imports it from the TUI
module. Breaking this direction (e.g. a measurement script importing
Textual, or a TUI importing `nicegui`) is always a bug.

### What the TUI module exports for the web page to reuse

These names are pure (no Textual/NiceGUI dependency) and are imported
verbatim by the matching `web/{suite}/{name}.py`:

| Name | What it is |
|------|-----------|
| `DEFAULTS` | dict of every form field's default value |
| `NUMERIC_FIELDS`, `TEXT_FIELDS`, `OPTIONAL_NUMERIC_FIELDS`, `LIST_FIELDS` | field-name groups + per-field validation metadata |
| `MEASUREMENT_TYPE` | the locked type code (`"HALL"`, `"IV"`, …) — see `data_convention.md` |
| `MeasurementPlan` | frozen dataclass: one parsed, validated run request |
| `build_summary(state) -> list[str]` | the live sidebar text + warnings/errors, computed from raw field values |
| `build_header_fields(plan, ctx, …) -> dict` | the `# key: value` CSV header for this run |
| `compute_filename_preview(state) -> str` | placeholder filename for the live preview (calls `preview_raw_filename`, never `allocate_run`) |
| `parse_sensor_uids(text)` | MercuryiTC sensor-UID parsing, shared |
| `{NAME}_DESCRIPTION` | one-paragraph blurb, shown in both the suite picker and the sidebar |

If you add a form field, it goes in `DEFAULTS` + the right `*_FIELDS`
group + `MeasurementPlan` **once**, in the TUI module, and both front ends
pick it up.

### Data root (changed 2026-09-03)

`_DEFAULT_DATA_DIR = <repo>/../data` is now only a **fallback**. Each run's
actual data root comes from the "Data root" field in the identity bar
(TUI: `instruments/data_dir.py` `DataDirPickerScreen` + `validate_directory`;
web: `web/directory_picker.py`), persisted per-TUI in a
`*_tui_settings.json` next to `_DEFAULT_DATA_DIR`. `MeasurementPlan.data_root`
carries the resolved choice into the run. (`data_convention.md` still
describes the old fixed `_DATA_DIR` computation — treat that as the
fallback path only.)


## 3. The `run_measurement()` contract

Every DC/MFLI measurement module exposes one orchestrator with this shape
(`dc/dc_hall_measurement.py`, `dc/dc_iv_curve.py`, `dc/dc_gate_sweep.py`,
`dc/dc_spin_valve.py`, `mfli/mfli_dual_harmonic.py`,
`mfli/mfli_diff_resistance_vs_bias.py`):

```python
def run_measurement(
    <instrument handles>,          # e.g. source, voltmeter  (already connected)
    <cfg dataclasses>,             # e.g. src_cfg, acq_cfg
    points: list[<Point>],         # the sweep, built by the caller
    stop_event: threading.Event | None = None,   # checked before every point (and mid-reversal); set it to break early and still return partial data
    on_point: Callable[[dict], None] | None = None,  # called with each record dict right after it is appended — live progress without polling the CSV
    gaussmeter=None, gauss_cfg=None,     # optional: measure real field per point instead of leaving it unset
    temp_ctrl=None,  temp_cfg=None,      # optional: log sample/probe temperature; None is never a reason to stop
    write_csv: Callable[[list[dict]], None] | None = None,  # optional: replaces the plain headerless to_csv() with a data-convention writer (see instruments/data_naming.make_incremental_writer)
) -> pandas.DataFrame                    # all recorded data; the CSV is rewritten in full after every point (crash-safe)
```

Callers (TUI, web, a plain script, a test) build `points`, connect the
instruments, call `run_measurement`, then run their own
`shutdown_*` path — the function never connects or disconnects hardware
itself.

**The one exception:** `mfli/mfli_phase_calibration.py` does not fit the
`points -> DataFrame` shape. Its orchestrator is
`run_phase_calibration(...) -> PhaseCalibrationReport`, and it takes a
third callback `on_status` alongside `stop_event` / `on_point`.
`format_report(report)` renders it for display.


## 4. Instrument layer (`instruments/`)

Each module wraps one instrument as a `*Config` dataclass plus
`connect_*` / `shutdown_*` (and sometimes `acquire_*` / `set_*`) free
functions. The measurement scripts only ever touch these wrappers, not the
raw driver classes.

| Module | Instrument | Config / helpers | Driver source |
|--------|-----------|------------------|---------------|
| `keithley6221.py` | Keithley 6221 current source | `SourceConfig`, `connect_source`, `shutdown_source`, `acquire_reversal_averaged_voltage`, low-level `connect()` | thin wrapper over pymeasure |
| `keithley2182.py` | Keithley 2182 nanovoltmeter | `VoltmeterConfig`, `connect_voltmeter`, `acquire_averaged_voltage` | thin wrapper over pymeasure |
| `keithley2400.py` | Keithley 2400 gate source | `GateConfig`, `connect_gate`, `set_gate_voltage`, `shutdown_gate` | thin wrapper over pymeasure |
| `keithley2450.py` + `keithley2450Buffer.py` | Keithley 2450 SourceMeter | `Keithley2450` class (native-2450 SCPI + buffer) | **hand-written** (pymeasure's 2450 lacks native buffer support) |
| `kepco_magnet.py` | Kepco BOP-GL bipolar supply → electromagnet | `KepkoBOPGL` class; `MagnetConfig`, `connect_magnet`, `set_magnet_current`, `shutdown_magnet` | **hand-written** on pymeasure `Instrument`/`SCPIMixin` |
| `lakeshore475.py` | Lake Shore 475 DSP Gaussmeter | `LakeShore475` class; `GaussmeterConfig`, `connect_gaussmeter`, `read_field_mT`, `shutdown_gaussmeter` | **hand-written** (pymeasure has 421/425, not 475) |
| `mercury_itc.py` | Oxford MercuryiTC temperature controller | `MercuryITC` class; `TemperatureControllerConfig`, `connect_temperature_controller`, `read_temperature`, `shutdown_temperature_controller` | **hand-written** (pymeasure has ITC 503 only) |
| `mfli_daq.py` | Zurich Instruments MFLI (dual, via MDS) | `connect`, `connect_device`, `setup_mds`, `sync_follower_oscillator`, `acquire_averaged` | wraps `zhinst-core` |

**Failure policy — deliberate, not accidental:**

- Kepco magnet and Lake Shore 475 are *load-bearing* in any field-swept
  measurement — a connection failure there is a real error and stops the run.
- MercuryiTC is *nice-to-have*: `connect_temperature_controller()` and
  `read_temperature()` never raise for "not connected" or "only one probe
  wired" — a measurement that doesn't otherwise need the iTC is never
  interrupted by it. Passing `temp_ctrl=None` just leaves the temperature
  columns blank.


## 5. Data output

Runs are saved per-sample, with folders encoding only the sample and
everything else (temperature, angle, status, timestamp) living in the
filename, a `# key: value` CSV header, and a per-sample `index.csv`.

Full spec and the writer/reader API (`ensure_sample`, `allocate_run`,
`make_incremental_writer`, `write_record`, **`read_raw`** — not
`pd.read_csv(comment="#")`, `finalize_index_row`, `proc_path`,
`format_axis_token`, `preview_raw_filename`) is in
[`data_convention.md`](data_convention.md). Read it before writing anything
that produces or consumes a run file.


## 6. Web-only machinery (`web/`)

The NiceGUI front end adds infrastructure the standalone scripts don't need:

| Module | Role |
|--------|------|
| `web/app.py` | entrypoint; registers all 7 pages; `reload=False` on purpose (a file-watch restart would drop the run lock + live instrument connections mid-measurement) |
| `web/run_manager.py` | **global** run lock (`RunHandle`) — only one measurement app-wide, because the magnet / gaussmeter / iTC are the same physical instruments shared by both suites. Also buffers live records/log so a fresh page load can repaint an in-progress run and abort it. |
| `web/run_index.py` | SQLite run history at a **fixed** path (`<repo>/../data/runs.db`), deliberately independent of any run's chosen data root, so history is always findable. Short-lived connection per statement; WAL mode. |
| `web/run_controller.py` | the shared page engine: form → parsed state → config dataclasses → background thread runs `run_measurement()` with `stop_event`/`on_point` → live updates over one `queue.Queue` drained per `ui.timer` tick. Each page supplies only the page-specific callables. |
| `web/identity_bar.py` | the sample / device / cooldown / temperature-setpoint / data-root fields + filename preview, built once, used by all 7 pages |
| `web/directory_picker.py` | server-side local-filesystem directory browser (safe: localhost-only, no auth) |
| `web/sample_picker.py` | NiceGUI sample picker + "+ New sample" + post-run status/comment dialogs |

`validate_directory()` lives in `instruments/data_dir.py` (pure) and is
re-exported by `web/directory_picker.py`, so the Textual TUIs get the same
rule without importing NiceGUI.


## 7. Worked examples

### 7a. Drive one measurement from a plain script (no TUI)

```python
from instruments.keithley6221 import SourceConfig, connect_source, shutdown_source
from instruments.keithley2182 import VoltmeterConfig, connect_voltmeter
from dc.dc_hall_measurement import AcquisitionConfig, FieldPoint, run_measurement

src_cfg  = SourceConfig(visa_resource="GPIB0::20::INSTR", sense_current_A=1e-4)
volt_cfg = VoltmeterConfig(visa_resource="GPIB0::7::INSTR", nplc=5)
acq_cfg  = AcquisitionConfig(settling_time_s=1.0, n_reversals=3, output_file="hall.csv")

source    = connect_source(src_cfg)
voltmeter = connect_voltmeter(volt_cfg)
try:
    points = [FieldPoint()]        # one reversal-averaged Hall reading, no field sweep
    df = run_measurement(
        source, voltmeter, src_cfg, acq_cfg, points,
        on_point=lambda rec: print(rec["hall_voltage_V"]),
    )
finally:
    shutdown_source(source)        # voltmeter needs no shutdown
print(df.head())
```

To sweep the field, give each `FieldPoint` a `magnet_current_A` **and** a
`set_action` that calls `set_magnet_current(...)`, connect the Kepco magnet
+ Lake Shore 475, and pass `gaussmeter=` / `gauss_cfg=` so the field axis
comes from the measured field, not the magnet current — see
`dc_hall_measurement.py`'s `main()` for the full pattern.

### 7b. Read a run back

```python
import pandas as pd
from instruments.data_naming import read_raw

runs = pd.read_csv("data/A/index.csv")                 # the searchable per-sample database
hall = runs[(runs["type"] == "HALL") & (runs["status"] == "good")]

df = read_raw("data/A/raw/A_0007_HB3_HALL_T010K_20260811T143022.csv")
# df columns keep their name_unit labels, e.g. "hall_voltage_V"
```

Never read a raw file with `pd.read_csv(path, comment="#")` — it eats the
units sub-header row as data. Use `read_raw()`.

### 7c. Add a new measurement — file-by-file checklist

1. **`{suite}/{name}.py`** — wiring diagram + physics in the module
   docstring; `*Config` dataclasses; a `run_measurement(...)` matching the
   §3 contract; a `plot_results()`; a `main()` for standalone use.
2. **`instruments/`** — only if a new instrument is involved: add a
   `{instr}.py` with `{Instr}Config` + `connect_*` / `shutdown_*`.
3. **`data_convention.md`** — add the new type code to the locked table
   (and a key-axis `kind` to `data_naming.py` if the run has a new fixed
   secondary axis). Nowhere else needs to know the code.
4. **`{suite}/{name}_tui.py`** — `DEFAULTS`, the `*_FIELDS` groups,
   `MeasurementPlan`, `build_summary`, `build_header_fields`,
   `compute_filename_preview`, `{NAME}_DESCRIPTION`, and the Textual `App`.
5. **`{suite}/{suite}_tui.py`** — register the new `App` + its schematic in
   the suite picker.
6. **`web/{suite}/{name}.py`** — import the pure names from step 4; supply
   the page-specific callables to `RunController`.
7. **`web/app.py`** — register the new page.
8. **`tests/`** — `test_{name}_tui.py` for the parse/plan/summary logic and
   `test_web_{name}.py` for the page's state→config mapping (see §9).


## 8. "Change X → touch these files"

| Change | Files |
|--------|-------|
| A form field's default / range / label | `{suite}/{name}_tui.py` only (`DEFAULTS` + `*_FIELDS` + `MeasurementPlan`) — both front ends inherit it |
| The measurement loop / what's recorded | `{suite}/{name}.py` `run_measurement()`; check `build_header_fields` / plot code for new columns |
| How a run is named or saved | `instruments/data_naming.py` + `data_convention.md` (locked tables) |
| Current-reversal averaging | `instruments/keithley6221.py` `acquire_reversal_averaged_voltage` + `current-reversal.md` |
| An instrument's SCPI / connect / shutdown behaviour | `instruments/{instr}.py` only |
| Live-plot / run-lock / run-history behaviour (web) | `web/run_controller.py` / `web/run_manager.py` / `web/run_index.py` |
| The identity bar / data-root picker | `web/identity_bar.py` + `web/directory_picker.py`; TUI side `instruments/data_dir.py` |
| Suite picker text or schematic | `{suite}/{suite}_tui.py` and the `{NAME}_DESCRIPTION` in the TUI module |


## 9. Tests

```
uv run pytest
```

No hardware and no VISA layer is touched. The suite covers the pure logic:

- `test_data_naming.py` — run allocation, header/index round-trips,
  `read_raw`, axis tokens.
- `test_dc_*` / `test_mfli_*_tui.py` — field parsing, `MeasurementPlan`
  construction, `build_summary` warnings/errors, filename previews.
- `test_web_*.py` — each page's form-state → config-dataclass mapping
  matches its TUI.
- `test_kepco_settle.py` — the magnet current/field settle wait.
- `test_dc_gate_sweep.py`, `dc_sweep_utils` coverage — `linear_sweep`
  bidirectional shape, `parse_value_list`.

Anything requiring a real 6221/2182/MFLI/magnet is manual bench testing.
