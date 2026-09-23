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
| `uv run python bridge_tui.py` | The TUI menu — every DC / MFLI / SOT program as a card (description, collapsible wiring schematic, Launch) in three suite columns, plus the shared "Recent runs" table; mirrors the web landing page. Quitting a program returns to the menu — Textual |
| `uv run python sot/sot_pulsed_switching_tui.py` | SOT pulsed switching — one form, two toggles: write pulse = 4200A PMU \| 6221 WAVE, read = DC R_xy (6221 ±I + 2182) \| lock-in harmonic (6221 AC + MFLI). The three valid combinations are the former programs `SOTPS` / `SOT2H` / `SOT1I` (see §2 "Merged programs"). `sot_pulsed_switching_2h_tui.py` / `sot_pulsed_switching_6221_tui.py` still run and open this form on their mode — Textual |
| `uv run python sot/sot_nonlocal_switching_tui.py` | Nonlocal spin-current switching, 6221 + 2182A only (optional Kepco field initialization, then a sweep of single-lobe 0 → ±I → 0 WAVE write pulses + DC nonlocal read, current-reversal averaged unless switched off; type `NLSW`) — Textual, also a web page. `sot/sot_nonlocal_switching.py` also runs standalone (plain CSV) |
| `uv run python web/app.py`     | Browser front end — every DC and MFLI measurement plus SOT pulsed switching (`/sot/pulsed-switching`, the same Pulse × Read toggles as the TUI) and nonlocal switching — NiceGUI, `http://localhost:8080`. The dual-harmonic page carries the MFLI \| 6221 AC-source toggle (`/mfli/dual-harmonic?source=6221`; the old `/mfli/dual-harmonic-6221` URL redirects there) |
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
| `resolve_state(state) -> dict` | the derived values of a parsed form (sweep lists, value lists, their parse errors) — called by the TUI's `parse_state()` and the web's `form_state()` alike |
| `build_summary(state) -> (info, warnings, errors)` | the live sidebar text + warnings/errors, computed from the parsed state |
| `build_plan(state, data_root) -> MeasurementPlan` | the validated run request; a single-run program allocates its run here (`plan.run_ctx`) |
| `run_plan(plan, stop_event, *, on_status, on_run_label, on_point, on_run_finished, run_contexts, run_extras)` | the whole run — connect, loop over the series (one `record_run()` per output file), teardown. Both the TUI `RunScreen` and the web page run exactly this; it fills `run_contexts` / `run_extras` for the post-run status/comment prompt |
| `save_run_png(plan, records, png_path, comment="")` + `PNG_SUFFIX` | the per-run PNG, same in both front ends |
| `build_header_fields(plan, ctx, …) -> dict` | the `# key: value` CSV header for this run |
| `compute_filename_preview(state) -> str` | placeholder filename for the live preview (calls `preview_raw_filename`, never `allocate_run`) |
| `parse_sensor_uids(text)` | MercuryiTC sensor-UID parsing, shared |
| `{NAME}_DESCRIPTION` | one-paragraph blurb, shown on the program's `bridge_tui.py` card, its web page and (DC/SOT) its sidebar |
| `{NAME}_SCHEMATIC` | the wiring schematic shown on the program's `bridge_tui.py` card |

If you add a form field, it goes in `DEFAULTS` + the right `*_FIELDS`
group + `MeasurementPlan` **once**, in the TUI module, and both front ends
pick it up.

### Merged programs (a mode toggle over several engines)

Two forms put former near-duplicate programs behind a toggle. Each mode
still runs its **own engine module** unchanged, so type codes, raw columns,
headers, filenames and PNG names are exactly what each program wrote before:

| Form (module) | Toggle | Mode → engine module (type code) |
|---|---|---|
| `mfli/mfli_dual_harmonic_tui.py` | `ac_source` | `mfli` → itself (`HARM`); `6221` → `mfli_dual_harmonic_6221_tui` (`HARM6`) |
| `sot/sot_pulsed_switching_tui.py` (+ `web/sot/pulsed_switching.py`) | `pulse_source` × `read_mode` | `pmu`+`dc` → itself (`SOTPS`); `pmu`+`harmonic` → `sot_pulsed_switching_2h_tui` (`SOT2H`); `6221`+`harmonic` → `sot_pulsed_switching_6221_tui` (`SOT1I`); `6221`+`dc` is blocked by the summary |

How it fits together:
- The form's `DEFAULTS` / `*_FIELDS` are the union of the engines'; the
  mode-only cards are hidden (`.display` in the TUI, `bind_visibility_from`
  on the web), and `mode_errors()` drops parse errors of hidden fields.
- `resolve_state` / `build_summary` / `compute_filename_preview` /
  `build_plan` dispatch on the toggle; `engine(plan)` returns the module
  whose `run_plan` / `build_header_fields` / `save_run_png` /
  `MEASUREMENT_TYPE` / `RunScreen` handle that plan (the base
  `MeasurementApp.run_screen()` and `web/run_controller._engine()` use it).
- Every form id appears once: a field two modes share lives in one card
  that stays visible (the TUI reads widgets by id, so a hidden twin would
  override the visible one). On the web, a mode select's value is a string
  ("6221") and must stay one — `form_state()` takes select values as typed.
- A key that means different things in two engines gets its own form id,
  renamed back before the engine sees the state (SOT: `wave_pulse_width_s`,
  `lock_settle_s` — `_FORM_IDS`).
- Settings: the form also reads the former programs' `*_tui_settings.json`
  key by key; until it has been saved with the toggle, the toggle follows
  the program used last (newest file). The engine modules keep their
  `SETTINGS_PATH` for exactly this.
- An engine module has no `App`; its `main()` opens the merged form on its
  mode.

### Shared TUI scaffolding (`instruments/tui_common.py`)

Every TUI's `App` subclasses `MeasurementApp` and its `RunScreen`
subclasses `MeasurementRunScreen`; the form widgets (`field`,
`switch_field`, `select_field`, `sweep_rows_field`, `card`,
`identity_bar`), `format_si`, `parse_sensor_uids` and `LogRelay` live
there too. A program keeps only what is its own:

- **App:** `TITLE`/`SUB_TITLE`/`CSS`, `compose()` (starting with
  `identity_bar(...)`), `update_summary()`, `_build_plan()` (usually
  `build_plan(state, self.data_root)`), and `SWITCH_DEPENDENTS` (switch id
  → the widget ids it greys out). `parse_state()` is generic: every
  `*_FIELDS` Input (non-finite numbers are errors), every `Switch` /
  `Select` / `TextArea` by id, the sample, then the module's
  `resolve_state()`. `refresh_summary()` wraps `update_summary()` so an
  exception shows up in the sidebar (and disables Start) instead of closing
  the app. Settings save/load is generic too, so a new widget is persisted
  just by having an `id`. The base reads `SETTINGS_PATH`,
  `_DEFAULT_DATA_DIR`, the `*_FIELDS` groups, `resolve_state`,
  `build_summary` and `RunScreen` from the program's own module at call
  time.
- **RunScreen:** only display hooks — `TABLE_COLUMNS` / `table_row(record)`,
  `live_plot_args()`, and `ABORT_LABEL` / `ABORT_STATUS` / `POINT_STATUS`
  text. The base `do_run()` calls the module's `run_plan()` (the same one
  the web page runs), saves each run's PNG via `save_run_png`, offers the
  status/comment prompt for the last run (`finish_last_run()` with the
  stored extras) and writes the session's row into `runs.db`
  (`instruments/run_index.py`) — the same history the web front end writes.

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
`mfli/mfli_dual_harmonic_6221.py`, `mfli/mfli_diff_resistance_vs_bias.py`):

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
| `keithley6221.py` | Keithley 6221 current source | **DC:** `SourceConfig`, `connect_source`, `shutdown_source`, `acquire_reversal_averaged_voltage`, low-level `connect()`. **AC/WAVE + phase marker:** `ACSourceConfig`, `connect_ac_source`, `shutdown_ac_source`. **Single hardware-timed pulse (WAVE square, 1 cycle — NOT Pulse Delta, no 2182 needed):** `PulseWaveConfig`, `fire_wave_pulse` | thin wrapper over pymeasure |
| `keithley2182.py` | Keithley 2182 nanovoltmeter | `VoltmeterConfig` (`channel` 1/2), `connect_voltmeter`, `acquire_averaged_voltage` | thin wrapper over pymeasure |
| `keithley2400.py` | Keithley 2400 SourceMeter | **gate:** `GateConfig`, `connect_gate`, `set_gate_voltage`, `shutdown_gate`. **general SMU:** `SMUConfig`, `connect_smu`, `set_source_level`, `read_measurement`, `acquire_measurement`, `measure_buffered`, `shutdown_smu` | thin wrapper over pymeasure |
| `keithley2450.py` | Keithley 2450 SourceMeter | `Keithley2450` class (native-2450 SCPI + `defbuffer1` stats, private `_Keithley2450Buffer` mixin in-file); general SMU wrapper — same `SMUConfig` / `connect_smu` / … surface as `keithley2400.py`, swap by import | **hand-written** (pymeasure's 2450 lacks native buffer support) |
| `kepco_magnet.py` | Kepco BOP-GL bipolar supply → electromagnet | `KepkoBOPGL` class; `MagnetConfig`, `connect_magnet`, `set_magnet_current`, `shutdown_magnet` | **hand-written** on pymeasure `Instrument`/`SCPIMixin` |
| `lakeshore475.py` | Lake Shore 475 DSP Gaussmeter | `LakeShore475` class; `GaussmeterConfig`, `connect_gaussmeter`, `read_field_mT`, `shutdown_gaussmeter` | **hand-written** (pymeasure has 421/425, not 475) |
| `mercury_itc.py` | Oxford MercuryiTC temperature controller | `MercuryITC` class; `TemperatureControllerConfig`, `connect_temperature_controller`, `read_temperature`, `shutdown_temperature_controller` | **hand-written** (pymeasure has ITC 503 only) |
| `keithley4200a.py` | Keithley 4200A-SCS — the 4225-PMU behind a 4225-RPM, over KXCI (the SMU cards are not driven; a general SMU wrapper existed until 2026-09-23 and is in git history) | `Keithley4200AConfig`, `connect_4200a`, `shutdown_4200a`, `PMUPulseConfig`, `configure_pmu_pulse`, `pulse_once`, `list_user_libraries` | **hand-written** KXCI line protocol (not SCPI); runs the KULT module in `instruments/kult/` |
| `mfli_daq.py` | Zurich Instruments MFLI (dual, via MDS) | `connect`, `connect_device`, `setup_mds`, `sync_follower_oscillator`, `acquire_averaged` | wraps `zhinst-core` |

**Failure policy — deliberate, not accidental:**

- Kepco magnet and Lake Shore 475 are *load-bearing* in any field-swept
  measurement — a connection failure there is a real error and stops the run.
- MercuryiTC is *nice-to-have*: `connect_temperature_controller()` and
  `read_temperature()` never raise for "not connected" or "only one probe
  wired" — a measurement that doesn't otherwise need the iTC is never
  interrupted by it. Passing `temp_ctrl=None` just leaves the temperature
  columns blank.

### Adding an instrument — the driver contract

A new instrument is one new file, `instruments/{name}.py`, imported directly
by whatever measurement/script needs it. **Nothing else registers it** — no
TUI, no web, no menu. The module exposes plain module-level
functions (not a class API — the class, if any, stays private):

1. **`{Name}Config`** — a `@dataclass`. First field `visa_resource: str`,
   then every knob the instrument needs, each with a default. This *is* the
   parameter surface; a caller constructs one and passes it around.

2. **`connect_{name}(cfg: {Name}Config) -> Handle`** — open the VISA
   session, `reset()`, apply `cfg`, return a live handle (a pymeasure
   instrument, or a hand-written driver instance). Log one line on success.

3. **`shutdown_{name}(handle[, cfg]) -> None`** — put the instrument in a
   safe state and close. Take `cfg` too only if teardown needs it (e.g. the
   magnet ramps down using `cfg.ramp_step_A`). A *nice-to-have* shutdown
   accepts `handle=None` and returns (`shutdown_temperature_controller`);
   a *load-bearing* one assumes a live handle and is a plain VISA call with
   no internal try/except (`shutdown_magnet`, `shutdown_gaussmeter`) —
   callers wrap it in `dc_sweep_utils.safe_shutdown()`, one guard per call,
   so one failing teardown never skips the rest.

4. *(optional)* **`set_{name}(handle, cfg, value) -> None | dict`** — apply
   a setpoint. Return a provenance `dict` if the caller should log how it
   went (see `set_magnet_current` → `{"field_settled": ..., ...}`),
   else `None`.

5. *(optional)* **`read_{name}(handle, cfg) -> float`** for one scalar in
   canonical units (`read_field_mT`), or a fixed-width tuple when the
   instrument has several probes/channels (`read_temperature ->
   (t1, t2)`); or **`acquire_{name}(handle, n, stop_event=None) -> dict`**
   returning `{"mean": ..., "sem": ...}` for an averaged reading.
   `stop_event: threading.Event | None` is checked between samples so a UI
   abort can interrupt a long average.

**Failure policy — decide per instrument, and write it in the docstring:**

- *Load-bearing* (the measurement is meaningless without it — magnet,
  gaussmeter): `connect_*` / `set_*` **raise** on any failure. The caller
  does not pass `None` for these.
- *Nice-to-have* (logging only — the iTC): `connect_*` returns `None` when
  the **hardware** can't be reached (never raises for that), and every
  other function accepts that `None` and no-ops (`read_* -> (None, None)`,
  `shutdown_*` returns immediately). A malformed **config** still raises
  (`connect_temperature_controller` rejects a `sensor_uids` that isn't 1–2
  long). A `set_*` is load-bearing even on a nice-to-have instrument — you
  only drive a setpoint because a measurement depends on it reaching it
  (see `set_temperature`).

**Driver class:** wrap a pymeasure driver directly if one fits — no
subclass, just call it in `connect_*` (`keithley2182.py`:
`Keithley2182(cfg.visa_resource)` then configure). Hand-write a private
`class {Name}(Instrument, SCPIMixin)` with `control`/`measurement`
properties only when pymeasure has no driver or a wrong one
(`lakeshore475.py`, `kepco_magnet.py`, `mercury_itc.py`); keep it in the
same file, keep it private.

**Skeleton** — thin wrapper over a real pymeasure driver, the common case
(drop in `instruments/newmeter.py`, rename):

```python
"""New Meter — connect/shutdown/read helpers. <wiring diagram + example>."""
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from pymeasure.instruments.keithley import Keithley2000   # ← the real driver

log = logging.getLogger(__name__)


@dataclass
class NewMeterConfig:
    visa_resource: str = "GPIB0::15::INSTR"
    nplc: float = 1.0


def connect_newmeter(cfg: NewMeterConfig) -> Keithley2000:
    dev = Keithley2000(cfg.visa_resource)
    dev.reset()
    dev.measure_voltage()
    dev.voltage_nplc = cfg.nplc          # ... apply the rest of cfg ...
    log.info("New Meter connected: %s  NPLC=%.1f", cfg.visa_resource, cfg.nplc)
    return dev


def read_newmeter(dev: Keithley2000, cfg: NewMeterConfig) -> float:
    return float(dev.voltage)


def shutdown_newmeter(dev: Keithley2000) -> None:
    dev.shutdown()
```

**Recorded-column naming:** when a reading lands in a measurement's
`record` dict, suffix the column with its unit from
`data_naming._COLUMN_UNITS` (`field_mT`, `voltage_V`, `temperature_K`) so
`write_record()` splits it into OriginLab's Units sub-header row. A column
with no known suffix keeps its full name and a blank unit — harmless, just
not unit-tagged.

The full "no-UI standalone script" wiring (this contract + the data
convention) is in [`../examples/custom_program.py`](../examples/custom_program.py).


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
| `web/app.py` | entrypoint; registers every page (DC, MFLI, SOT pulsed + nonlocal switching); `reload=False` on purpose (a file-watch restart would drop the run lock + live instrument connections mid-measurement) |
| `web/run_manager.py` | **global** run lock (`RunHandle`) — only one measurement app-wide, because the magnet / gaussmeter / iTC are the same physical instruments shared by both suites. Also buffers live records/log so a fresh page load can repaint an in-progress run and abort it. |
| `instruments/run_index.py` | SQLite run history at a **fixed** path (`<repo>/../data/runs.db`), deliberately independent of any run's chosen data root, so history is always findable. Written by **both** front ends (web `RunController`, TUI `RunScreen`); lives in `instruments/` because it is pure sqlite. Short-lived, always-closed connection per statement; WAL mode. |
| `web/run_controller.py` | the shared page engine: `RunController` runs the program's `run_plan()` in a background thread and drains live points/log from one `queue.Queue` per `ui.timer` tick (`on_tick` pushes the plot + table once per tick, not per point). Page helpers every page uses: `form_state()` (the web twin of `parse_state()`), `program_run_fn()` / `program_artifacts()` / `run_png_path()`, `prompt_last_run()` (status/comment → `finish_last_run()`), `finished_handler()`, `load_settings()` / `save_settings()`, `refresh_on_busy_change()` (the summary is rebuilt on input, the timer only tracks busy/idle). |
| `web/identity_bar.py` | the sample / device / cooldown / temperature-setpoint / data-root fields + filename preview, built once, used by every page |
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

The snippet above writes a **plain headerless CSV** (the `main()`s in
`dc/` / `mfli/` do too). For a standalone script that also saves into the
per-sample data convention — `ensure_sample` → `allocate_run` →
`make_incremental_writer` → an unconditional `finalize_index_row` — and for
the "roll your own loop with a new instrument driver" case, see
[`../examples/custom_program.py`](../examples/custom_program.py).

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
   `MeasurementPlan`, `resolve_state`, `build_summary`, `build_plan`,
   `run_plan` (one `data_naming.record_run()` per output file),
   `build_header_fields`, `save_run_png` (+ `PNG_SUFFIX`),
   `compute_filename_preview`, `{NAME}_DESCRIPTION`, `{NAME}_SCHEMATIC`,
   and the Textual `App` + `RunScreen` as subclasses of
   `instruments/tui_common.py`'s `MeasurementApp` / `MeasurementRunScreen`
   (see §2). If it is a variant of an existing program (same form, other
   instrument), consider a mode of that form instead (§2 "Merged programs").
5. **`bridge_tui.py`** — add a `Program(...)` (key, title, description,
   schematic, App) to its suite in `PROGRAMS`.
6. **`web/{suite}/{name}.py`** — name it so its basename matches NO module in the
   top-level `{suite}/` package: `python web/app.py` puts `web/` first on
   `sys.path`, so `web/sot/foo.py` shadows `sot/foo.py` (circular import). The
   convention `{suite}/{suite}_{name}.py` ↔ `web/{suite}/{name}.py` avoids it.
   Import the pure names from step 4; build the state with `form_state()`,
   the plan with `build_plan()`, and run it with `program_run_fn()`. Build the layout with
   `measurement_layout()` from `web/run_controller.py`: identity bar into
   `regions.identity`, the param/stable grids into `regions.params`,
   `summary_box` + start button into `regions.summary`, and status / plot /
   table / log into `regions.output` — the plot wrapped in a
   `ui.element("div").style("aspect-ratio: 1 / <n_panels>")` with no fixed
   Plotly `height`.
7. **`web/app.py`** — register the new page.
8. **`tests/`** — `test_{name}_tui.py` for the parse/plan/summary logic and
   `test_web_{name}.py` for the page's state→config mapping (see §9).


## 8. "Change X → touch these files"

| Change | Files |
|--------|-------|
| A form field's default / range / label | `{suite}/{name}_tui.py` only (`DEFAULTS` + `*_FIELDS` + `MeasurementPlan`) — both front ends inherit it |
| The measurement loop / what's recorded | `{suite}/{name}.py` `run_measurement()`; check `build_header_fields` / plot code for new columns |
| How a run is named or saved | `instruments/data_naming.py` + `data_convention.md` (locked tables) |
| A program's "Estimated run time" / live ETA | `{suite}/{name}_tui.py` `run_costs()` (one `RunCost`: seconds per point, shown by `build_summary` **and** fed to the progress bar via `progress_total`/`progress_step`; web banner via `RunController(run_cost=…)`). Per-instrument helpers sit beside their instrument (`magnet_move_s`, `read_field_s`, `read_time_s`, `reversal_avg_s`, `poll_window_s`); the best-guess hardware latencies (`GPIB_TXN_S`, `READ_2182_FACTOR`, `LOCK_TYP_S`, `PMU_PULSE_S`, …) are the knobs at the top of `instruments/run_time.py` — unmeasured, tune there. A new sleep in a loop means a new term in its `run_costs()` |
| Current-reversal averaging | `instruments/keithley6221.py` `acquire_reversal_averaged_voltage` + `current-reversal.md` |
| An instrument's SCPI / connect / shutdown behaviour | `instruments/{instr}.py` only |
| Live-plot / run-lock / run-history behaviour (web) | `web/run_controller.py` / `web/run_manager.py` / `instruments/run_index.py` |
| TUI live-plot window (spawned matplotlib process fed over a Queue) | plot body: the TUI's `_live_plot_worker`; process launch: `instruments/live_plot.py` `start_live_plot` — always launch through it, Textual's `sys.stderr.fileno() == -1` breaks a bare `mp.Queue()` |
| Web page layout skeleton (top band + params/output columns, square plot) | `web/run_controller.py` `measurement_layout()` — every page fills `regions.identity` / `.params` / `.summary` / `.output`; the app-wide colour theme is `web/app.py` (`app.colors()` + shared head CSS) |
| The identity bar / data-root picker | `web/identity_bar.py` + `web/directory_picker.py`; TUI side `instruments/data_dir.py` |
| Menu card text or schematic | `{NAME}_DESCRIPTION` / `{NAME}_SCHEMATIC` in the TUI module; card title in `bridge_tui.py` `PROGRAMS` |
| Anything every TUI form / run screen does the same way (settings file, sample picker, identity bar, abort/back, status/comment prompt, run history, run-screen CSS) | `instruments/tui_common.py` only |
| How a program runs (connect, series loop, teardown, per-run finalize) | its `run_plan()` in `{suite}/{name}_tui.py` — both front ends call it; the per-run record/finalize block is `data_naming.record_run()` |
| A mode of a merged program (HARM/HARM6, SOTPS/SOT2H/SOT1I) | its engine module for the run itself; the toggle, card visibility and settings fallback in the merged form's module (+ its web page for HARM) |
| The sweep-size cap (a form error instead of a frozen form) | `dc/dc_sweep_utils.py` `MAX_SWEEP_POINTS` / `check_sweep_size` |


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
- `test_run_time.py`, `test_run_costs_*.py` — the run-time model: helper arithmetic, per-program `run_costs()` defaults / multiplicity, plan `run_cost` length == `total_points`.
- `test_dc_gate_sweep.py`, `dc_sweep_utils` coverage — `linear_sweep`
  bidirectional shape, `parse_value_list`.
- `test_tui_smoke.py` — every program App (and every mode of a merged
  one) mounts headless, round-trips through its settings file, and turns a
  huge sweep into a form error instead of freezing; `test_tui_common.py` —
  the shared run-screen lifecycle; `test_run_plans.py` — every
  `run_plan()` end to end with the hardware faked (one finalized run per
  series value, error status + teardown on failure);
  `test_merged_programs.py` — the merged forms' legacy-settings fallback;
  `test_bridge_tui.py` — the menu; `test_run_index.py` — the run history.
  `conftest.py` points `run_index` and the merged forms' legacy settings
  paths at temp files for every test.

Anything requiring a real 6221/2182/MFLI/magnet is manual bench testing.
