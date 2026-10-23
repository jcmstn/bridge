# Architecture & code map

The cross-cutting picture of `bridge`: how the layers fit together, what
contract each layer exposes, and which files to touch for a given change.
Read this once before working anywhere in the tree.

Companion documents (this file does **not** repeat them):

- [`../README.md`](../README.md) — install, run commands.
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
| `uv run python bridge_tui.py` | The TUI menu — every DC / MFLI / SOT program as a card (description, collapsible wiring schematic, Launch) in three suite columns, plus the shared "Recent runs" table; quitting a program returns to the menu — Textual |
| `uv run python sot/sot_pulsed_switching_tui.py` | SOT pulsed switching — one form, two toggles: write pulse = 4200A PMU \| 6221, read = DC R_xy \| lock-in harmonic (6221 AC + MFLI). The four combinations are `SOTPS` / `SOT2H` / `SOT1I` (the former programs) and `SOTPD` (6221 Pulse Delta write + Delta read, 2182A on the 6221's RS-232) (see §2 "Merged programs"). `sot_pulsed_switching_2h_tui.py` / `sot_pulsed_switching_6221_tui.py` still run and open this form on their mode — Textual |
| `uv run python sot/sot_nonlocal_switching_tui.py` | Nonlocal spin-current switching, 6221 + 2182A only (optional Kepco field initialization, then a sweep of single-lobe 0 → ±I → 0 WAVE write pulses + DC nonlocal read, current-reversal averaged unless switched off; type `NLSW`) — Textual. `sot/sot_nonlocal_switching.py` also runs standalone (plain CSV) |
| `uv run python dc/dc_rt_log_tui.py` | R vs T log — fixed ±I reversal (6221 + 2182) sampled back to back while the temperature drifts freely (optionally toward one iTC target sent at start — no wait); MercuryiTC read before and after each sample (`temperature_N_drift_K` per point), one R-vs-T panel per reading sensor; stops on Stop / max duration / a T crossing, Stop saves as `completed` (type `RT`) — Textual |
| `uv run python tools/curate_sample.py <sample>` | Post-hoc curation TUI: mark runs `paper_include` / `figure_ref` |

Each measurement's `*_tui.py` is also runnable on its own
(`uv run python dc/dc_iv_curve_tui.py`). `tools/` is scripts only — it is
not a package and nothing imports from it.

See the README for `uv sync` and prerequisites.


## 2. The three-layer pattern (the important part)

Every measurement exists as **two files** on top of the shared drivers, with
a strict one-way import direction:

```
{suite}/{name}_tui.py         Textual TUI            ─┐
      imports run_measurement + configs from ↓       │ imports
{suite}/{name}.py             hardware + physics    ─┤ never
      imports drivers from ↓                         │ upward
instruments/*.py             shared drivers        ─┘
```

**`tui → measurement → instruments`, never backwards.** Breaking this
direction (e.g. a measurement script importing Textual) is always a bug.

### What the TUI module exports

These names are pure (no Textual dependency) — the program's parameter
surface, used by the Textual form, `bridge_tui.py`, plain scripts and tests:

| Name | What it is |
|------|-----------|
| `DEFAULTS` | dict of every form field's default value |
| `NUMERIC_FIELDS`, `TEXT_FIELDS`, `OPTIONAL_NUMERIC_FIELDS`, `LIST_FIELDS` | field-name groups + per-field validation metadata |
| `MEASUREMENT_TYPE` | the locked type code (`"HALL"`, `"IV"`, …) — see `data_convention.md` |
| `MeasurementPlan` | frozen dataclass: one parsed, validated run request |
| `resolve_state(state) -> dict` | the derived values of a parsed form (sweep lists, value lists, their parse errors) — called by the TUI's `parse_state()` |
| `build_summary(state) -> (info, warnings, errors)` | the live sidebar text + warnings/errors, computed from the parsed state. Each `info` line is `"Key: value — note"` (`instruments/summary_lines.py`): the sidebar shows the value bold, key and note muted |
| `build_plan(state, data_root) -> MeasurementPlan` | the validated run request; a single-run program allocates its run here (`plan.run_ctx`) |
| `run_plan(plan, stop_event, *, on_status, on_run_label, on_point, on_run_finished, run_contexts, run_extras)` | the whole run — connect, loop over the series (one `record_run()` per output file), teardown. The TUI `RunScreen` runs exactly this; it fills `run_contexts` / `run_extras` for the post-run status/comment prompt |
| `save_run_png(plan, records, png_path, comment="")` + `PNG_SUFFIX` | the per-run PNG |
| `build_header_fields(plan, ctx, …) -> dict` | the `# key: value` CSV header for this run |
| `compute_filename_preview(state) -> str` | placeholder filename for the live preview (calls `preview_raw_filename`, never `allocate_run`) |
| `parse_sensor_uids(text)` | MercuryiTC sensor-UID parsing, shared |
| `TEMPERATURE_*` field groups, `temperature_summary`, `temperature_cfg_from_state` | the MercuryiTC log + setpoint-control fields, merged into the module's own groups from `instruments/temperature_control.py` (see §3a) |
| `{NAME}_DESCRIPTION` | one- or two-line blurb — instruments + anything non-obvious, not what the measurement is — shown on the program's `bridge_tui.py` card and (DC/SOT) its sidebar |
| `{NAME}_SCHEMATIC` | the wiring schematic shown on the program's `bridge_tui.py` card |

If you add a form field, it goes in `DEFAULTS` + the right `*_FIELDS`
group + `MeasurementPlan` **once**, in the TUI module, and the form,
summary and plan pick it up.

### Merged programs (a mode toggle over several engines)

Two forms put former near-duplicate programs behind a toggle. Each mode
still runs its **own engine module** unchanged, so type codes, raw columns,
headers, filenames and PNG names are exactly what each program wrote before:

| Form (module) | Toggle | Mode → engine module (type code) |
|---|---|---|
| `mfli/mfli_dual_harmonic_tui.py` | `ac_source` | `mfli` → itself (`HARM`); `6221` → `mfli_dual_harmonic_6221_tui` (`HARM6`) |
| `sot/sot_pulsed_switching_tui.py` | `pulse_source` × `read_mode` | `pmu`+`dc` → itself (`SOTPS`); `pmu`+`harmonic` → `sot_pulsed_switching_2h_tui` (`SOT2H`); `6221`+`harmonic` → `sot_pulsed_switching_6221_tui` (`SOT1I`); `6221`+`dc` → `sot_pulsed_switching_pdelta_tui` (`SOTPD`) |

How it fits together:
- The form's `DEFAULTS` / `*_FIELDS` are the union of the engines'; the
  mode-only cards are hidden (`.display`), and `mode_errors()` drops parse errors of hidden fields.
- `resolve_state` / `build_summary` / `compute_filename_preview` /
  `build_plan` dispatch on the toggle; `engine(plan)` returns the module
  whose `run_plan` / `build_header_fields` / `save_run_png` /
  `MEASUREMENT_TYPE` / `RunScreen` handle that plan (the base
  `MeasurementApp.run_screen()` uses it).
- Every form id appears once: a field two modes share lives in one card
  that stays visible (the TUI reads widgets by id, so a hidden twin would
  override the visible one).
- A key that means different things in two engines gets its own form id,
  renamed back before the engine sees the state (SOT: `wave_pulse_width_s`,
  `lock_settle_s` — `_FORM_IDS`).
- Settings: the form also reads the former programs' `*_tui_settings.json`
  key by key; until it has been saved with the toggle, the toggle follows
  the program used last (newest file). The engine modules keep their
  `SETTINGS_PATH` for exactly this.
- An engine module has no `App`; its `main()` opens the merged form on its
  mode.

**Lock-in toggle (MFLI | SR830).** Both forms also have a `lockin` select
(SOT: harmonic read only). It is not another engine: the active mode's own
plan gets `sr830_cfgs` (a tuple of `sr830.LockinConfig`, None = MFLI) plus
`header_extra["lockin"] = "SR830"`, built by the form's `build_plan`
dispatcher from the shared form fields (`sr830.config_from_form`; TC /
sensitivity / buffer rate snapped to the unit's tables first, so the summary
and estimate see the applied values). The engine's `run_plan` connects an
`sr830.SR830Read` instead of the MFLI and passes it as
`run_measurement(..., lockin=reader)`; the loop swaps only its lock-in
calls (acquire, lock check, frequency/phase/filter metadata). `lockin=None`
is the unchanged MFLI path. Same type codes and columns. DIFFR (needs a DC
offset on the source) and NOISE (wideband streaming) have no SR830 mode.

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
  text. The base `do_run()` calls the module's `run_plan()` ), saves each run's PNG via `save_run_png`, offers the
  status/comment prompt for the last run (`finish_last_run()` with the
  stored extras) and writes the session's row into `runs.db`
  (`instruments/run_index.py`).

### Data root (changed 2026-09-03)

`_DEFAULT_DATA_DIR = <repo>/../data` is now only a **fallback**. Each run's
actual data root comes from the "Data root" field in the identity bar
(`instruments/data_dir.py` `DataDirPickerScreen` + `validate_directory`),
persisted per-TUI in a
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

Callers (TUI, a plain script, a test) build `points`, connect the
instruments, call `run_measurement`, then run their own
`shutdown_*` path — the function never connects or disconnects hardware
itself.

### 3a. MercuryiTC setpoints — an outer loop around the whole program

Every series program (Hall, IV, gate sweep, spin valve, DIFFR, HARM/HARM6,
SOTPS/SOT2H/SOT1I/SOTPD, NLSW) has the shared temperature card
(`tui_common.temperature_card`):
log switch, software max T, and optionally "Drive iTC to setpoint(s)" with a
setpoint list, settle margin (mK) and hold time (s). With setpoints on,
`instruments/temperature_control.run_temperature_steps` repeats the **whole
program** once per setpoint, as if Start had been pressed at each T:

1. `go_to_temperature`: own short iTC session → `set_temperature` (TSET +
   heater auto — the iTC's own PID; PID values stay on its touch screen) →
   `wait_for_temperature_stable` (within ±margin for `hold_s` *unbroken*,
   no timeout; the run screen's **Skip T wait** button / `s` key starts now)
   → close.
2. `build_plan({**state, "temperature_setpoint_K": T})` (step 0: the plan
   Start already built — `with_first_setpoint` makes the first setpoint the
   form's filename setpoint), then that engine's own `run_plan` — its
   connect / series loop / teardown are untouched, so between steps the
   sources are off and the magnet is at 0.
3. Records are re-tagged: `series_index` unique across the session
   (`finish_last_run` relies on it), `series_label` prefixed `T=…K`.

Nothing ever turns the heater off at the end — the last setpoint stays on
the iTC. The one exception is the software max
(`TemperatureControllerConfig.max_temperature_K`, default 320 K): every
`read_temperature` / wait reading at or above it calls `heater_off` (auto
off + manual 0 %) and raises `TemperatureLimitError`, ending the run with
an error. R vs T (`MeasurementApp.TEMPERATURE_STEPS = False`) takes one
target, sent by its `run_plan` at start, no wait. NOISE has no control.


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
| `mercury_itc.py` | Oxford MercuryiTC temperature controller | `MercuryITC` class; `TemperatureControllerConfig` (+ `max_temperature_K`), `connect_temperature_controller`, `read_temperature`, `set_temperature`, `wait_for_temperature_stable`, `enforce_max_temperature`, `shutdown_temperature_controller` (closes only — the setpoint stays) | **hand-written** (pymeasure has ITC 503 only) |
| `keithley4200a.py` | Keithley 4200A-SCS — the 4225-PMU behind a 4225-RPM, over KXCI (the SMU cards are not driven; a general SMU wrapper existed until 2026-09-23 and is in git history) | `Keithley4200AConfig`, `connect_4200a`, `shutdown_4200a`, `PMUPulseConfig`, `configure_pmu_pulse`, `pulse_once`, `list_user_libraries` | **hand-written** KXCI line protocol (not SCPI); runs the KULT module in `instruments/kult/` |
| `mfli_daq.py` | Zurich Instruments MFLI (dual, via MDS) | `connect`, `connect_device`, `setup_mds`, `sync_follower_oscillator`, `acquire_averaged` | wraps `zhinst-core` |
| `sr830.py` | SRS SR830 lock-in (MFLI alternative; one input + one demod per unit, so two units for 1f/2f or Rxy/Rxx) | `LockinConfig`, `connect`, `acquire_averaged`, `acquire_averaged_pair` (internal data buffer), `wait_for_reference_lock` / `check_reference_locked` (LIAS bit 3), `auto_gain` / `auto_reserve` / `auto_phase`, `settle_time_s`, `acquire_s`, `shutdown` | thin wrapper over pymeasure + raw manual commands |

**Failure policy — deliberate, not accidental:**

- Kepco magnet and Lake Shore 475 are *load-bearing* in any field-swept
  measurement — a connection failure there is a real error and stops the run.
- MercuryiTC is *nice-to-have*: `connect_temperature_controller()` and
  `read_temperature()` never raise for "not connected" or "only one probe
  wired" — a measurement that doesn't otherwise need the iTC is never
  interrupted by it. Passing `temp_ctrl=None` just leaves the temperature
  columns blank. Two deliberate exceptions: driving a setpoint is
  load-bearing (an unreachable iTC is an error once setpoints are on), and
  a reading at/above the software max T raises `TemperatureLimitError`
  after forcing the heater off.

### Adding an instrument — the driver contract

A new instrument is one new file, `instruments/{name}.py`, imported directly
by whatever measurement/script needs it. **Nothing else registers it** — no
TUI, no menu. The module exposes plain module-level
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


## 6. Run history (`instruments/run_index.py`)

SQLite run history at a **fixed** path (`<repo>/../data/runs.db`),
deliberately independent of any run's chosen data root, so history is always
findable. Written by every TUI `RunScreen` (`tui_common` `do_run`) and shown
in the `bridge_tui.py` "Recent runs" table. Pure sqlite; short-lived,
always-closed connection per statement; WAL mode.


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
6. **`tests/`** — `test_{name}_tui.py` for the parse/plan/summary logic (see §9).


## 8. "Change X → touch these files"

| Change | Files |
|--------|-------|
| A form field's default / range / label | `{suite}/{name}_tui.py` only (`DEFAULTS` + `*_FIELDS` + `MeasurementPlan`) — the form, summary and plan all read it |
| The measurement loop / what's recorded | `{suite}/{name}.py` `run_measurement()`; check `build_header_fields` / plot code for new columns |
| How a run is named or saved | `instruments/data_naming.py` + `data_convention.md` (locked tables) |
| A program's "Estimated run time" / live ETA | `{suite}/{name}_tui.py` `run_costs()` (one `RunCost`: seconds per point, shown by `build_summary` **and** fed to the progress bar via `progress_total`/`progress_step`). Per-instrument helpers sit beside their instrument (`magnet_move_s`, `read_field_s`, `read_time_s`, `reversal_avg_s`, `poll_window_s`); the best-guess hardware latencies (`GPIB_TXN_S`, `READ_2182_FACTOR`, `LOCK_TYP_S`, `PMU_PULSE_S`, …) are the knobs at the top of `instruments/run_time.py` — tune there. Only the field-settle pair (`FIELD_SETTLE_EXTRA_S` + `FIELD_SETTLE_S_PER_A` × last ramp step) is fitted to rig data (ChipL HARM6 runs 68-102); the rest are unmeasured. A new sleep in a loop means a new term in its `run_costs()` |
| Current-reversal averaging | `instruments/keithley6221.py` `acquire_reversal_averaged_voltage` + `current-reversal.md` |
| An instrument's SCPI / connect / shutdown behaviour | `instruments/{instr}.py` only |
| Run-history behaviour | `instruments/run_index.py` (written by `tui_common` `do_run`, shown by `bridge_tui.py`) |
| TUI live-plot window (spawned matplotlib process fed over a Queue) | plot body: the TUI's `_live_plot_worker`; process launch: `instruments/live_plot.py` `start_live_plot` — always launch through it, Textual's `sys.stderr.fileno() == -1` breaks a bare `mp.Queue()` |
| The data-root picker | `instruments/data_dir.py` |
| Menu card text or schematic | `{NAME}_DESCRIPTION` / `{NAME}_SCHEMATIC` in the TUI module; card title in `bridge_tui.py` `PROGRAMS` |
| Anything every TUI form / run screen does the same way (settings file, sample picker, identity bar, abort/back, status/comment prompt, run history, run-screen CSS) | `instruments/tui_common.py` only |
| How a program runs (connect, series loop, teardown, per-run finalize) | its `run_plan()` in `{suite}/{name}_tui.py` — the `RunScreen` calls it; the per-run record/finalize block is `data_naming.record_run()` |
| A mode of a merged program (HARM/HARM6, SOTPS/SOT2H/SOT1I/SOTPD) | its engine module for the run itself; the toggle, card visibility and settings fallback in the merged form's module |
| The SR830 lock-in mode (HARM/HARM6/SOT2H/SOT1I) | form → `sr830_cfgs()` / `_sr830_checks()` in `mfli/mfli_dual_harmonic_tui.py` or `sot/sot_pulsed_switching_tui.py`; connect/teardown in the engine `*_tui.py` `run_plan`; per-point reads at the `lockin` branches in the engine `run_measurement`; instrument side `instruments/sr830.py` (`SR830Read`, `config_from_form`) |
| MercuryiTC setpoint control (outer loop, settle wait, Skip, max T) | `instruments/temperature_control.py` (fields, summary, step runner) + `instruments/mercury_itc.py` (commands, wait, max-T cutoff); TUI side: `tui_common` (`temperature_card`, `do_run`, Skip button) |
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
