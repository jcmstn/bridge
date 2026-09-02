# Data storage convention

How every measurement in `bridge` saves its data, and the API a new
measurement script should use to do the same. The implementation lives in
`instruments/data_naming.py` (naming/header/index core), plus two thin UI
helpers, `instruments/tui_sample_picker.py` (Textual) and
`web/sample_picker.py` (NiceGUI). Read the docstrings in those three files
for the authoritative details; this document is the narrative overview and
the "how do I wire up a new script" recipe.

## Principle

Folders encode only what is **physically permanent** — the sample.
Everything else (temperature, field angle, cooldown, run status,
timestamp) lives in the filename, in a comment header inside the CSV
itself, and in a per-sample `index.csv`. Nothing is ever inferred by
parsing a folder path.

## Folder tree

Exactly three levels, per sample, under a `data_root` (a sibling directory
of the `bridge` repo, never inside the git-tracked tree):

```
data_root/{sample}/sample.yaml   stack, geometry, device map — hand-edited, never validated
data_root/{sample}/index.csv     one row per run — the searchable database
data_root/{sample}/notes.md      running lab-book prose — hand-edited
data_root/{sample}/raw/          append-only: never edit, never rename a file in here
data_root/{sample}/proc/         plots, fits, extracted parameters
```

`ensure_sample(data_root, sample, create=True)` idempotently stubs all of
this (never overwrites an existing `sample.yaml`/`notes.md`). Every
measurement script computes `_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"`
(three `.parent`s — the script lives at `bridge/{dc,mfli}/foo.py`, and
`data/` is a sibling of `bridge/`, not inside it).

### The `_test` pseudo-sample

`instruments.data_naming.TEST_SAMPLE = "_test"` is a built-in sample for
zero-setup smoke testing. It is auto-created (`ensure_test_sample`) every
time a sample picker renders, is pinned first in the picker list, and goes
through **exactly** the same `allocate_run`/`write_record`/
`finalize_index_row` machinery as any real sample — nothing special-cases
it beyond being excluded from `list_samples()`'s normal listing.

## Filename convention

```
{sample}_{run}_{device}_{type}_{T}_{key_axis}_{timestamp}.csv
```

- **`run`** — zero-padded to 4 digits (`0001`, `0047`), monotonic per
  sample, allocated under a file lock together with its `index.csv` row
  (see `allocate_run()` below). A crashed run permanently consumes its
  number — numbers are never reused, never recomputed by scanning `raw/`.
- **`device`** — free text, e.g. `HB3`.
- **`type`** — a fixed abbreviation from the locked set below. Never
  invent a new one without adding it to this table.
- **`T`** — nominal *setpoint* only (`T010K`), never the measured
  temperature. Omitted entirely if no setpoint was given for the run.
- **`key_axis`** — the one *fixed* secondary parameter that distinguishes
  this run within a family, e.g. a fixed field angle while sweeping
  current. Omitted when not applicable. See "Key-axis tokens" below.
- **`timestamp`** — ISO 8601 basic, `YYYYMMDDThhmmss`, local time (the
  header's own `timestamp:` field carries the full UTC-offset ISO
  string — the filename token is deliberately just the compact form).

Example: `A_0001_HB3_NOISE_T293K_20260811T143022.csv`

### Type codes (locked set)

| Code    | Measurement                                  |
|---------|-----------------------------------------------|
| `IV`    | I–V curve (`dc_iv_curve`)                     |
| `HALL`  | Hall measurement (`dc_hall_measurement`)      |
| `GSWP`  | Gate sweep (`dc_gate_sweep`)                  |
| `BSWP`  | Field/spin-valve sweep (`dc_spin_valve`)      |
| `HARM`  | Dual-harmonic lock-in (`mfli_dual_harmonic`)  |
| `DIFFR` | Differential resistance vs. bias (`mfli_diff_resistance_vs_bias`) |
| `PHCAL` | Phase calibration (`mfli_phase_calibration`)  |
| `NOISE` | Noise spectrum (`mfli_noise_spectrum`)        |

A genuinely new measurement kind gets a new short all-caps code added
here and to nowhere else — `type_code` is just a string parameter to
`allocate_run()`.

### Key-axis tokens

`format_axis_token(kind, value)` is the single source of truth for
spelling; a new script must reuse one of the existing `kind`s or extend
the table in `data_naming.py`, never invent ad hoc formatting:

| kind         | prefix | unit | zero-pad | example                          |
|--------------|--------|------|----------|-----------------------------------|
| `angle_deg`  | `a`    | (none) | 3 digits | `format_axis_token("angle_deg", -90)` → `am090` |
| `field_T`    | `B`    | `T`  | none     | `format_axis_token("field_T", 2)` → `B2T` |
| `gate_V`     | `Vg`   | `V`  | none     | `format_axis_token("gate_V", -0.5)` → `Vgm0p5V` |
| `current_A`  | `I`    | `A`  | none     | `format_axis_token("current_A", 1e-6)` → `I1e-06A` |

Negative values get an `m` prefix on the magnitude (`m` = minus), applied
uniformly. Non-integer values replace `.` with `p`.

## The raw CSV header

Every file in `raw/` starts with a `# key: value` comment block (one line
per key, in insertion order), **immediately** followed by a two-row column
header — a Long Names row, then a Units row — then the data rows. No blank
line anywhere. Written by `write_record()`, **never hand-typed**:

```
# run: 1
# timestamp: 2026-08-11T14:30:22+02:00
# sample: A
# device: HB3
# type: NOISE
# T_setpoint_K: 293.0
# T_K: 293.14
# cooldown: 3
# status: completed
# comment:
# series: A_HB3_NOISE_20260811T143022
...(measurement-specific keys)...
time_s,x,y,...
s,V,V,...
0.0,1.2e-6,...
```

The two-row column header is what makes the file drag-and-drop friendly in
OriginLab's Text/CSV connector: Origin auto-detects the `#` block as the
file header and the next two rows as its Long Name / Units subheaders —
but only when they sit **directly** above the data with no blank line.
`write_record()` builds the Units row by splitting a known unit suffix off
each column name (`magnet_field_mT` → `magnet_field` + `mT`); the suffix
set is `_COLUMN_UNITS` in `data_naming.py`. A column with no recognised
suffix keeps its full name and an empty unit.

Header lines still start with `#` in column 0, but `pd.read_csv(path,
comment="#")` is **no longer** a correct read — it would treat the Units
row as the first data row. Use **`read_raw(path)`** (below), which skips
the `#` block, consumes both header rows, and re-joins them into the
original `name_unit` column labels.

**Universal header/index columns** (`data_naming.BASE_COLUMNS`), in order:
`run, timestamp, sample, device, type, T_setpoint_K, T_K, cooldown,
status, comment, series`. A caller's additional (measurement-specific)
keys are appended after these, in first-seen order, and the same union
happens automatically in `index.csv`.

- `T_setpoint_K` — the nominal target, or blank if none.
- `T_K` — the actual measured temperature (e.g. averaged from logged
  readings over the run), filled in at finalize time, not necessarily at
  allocation time.
- `status` — starts as `"in_progress"` at allocation, then
  **unconditionally** rewritten to an outcome-derived value
  (`"completed"` / `"aborted"` / `"error"`) the instant the run's loop
  returns/raises/aborts, and **optionally** overwritten again with a
  physical judgement (`"good"` / `"open"` / `"short"` / `"noisy"`) if the
  user answers the post-run prompt. The first write must never be gated
  on the second happening — that prompt can always be skipped.
- `series` — a caller-chosen tag shared across every file belonging to
  one multi-file session (e.g. one gate-sweep session producing a file
  per gate voltage) — how you group related `index.csv` rows back
  together after the fact.

## `index.csv`

One row per run, the same union-of-columns as the header, kept as a small
(hundreds-of-rows) plain CSV — the searchable database for the sample.
Rewritten in full on every write (not append-only; only `raw/` is
append-only). Matched by the `run` column, never by row position.

## The API (`instruments/data_naming.py`)

For a **new script**, the sequence is:

```python
from instruments.data_naming import (
    ensure_sample, allocate_run, make_incremental_writer,
    write_record, read_raw, finalize_index_row, proc_path, preview_raw_filename,
)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MEASUREMENT_TYPE = "XYZ"  # add to the type-code table above

# 1. Make sure the sample folder tree exists (idempotent).
ensure_sample(_DATA_DIR, sample, create=True)

# 2. Allocate ONE run per output file — under a file lock, this both
#    picks the run number and appends a status="in_progress" index.csv
#    row, so a crash never reuses or loses a number.
ctx = allocate_run(
    _DATA_DIR, sample, device, MEASUREMENT_TYPE,
    temperature_setpoint_K=temperature_setpoint_K,   # or None
    key_axis=("angle_deg", 45.0),                      # or None
    series=series_tag,                                 # shared across a multi-file session, or None
)
# ctx: RunContext(sample, device, run_number, run_str, timestamp, raw_path, sample_dir)

# 3. Build a writer closure bound to this ONE raw_path/run.
def header_fields(records: list[dict]) -> dict:
    return {
        "run": ctx.run_number, "timestamp": ctx.timestamp.isoformat(timespec="seconds"),
        "sample": ctx.sample, "device": ctx.device, "type": MEASUREMENT_TYPE,
        "T_setpoint_K": temperature_setpoint_K, "T_K": ..., "cooldown": cooldown,
        "status": "in_progress", "comment": "", "series": series_tag,
        # ...any measurement-specific keys...
    }
write_csv = make_incremental_writer(ctx.raw_path, header_fields)

# 4. Call write_csv(records) after every point (crash-safety: full
#    rewrite of the file every time, never append-then-hope).
write_csv(records)

# 5. On the run finishing/raising/aborting, finalize UNCONDITIONALLY with
#    the outcome-derived status, before any optional user prompt:
finalize_index_row(_DATA_DIR, ctx.sample, ctx.run_number, header_fields(records) | {"status": "completed"})

# 6. If/when the user supplies a physical judgement + comment afterward,
#    call finalize_index_row() again — see "never-truncate guard" below.
```

Key functions:

- **`ensure_sample(data_root, sample, create=False)`** → `Path`. Raises
  `SampleNotFoundError` if `create=False` and the sample doesn't exist yet.
- **`list_samples(data_root)`** → real samples only (excludes `_test`),
  alphabetical.
- **`ensure_test_sample(data_root)`** → `ensure_sample(..., TEST_SAMPLE, create=True)`.
- **`allocate_run(...)`** → `RunContext`. The single side-effecting entry
  point — call it once per output **file**. Multi-file-per-session
  suites (one file per swept value) call it once per loop iteration;
  **never reuse one `RunContext`/writer closure across iterations**, or
  every subsequent file in the series silently inherits the first
  iteration's run number.
- **`write_record(raw_path, records, header_fields)`** — rewrites
  `raw_path` in full: `#` header block, then the Long Names row, then the
  Units row, then the data rows — no blank line anywhere (see "The raw CSV
  header" above for the OriginLab reason). Direct replacement for the old
  `Path(...).parent.mkdir(...); pd.DataFrame(records).to_csv(...)` pair.
- **`read_raw(raw_path)`** → `pd.DataFrame`. The matching reader for
  `write_record()`'s body: skips the `# key: value` block, consumes the
  two-row (Long Name + Units) column header, and returns a frame keyed by
  the original `name_unit` labels (units re-joined onto the name, not kept
  as a column level). Empty / header-only files yield an empty DataFrame.
  This is the standard way any downstream analysis script reads a raw file
  back — **not** `pd.read_csv(path, comment="#")`.
- **`make_incremental_writer(raw_path, header_fields_fn)`** → a
  `Callable[[list[dict]], None]` closure. `header_fields_fn` is
  re-invoked on every call with the records-so-far, so e.g. an averaged
  `T_K` or a changing `status` can be recomputed without rebuilding the
  closure. This is what gets passed as a script's `write_csv=` parameter.
- **`finalize_index_row(data_root, sample, run_number, header_fields)`** —
  rewrites just that run's `index.csv` row (matched by `run` number).
  Call unconditionally the instant a run ends (with an outcome-derived
  status), and again if the user supplies a real status/comment.
- **`preview_raw_filename(...)`** — **pure, no I/O, no run-number
  allocation.** Renders a placeholder filename (run token `"NNNN"`, no
  timestamp) for a live "this is roughly what will be saved" hint in a
  UI summary panel. **Never call `allocate_run()` from a
  keystroke-reactive code path** — always use this instead.
- **`proc_path(data_root, sample, run_label, device, type_code, suffix, *, combined=False)`**
  → `Path` under `proc/` for a plot/derived-data file, e.g. a PNG. Pure;
  caller `mkdir(parents=True, exist_ok=True)`s if needed. `run_label` is
  either one run's `run_str` (`"0013"`) or a caller-built range
  (`"0013-0020"`) for a session-level combined plot.
- **`format_axis_token(kind, value)`** — pure filename-token formatting,
  see the table above.

### The never-truncate guard

In every post-run status/comment handler (after the user answers or
skips the prompt), `write_record()`/the incremental writer must **not**
be called with an empty `records` list if the raw file already has data
in it — that would truncate an already-written file to a header-only
stub. The guard used throughout the repo:

```python
if iter_records or not ctx.raw_path.exists():
    write_record(ctx.raw_path, iter_records, header_fields)
```

### Crash-safety pattern

Every `run_measurement()`-style function writes the full CSV after
**every** point (`write_csv(records)`, or the old default plain-CSV
write when no `write_csv` is supplied — most scripts keep a
`write_csv: Optional[Callable] = None` parameter that falls back to the
pre-migration behavior so the function stays usable standalone/in tests
without the data-convention machinery).

### Partial-result preservation on exception

For scripts that only produce a meaningful record at the *end* of a
computation (e.g. `mfli_noise_spectrum.py`'s Welch-transform spectra —
there's no valid "partial spectrum" to write incrementally), wrap the
acquisition loop and defer the exception instead of losing already-computed
results:

```python
error: Optional[BaseException] = None
try:
    for ...:
        results[key] = compute(...)
except BaseException as exc:
    error = exc
finally:
    <unconditional instrument cleanup>

if results:
    save_results(results, ..., status="completed" if error is None else "error")
if error is not None:
    raise error
```

## Single-file vs. multi-file sessions

- **Single-file suites** (I–V curve, Hall, dual-harmonic, diff-resistance,
  phase calibration, noise spectrum runs one `allocate_run()` per
  condition/channel pair): one `allocate_run()` call, one writer closure,
  used for the whole run.
- **Multi-file suites** (gate sweep, spin valve — one CSV per swept
  value): one shared `series` tag built once before the loop, then a
  **fresh** `allocate_run()` + `make_incremental_writer()` **each loop
  iteration**, bound to a fresh `RunContext`. This is what lets
  `index.csv` group them back together via the shared `series` value
  while still giving each file its own `run` number and its own
  finalize/status-comment cycle.

## Sample-picker UI helpers

Both UI toolkits get the same three affordances, implemented once and
reused by every `*_tui.py` / `web/**/*.py` page:

- **Textual** (`instruments/tui_sample_picker.py`):
  `sample_options(data_root)` for a `Select` widget's choices,
  `NewSampleScreen` (modal, name → `ensure_sample(..., create=True)`),
  `StatusCommentScreen` (modal, status `Select` from `STATUS_OPTIONS =
  ["good", "open", "short", "noisy"]` + comment `Input`, dismisses with
  `(status, comment)` or `None` if skipped).
- **NiceGUI** (`web/sample_picker.py`): `sample_select(data_root_getter,
  default=TEST_SAMPLE)` returns `(select, refresh_options)` for a
  `ui.select`; `new_sample_dialog(data_root)` and
  `status_comment_dialog()` are the modal-dialog equivalents, both
  `async` (`await`ed from a NiceGUI coroutine).

Both share the `NEW_SAMPLE_SENTINEL` pattern: the picker's last option is
`"+ New sample…"`; selecting it pops the new-sample dialog/screen and,
on success, refreshes the options with the new name selected.

## What NOT to do

- Don't parse a folder name to recover temperature/angle/status — those
  only ever come from the header/`index.csv`, never from path structure.
- Don't hand-write a `# key: value` header — always go through
  `write_record()`/`make_incremental_writer()`.
- Don't read a raw file with a bare `pd.read_csv(path, comment="#")` — it
  eats the Units subheader row as data. Use `read_raw()`.
- Don't invent a new key-axis `kind` or type code without adding it to
  the locked tables in `data_naming.py` (and this document).
- Don't call `allocate_run()` on every keystroke of a live-updating
  summary panel — use `preview_raw_filename()`.
- Don't reuse one `RunContext`/writer closure across a multi-file loop.
- Don't skip the outcome-derived `finalize_index_row()` call while
  waiting on an optional user prompt — write it unconditionally first.
