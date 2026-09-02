"""
Per-sample data-convention naming, header, and index writer for bridge
=========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-11

Implements the data convention: folders encode only what is physically
permanent (the sample); everything else — temperature, field angle,
cooldown, run status — lives in filenames, CSV headers, and a per-sample
index.csv. Supersedes instruments/output_paths.py.

Folder tree (exactly three levels, per sample):
    data_root/{sample}/sample.yaml   stack, geometry, device map (hand-edited)
    data_root/{sample}/index.csv      one row per run — the searchable database
    data_root/{sample}/notes.md        running lab-book prose (hand-edited)
    data_root/{sample}/raw/             append-only: never edit, never rename
    data_root/{sample}/proc/            plots, fits, extracted parameters

Filename: {sample}_{run}_{device}_{type}_{T}_{key_axis}_{timestamp}.csv
  - run: zero-padded (>=4 digits), monotonic, per-sample, allocated under a
    file lock together with its index.csv row so a crashed run permanently
    consumes its number (never reused, never recomputed from raw/ scanning).
  - T: nominal setpoint only (e.g. T010K) — NOT the measured temperature.
    Omitted if no setpoint was given.
  - key_axis: the one FIXED secondary parameter distinguishing this run
    within a family (e.g. angle, while sweeping field) — omitted when not
    applicable, see format_axis_token().
  - timestamp: ISO 8601 basic, YYYYMMDDThhmmss, local time (the header's
    `timestamp:` field carries the full UTC-offset ISO string).

Every raw file gets a `# key: value` comment header, written by
write_record() on every incremental point-write (status="in_progress")
and once more at end-of-run with the real outcome/status/comment.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from filelock import FileLock

TEST_SAMPLE = "_test"

# Universal index.csv / header columns, in the order they should appear.
# Any additional (measurement-specific) keys a caller passes are appended
# after these, in first-seen order.
BASE_COLUMNS = [
    "run", "timestamp", "sample", "device", "type",
    "T_setpoint_K", "T_K", "cooldown", "status", "comment", "series",
]

_LOCK_TIMEOUT_S = 5.0

_SAMPLE_YAML_STUB = """\
# {sample} -- sample metadata
# Fill in by hand. This file is documentation only; nothing in bridge
# validates or enforces its contents.
name: {sample}
created_at: {created_at}
stack: null
geometry: null
device_map: null
contacts: null
"""

_AXIS_PREFIXES = {"angle_deg": "a", "field_T": "B", "gate_V": "Vg", "current_A": "I"}
_AXIS_UNITS = {"angle_deg": "", "field_T": "T", "gate_V": "V", "current_A": "A"}
_AXIS_ZERO_PAD = {"angle_deg": 3}  # digits; other kinds are unpadded


class SampleNotFoundError(Exception):
    """Raised by ensure_sample(create=False)/allocate_run() when a sample
    has no data_root/{sample}/sample.yaml yet."""


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunContext:
    """Returned by allocate_run() — everything a caller needs to proceed
    with a measurement and finalize it later."""
    sample: str
    device: str
    run_number: int
    run_str: str          # zero-padded, e.g. "0047"
    timestamp: datetime    # tz-aware (local), carries the UTC offset
    raw_path: Path
    sample_dir: Path


# ─────────────────────────────────────────────────────────────────────────────
# Axis-token / filename formatting  ── pure, no I/O
# ─────────────────────────────────────────────────────────────────────────────

def format_axis_token(kind: str, value: float) -> str:
    """
    Format a fixed key-axis value into its filename token, e.g.
    format_axis_token("angle_deg", -90) -> "am090"
    format_axis_token("field_T", 2)     -> "B2T"

    `kind` must be one of the locked prefixes below — this table is the
    single source of truth for axis-token spelling; no measurement suite
    should invent its own. Negative values get an "m" prefix on the
    magnitude (m = minus), per the angle-token convention, applied
    uniformly to every kind.
    """
    if kind not in _AXIS_PREFIXES:
        raise ValueError(f"Unknown key-axis kind {kind!r}; known kinds: {sorted(_AXIS_PREFIXES)}")
    prefix = _AXIS_PREFIXES[kind]
    unit = _AXIS_UNITS[kind]
    sign = "m" if value < 0 else ""
    av = abs(value)
    if float(av).is_integer():
        digits = str(int(av))
        pad = _AXIS_ZERO_PAD.get(kind)
        if pad:
            digits = digits.zfill(pad)
        num = digits
    else:
        num = f"{av:g}".replace(".", "p")
    return f"{prefix}{sign}{num}{unit}"


def _temperature_token(temperature_setpoint_K: Optional[float]) -> Optional[str]:
    if temperature_setpoint_K is None:
        return None
    return f"T{int(round(temperature_setpoint_K)):03d}K"


def _filename_stem_parts(
    sample: str, run_token: str, device: str, type_code: str, *,
    temperature_setpoint_K: Optional[float],
    key_axis: Optional[tuple[str, Union[str, float]]],
) -> list[str]:
    parts = [sample, run_token, device, type_code]
    t_token = _temperature_token(temperature_setpoint_K)
    if t_token is not None:
        parts.append(t_token)
    if key_axis is not None:
        kind, value = key_axis
        parts.append(format_axis_token(kind, float(value)))
    return parts


def preview_raw_filename(
    sample: str, device: str, type_code: str, *,
    temperature_setpoint_K: Optional[float] = None,
    key_axis: Optional[tuple[str, Union[str, float]]] = None,
) -> str:
    """
    PURE — no I/O, no lock, no run-number allocation. Renders a
    placeholder filename (run token "NNNN", no timestamp — neither is
    known until write time) for a "this is roughly what will be saved"
    hint in a live-updating summary/validation panel.

    NEVER call allocate_run() from a keystroke-reactive code path — use
    this instead.
    """
    parts = _filename_stem_parts(
        sample, "NNNN", device, type_code,
        temperature_setpoint_K=temperature_setpoint_K, key_axis=key_axis,
    )
    return "_".join(parts)


def _build_raw_filename(
    sample: str, run_str: str, device: str, type_code: str, timestamp: datetime, *,
    temperature_setpoint_K: Optional[float],
    key_axis: Optional[tuple[str, Union[str, float]]],
) -> str:
    parts = _filename_stem_parts(
        sample, run_str, device, type_code,
        temperature_setpoint_K=temperature_setpoint_K, key_axis=key_axis,
    )
    parts.append(timestamp.strftime("%Y%m%dT%H%M%S"))
    return "_".join(parts) + ".csv"


# ─────────────────────────────────────────────────────────────────────────────
# Sample folder-tree bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def list_samples(data_root: Path) -> list[str]:
    """
    Real samples only (excludes the built-in test pseudo-sample — pin
    TEST_SAMPLE first yourself in a picker), sorted alphabetically.
    Missing data_root simply yields [].
    """
    if not data_root.is_dir():
        return []
    names = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name != TEST_SAMPLE and (p / "sample.yaml").is_file():
            names.append(p.name)
    return names


def ensure_sample(data_root: Path, sample: str, *, create: bool = False) -> Path:
    """
    Resolve a sample's folder tree.

    create=False: raises SampleNotFoundError if sample.yaml is missing.
    create=True: idempotently stubs sample.yaml (starter YAML, NEVER
    overwritten if already present), notes.md (single header line, same
    non-overwrite rule), raw/, proc/, and an EMPTY index.csv (header row
    only) — created now, not lazily on first run, so allocate_run() and
    finalize_index_row() never need an exists-or-create branch inside
    their locked critical section.
    """
    sample_dir = data_root / sample
    yaml_path = sample_dir / "sample.yaml"

    if not create:
        if not yaml_path.is_file():
            raise SampleNotFoundError(sample)
        return sample_dir

    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "raw").mkdir(exist_ok=True)
    (sample_dir / "proc").mkdir(exist_ok=True)

    if not yaml_path.exists():
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        yaml_path.write_text(_SAMPLE_YAML_STUB.format(sample=sample, created_at=created_at))

    notes_path = sample_dir / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(f"# {sample}\n")

    index_path = sample_dir / "index.csv"
    if not index_path.exists():
        _write_index(index_path, [], list(BASE_COLUMNS))

    return sample_dir


def ensure_test_sample(data_root: Path) -> Path:
    """
    ensure_sample(data_root, TEST_SAMPLE, create=True) — idempotent, meant
    to be called on every sample-picker render/page load so the built-in
    quick-test pseudo-sample is always present with zero user action.
    Nothing else in this module special-cases TEST_SAMPLE; a run against
    it goes through the exact same allocate_run/write_record/
    finalize_index_row machinery as any real sample.
    """
    return ensure_sample(data_root, TEST_SAMPLE, create=True)


# ─────────────────────────────────────────────────────────────────────────────
# index.csv read/write helpers  ── private, small files (hundreds of rows)
# ─────────────────────────────────────────────────────────────────────────────

def _read_index(index_path: Path) -> tuple[list[dict], list[str]]:
    if not index_path.exists():
        return [], list(BASE_COLUMNS)
    with open(index_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or BASE_COLUMNS)
        rows = list(reader)
    return rows, fieldnames


def _write_index(index_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({fn: row.get(fn, "") for fn in fieldnames})


def _next_run_number(rows: list[dict]) -> int:
    mx = 0
    for row in rows:
        try:
            mx = max(mx, int(row.get("run", "")))
        except (TypeError, ValueError):
            continue
    return mx + 1


def _union_columns(fieldnames: list[str], new_keys) -> list[str]:
    result = list(fieldnames)
    for k in new_keys:
        if k not in result:
            result.append(k)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Run allocation  ── the single side-effecting entry point
# ─────────────────────────────────────────────────────────────────────────────

def allocate_run(
    data_root: Path, sample: str, device: str, type_code: str, *,
    temperature_setpoint_K: Optional[float] = None,
    key_axis: Optional[tuple[str, Union[str, float]]] = None,
    series: Optional[str] = None,
) -> RunContext:
    """
    Under a per-sample file lock: read index.csv, take max(run) + 1 (1 if
    no data rows yet), build the raw filename, APPEND a status="in_progress"
    index.csv row for this run number — all before releasing the lock.
    This is what makes a crash-consumed run number permanently unreusable:
    the next allocate_run() call for this sample sees the row and moves
    past it, with no filename-scanning fallback needed anywhere.

    Call once per output file. Multi-file-per-session suites (a swept
    series value producing one file each) call this once per loop
    iteration — never reuse one RunContext/writer across iterations.
    """
    sample_dir = data_root / sample
    if not (sample_dir / "sample.yaml").is_file():
        raise SampleNotFoundError(sample)

    index_path = sample_dir / "index.csv"
    lock_path = sample_dir / ".index.lock"

    with FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_S):
        rows, fieldnames = _read_index(index_path)
        run_number = _next_run_number(rows)
        run_str = f"{run_number:04d}"
        timestamp = datetime.now().astimezone()

        filename = _build_raw_filename(
            sample, run_str, device, type_code, timestamp,
            temperature_setpoint_K=temperature_setpoint_K, key_axis=key_axis,
        )
        raw_path = sample_dir / "raw" / filename

        new_row = {
            "run": run_number,
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "sample": sample,
            "device": device,
            "type": type_code,
            "T_K": "" if temperature_setpoint_K is None else temperature_setpoint_K,
            "cooldown": "",
            "status": "in_progress",
            "comment": "",
            "series": series or "",
        }
        fieldnames = _union_columns(fieldnames, new_row.keys())
        rows.append(new_row)
        _write_index(index_path, rows, fieldnames)

    return RunContext(
        sample=sample, device=device, run_number=run_number, run_str=run_str,
        timestamp=timestamp, raw_path=raw_path, sample_dir=sample_dir,
    )


def finalize_index_row(data_root: Path, sample: str, run_number: int, header_fields: dict) -> None:
    """
    Under the sample's lock, rewrite index.csv's row for run_number
    (matched by the "run" column, not row position), union-ing in any new
    header_fields column names. Full rewrite is fine — index.csv is a
    few-hundred-row file with no append-only requirement (only raw/ is
    append-only).

    Called once, unconditionally, the instant a run's loop
    returns/raises/aborts (outcome-derived status: "completed"/"aborted"/
    "error" — not yet a physical judgment), and again if/when the user
    supplies a real status ("good"/"open"/"short"/"noisy") + comment
    afterward. Never gate the first call on the second happening.
    """
    sample_dir = data_root / sample
    index_path = sample_dir / "index.csv"
    lock_path = sample_dir / ".index.lock"

    with FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_S):
        rows, fieldnames = _read_index(index_path)
        fieldnames = _union_columns(fieldnames, header_fields.keys())

        for row in rows:
            try:
                matches = int(row.get("run", "")) == run_number
            except (TypeError, ValueError):
                matches = False
            if matches:
                row.update(header_fields)
                break
        else:
            new_row = {fn: "" for fn in fieldnames}
            new_row.update(header_fields)
            new_row["run"] = run_number
            rows.append(new_row)

        _write_index(index_path, rows, fieldnames)


# ─────────────────────────────────────────────────────────────────────────────
# Raw-file header + CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def _format_header_value(value) -> str:
    return "" if value is None else str(value)


# Column-name unit suffixes split into Origin's auto-detected Units subheader
# row by write_record(). A column whose last "_"-separated token is in this set
# is written as `<name>` on the Long Name row and `<token>` on the Units row;
# every other column keeps its full name and an empty unit. read_raw() re-joins
# the two rows back into the original `name_unit` labels.
_COLUMN_UNITS = {
    "T", "mT", "uT", "nT",
    "V", "mV", "uV", "nV", "kV",
    "A", "mA", "uA", "nA", "pA",
    "ohm", "kohm", "Mohm",
    "K", "mK", "degC",
    "Hz", "kHz", "MHz",
    "deg", "rad", "s", "ms", "us", "ns",
    "W", "dBm", "dBV", "Oe", "emu",
}


def _split_column_unit(col: str) -> tuple[str, str]:
    """'magnet_field_mT' -> ('magnet_field', 'mT'); 'point_index' -> ('point_index', '')."""
    name, _, suffix = col.rpartition("_")
    return (name, suffix) if name and suffix in _COLUMN_UNITS else (col, "")


def write_record(raw_path: Path, records: list[dict], header_fields: dict) -> None:
    """
    Rewrite raw_path in full: a `# key: value` comment header block
    (header_fields, in insertion order), then a two-row column header —
    Long Names, then Units (split off known unit suffixes, see
    _COLUMN_UNITS) — then the data rows. No blank line anywhere: OriginLab's
    Text/CSV drag-and-drop connector auto-detects the `#` block as file
    header and the next two rows as Long Name / Units subheaders only when
    they sit directly above the data. This is the direct replacement for
    every script's current 'Path(...).parent.mkdir(...);
    pd.DataFrame(records).to_csv(...)' pair.

    Call on EVERY incremental point-write (header_fields["status"] =
    "in_progress") AND once more at end-of-run with the final status/
    comment. Read it back with read_raw() (the `#` lines stay skippable by
    a plain `pd.read_csv(path, comment="#")`, but that would take the Units
    row as data — use read_raw()).
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [f"# {k}: {_format_header_value(v)}" for k, v in header_fields.items()]
    df = pd.DataFrame(records)
    with open(raw_path, "w", newline="") as f:
        f.write("\n".join(header_lines) + "\n")
        if len(df.columns):
            names, units = zip(*(_split_column_unit(c) for c in df.columns))
            f.write(",".join(names) + "\n")
            f.write(",".join(units) + "\n")
        df.to_csv(f, index=False, header=False)


def read_raw(raw_path: Path) -> pd.DataFrame:
    """
    Inverse of write_record()'s body: skip the `# key: value` header, read
    the Long Name + Units two-row column header, and return a DataFrame
    keyed by the original `name_unit` column labels (the Units row is
    re-joined onto the name, not kept as a level). Empty/header-only files
    yield an empty DataFrame.
    """
    try:
        df = pd.read_csv(raw_path, comment="#", header=[0, 1])
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    df.columns = [
        name if not unit or str(unit).startswith("Unnamed") else f"{name}_{unit}"
        for name, unit in df.columns
    ]
    return df


def make_incremental_writer(
    raw_path: Path, header_fields: Callable[[list[dict]], dict],
) -> Callable[[list[dict]], None]:
    """
    Returns a closure bound to ONE raw_path (one run number). `header_fields`
    is re-invoked on every call WITH the records collected so far (so it can
    e.g. average measured temperature readings into T_K), so status/comment
    can change between mid-run ("in_progress") and the final call without
    rebuilding the closure. Pass the result as run_measurement()'s new
    write_csv= parameter.

    For multi-file-per-session suites: build a NEW closure (from a fresh
    allocate_run() RunContext) each loop iteration — never reuse one
    closure across iterations, or every file in the series silently
    inherits the first iteration's run number.
    """
    def _write(records: list[dict]) -> None:
        write_record(raw_path, records, header_fields(records))
    return _write


# ─────────────────────────────────────────────────────────────────────────────
# proc/ path helper
# ─────────────────────────────────────────────────────────────────────────────

def proc_path(
    data_root: Path, sample: str, run_label: str, device: str, type_code: str,
    suffix: str, *, combined: bool = False,
) -> Path:
    """
    sample/proc/ path for a PNG etc. PURE — no I/O, caller mkdir(parents=
    True, exist_ok=True)s if needed (proc/ already exists once
    ensure_sample() has run, but callers shouldn't assume that).

    `run_label`: a single run's zero-padded run_str (e.g. "0013") for a
    normal per-run plot, or a caller-built range string (e.g.
    "0013-0020") for combined=True's session-level combined plot — the
    `combined` flag exists purely for readability at call sites, it makes
    no formatting decision `run_label` doesn't already encode.
    """
    del combined  # documentation-only, see docstring
    name = f"{sample}_{run_label}_{device}_{type_code}_{suffix}.png"
    return data_root / sample / "proc" / name
