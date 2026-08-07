"""
Shared output-path helper for bridge measurement programs
=============================================================
Promoted out of bridge/DC/dc_sweep_utils.py (which still re-exports it, so
every existing DC *_tui.py import keeps working unchanged) so both the DC
and MFLI suites — and bridge/web/ — can build a consistent
data_dir/subdir/prefix_timestamp.ext path without duplicating the logic.
Pure pathlib, no DC- or MFLI-specific imports, so it belongs in
bridge/instruments/ alongside the other cross-suite shared helpers.
"""

from __future__ import annotations

from pathlib import Path


def build_output_path(data_dir: Path, subdir: str, prefix: str, timestamp: str,
                       suffix: str = "", ext: str = "csv") -> Path:
    """
    Build `data_dir / subdir / f"{prefix}_{timestamp}{suffix}.{ext}"`.

    `subdir` may be empty, in which case the file is written directly
    into `data_dir` (unchanged from the pre-sub-directory behavior).
    """
    out_dir = data_dir / subdir if subdir else data_dir
    return out_dir / f"{prefix}_{timestamp}{suffix}.{ext}"
