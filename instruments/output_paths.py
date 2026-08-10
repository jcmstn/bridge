"""
Shared output-path helper for bridge measurement programs
=============================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Builds a consistent data_dir/subdir/prefix_timestamp.ext path, used by the
DC and MFLI measurement programs and by bridge/web/.
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
