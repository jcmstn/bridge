"""
web/sample_picker.py — the pure data-root helpers behind the identity bar:
a half-typed Data-root path is never listed (listing creates _test/), and a
page's Start handler gets a `~`-expanded root with the sample bootstrapped.
"""

from __future__ import annotations

from pathlib import Path

from instruments.data_naming import TEST_SAMPLE
from web.sample_picker import (
    NEW_SAMPLE_SENTINEL, _existing_root, _sample_option_map, prepare_data_root,
)


def test_half_typed_root_is_not_listed_and_creates_nothing(tmp_path: Path) -> None:
    half = tmp_path / "da"
    assert _existing_root(str(half)) is None
    assert list(_sample_option_map(None)) == [TEST_SAMPLE, NEW_SAMPLE_SENTINEL]
    assert not half.exists()


def test_existing_root_lists_samples(tmp_path: Path) -> None:
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "sample.yaml").write_text("name: A\n")
    root = _existing_root(f"  {tmp_path}  ")
    assert root == tmp_path.resolve()
    assert list(_sample_option_map(root)) == [TEST_SAMPLE, "A", NEW_SAMPLE_SENTINEL]


def test_prepare_data_root_expands_home_and_bootstraps_sample(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = prepare_data_root(" ~/data ", "B")
    assert root == str(tmp_path / "data")
    assert (tmp_path / "data" / "B" / "sample.yaml").is_file()
    assert (tmp_path / "data" / "B" / "index.csv").is_file()
