"""
Shared NiceGUI sample-picker + post-run status/comment dialogs
====================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-11

NiceGUI analog of instruments/tui_sample_picker.py — every web/dc/*.py and
web/mfli/*.py page needs the same sample picker (with a "+ New sample"
flow) and post-run status/comment prompt from the data convention (see
instruments/data_naming.py). Implemented once here rather than duplicated
across 7 pages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from nicegui import ui

from instruments.data_naming import TEST_SAMPLE, ensure_sample, ensure_test_sample, list_samples

NEW_SAMPLE_SENTINEL = "__new_sample__"
STATUS_OPTIONS = ["good", "open", "short", "noisy"]


def validate_sample_name(name: str, data_root: Path) -> str:
    """Returns an error message, or "" if `name` is available for a new sample."""
    if not name:
        return "Sample name is required."
    if not re.match(r"^[A-Za-z0-9_-]+$", name):
        return "Only letters, digits, '_' and '-' are allowed."
    if name == TEST_SAMPLE:
        return f"'{TEST_SAMPLE}' is reserved for quick tests — choose another name."
    if (data_root / name / "sample.yaml").exists():
        return f"Sample '{name}' already exists — pick it from the list instead."
    return ""


def _sample_option_map(data_root: Path) -> dict:
    ensure_test_sample(data_root)
    options: dict = {TEST_SAMPLE: "— quick test (_test) —"}
    for s in list_samples(data_root):
        options[s] = s
    options[NEW_SAMPLE_SENTINEL] = "+ New sample…"
    return options


async def new_sample_dialog(data_root: Path) -> Optional[str]:
    """Modal: type a sample name, stub its folder tree, return the name
    (or None if cancelled)."""
    with ui.dialog() as dialog, ui.card():
        ui.label("New sample name").classes("font-bold")
        name_input = ui.input(placeholder="e.g. A, ChipL, SampleB2").classes("w-full")
        error_label = ui.label("").classes("text-negative text-sm")

        def create() -> None:
            name = name_input.value.strip()
            error = validate_sample_name(name, data_root)
            if error:
                error_label.set_text(error)
                return
            ensure_sample(data_root, name, create=True)
            dialog.submit(name)

        with ui.row():
            ui.button("Create", on_click=create, color="primary")
            ui.button("Cancel", on_click=lambda: dialog.submit(None))
    return await dialog


async def status_comment_dialog() -> Optional[tuple[str, str]]:
    """
    Post-run prompt: physical judgement (good/open/short/noisy) + a free
    comment, only knowable after the run finishes. Returns (status,
    comment), or None if skipped -- skipping leaves the record at whatever
    outcome-derived status ("completed"/"aborted"/"error") the page already
    wrote unconditionally right when the run finished.
    """
    with ui.dialog() as dialog, ui.card():
        ui.label("Run finished — record the outcome").classes("font-bold")
        status_select = ui.select(STATUS_OPTIONS, value="good", label="Status").classes("w-full")
        comment_input = ui.input(
            label="Comment", placeholder="e.g. clear AHE hysteresis, coercive ~80 mT",
        ).classes("w-full")
        with ui.row():
            ui.button(
                "Save",
                on_click=lambda: dialog.submit((status_select.value, comment_input.value.strip())),
                color="primary",
            )
            ui.button("Skip", on_click=lambda: dialog.submit(None))
    return await dialog


def sample_select(data_root_getter: Callable[[], str], *, default: str = TEST_SAMPLE):
    """
    A ui.select populated from data_root_getter()'s sample folders, with a
    "+ New sample…" entry. `data_root_getter` is a callable (not a plain
    Path) because the data root is itself a live ui.input the user can
    change (see web/directory_picker.py's directory_field()).

    Returns (select, refresh_options) — call refresh_options(value=None)
    whenever the data root changes, to repopulate without altering the
    current selection.
    """
    select = ui.select(_sample_option_map(Path(data_root_getter())), value=default, label="Sample") \
        .classes("w-full").props("outlined dense")

    def refresh_options(select_value: Optional[str] = None) -> None:
        select.options = _sample_option_map(Path(data_root_getter()))
        if select_value is not None:
            select.value = select_value
        select.update()

    async def _on_change() -> None:
        if select.value != NEW_SAMPLE_SENTINEL:
            return
        result = await new_sample_dialog(Path(data_root_getter()))
        refresh_options(result if result else TEST_SAMPLE)

    select.on_value_change(_on_change)
    return select, refresh_options
