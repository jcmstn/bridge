"""
Shared Textual sample-picker + post-run status/comment widgets
===================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-11

Every dc/*_tui.py and mfli/*_tui.py program needs the same three things
from the new data convention (see instruments/data_naming.py): a sample
picker (with a "+ New sample" flow) populated by scanning the data root,
and a post-run prompt collecting the physical status/comment judgement
that can only be known after a measurement finishes. Implemented once
here rather than duplicated across 7 TUIs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from instruments.data_naming import TEST_SAMPLE, ensure_sample, ensure_test_sample, list_samples

NEW_SAMPLE_SENTINEL = "__new_sample__"
STATUS_OPTIONS = ["good", "open", "short", "noisy"]


def sample_options(data_root: Path) -> list[tuple[str, str]]:
    """
    (label, value) pairs for a Select widget: the built-in quick-test
    pseudo-sample pinned first (auto-created if missing), then real
    samples alphabetically, then a synthetic "+ New sample…" entry.
    """
    ensure_test_sample(data_root)
    options: list[tuple[str, str]] = [("— quick test (_test) —", TEST_SAMPLE)]
    options += [(s, s) for s in list_samples(data_root)]
    options.append(("+ New sample…", NEW_SAMPLE_SENTINEL))
    return options


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


class NewSampleScreen(ModalScreen[Optional[str]]):
    """
    Small modal: type a sample name, stubs its folder tree (sample.yaml,
    notes.md, raw/, proc/, index.csv — see ensure_sample()), and dismisses
    with the new sample name, or None if cancelled.
    """

    CSS = """
    NewSampleScreen { align: center middle; }
    #dialog { width: 60; height: auto; padding: 1 2; border: solid $primary; background: $panel; }
    #error { color: $error; height: auto; margin-top: 1; }
    """

    def __init__(self, data_root: Path) -> None:
        super().__init__()
        self.data_root = data_root

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("New sample name", classes="field-label")
            yield Input(placeholder="e.g. A, ChipL, SampleB2", id="new_sample_name")
            yield Static("", id="error")
            with Horizontal():
                yield Button("Create", id="create", variant="success")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        name = self.query_one("#new_sample_name", Input).value.strip()
        error = validate_sample_name(name, self.data_root)
        if error:
            self.query_one("#error", Static).update(error)
            return
        ensure_sample(self.data_root, name, create=True)
        self.dismiss(name)


class StatusCommentScreen(ModalScreen[Optional[tuple[str, str]]]):
    """
    Post-run prompt: the physical judgement (good/open/short/noisy) and a
    free-text comment that can only be known after the measurement
    finishes. Dismisses with (status, comment), or None if skipped —
    skipping leaves the run's index/header row at whatever outcome-derived
    status ("completed"/"aborted"/"error") was already written
    unconditionally right after the run finished (see each *_tui.py's
    RunScreen._on_finished()).
    """

    CSS = """
    StatusCommentScreen { align: center middle; }
    #dialog { width: 70; height: auto; padding: 1 2; border: solid $primary; background: $panel; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Run finished — record the outcome", classes="field-label")
            yield Select([(s, s) for s in STATUS_OPTIONS], value="good", allow_blank=False,
                         id="status_select")
            yield Label("Comment", classes="field-label")
            yield Input(placeholder="e.g. clear AHE hysteresis, coercive ~80 mT", id="comment_input")
            with Horizontal():
                yield Button("Save", id="save", variant="success")
                yield Button("Skip", id="skip")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skip":
            self.dismiss(None)
            return
        status = self.query_one("#status_select", Select).value
        comment = self.query_one("#comment_input", Input).value.strip()
        self.dismiss((status, comment))
