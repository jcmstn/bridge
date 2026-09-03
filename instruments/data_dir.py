"""
Shared data-root directory chooser for the Textual TUIs
=======================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-03

The web app lets a run's data root be chosen anywhere on disk
(web/directory_picker.py); this is the Textual equivalent every
dc/*_tui.py and mfli/*_tui.py uses. `validate_directory()` is the shared
pure check — web/directory_picker.py re-exports it from here so the rule
lives in one place and isn't tied to NiceGUI being installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Label, Static


def validate_directory(raw: str) -> tuple[Optional[str], Optional[str]]:
    """
    Validate a save-directory string typed/pasted/picked into a data-root
    field. Returns (warning, error) — at most one is not None.

    Never errors on "doesn't exist yet": every run mkdir(parents=True,
    exist_ok=True)s its output path, so a not-yet-existing directory is
    fine, just worth a heads-up. Only a missing/relative/non-directory
    path is a genuine blocking error.
    """
    text = raw.strip()
    if not text:
        return None, "Data directory is required."
    path = Path(text).expanduser()
    if not path.is_absolute():
        return None, f"'{text}' is not an absolute path."
    if path.exists() and not path.is_dir():
        return None, f"'{path}' exists but is not a directory."
    if not path.exists():
        return f"'{path}' does not exist yet — it will be created when the run starts.", None
    return None, None


class _DirOnlyTree(DirectoryTree):
    """DirectoryTree that hides files and dot-directories — only real
    sub-folders are navigable destinations here."""

    def filter_paths(self, paths):
        return [p for p in paths if p.is_dir() and not p.name.startswith(".")]


class DataDirPickerScreen(ModalScreen[Optional[str]]):
    """
    Browse to a directory and dismiss with its absolute path (or None if
    cancelled). The DirectoryTree can't navigate above its root, so the
    root itself is a typeable Input at the top — set it to jump anywhere;
    the tree drills down from there. "Use this folder" submits whichever
    directory row is highlighted (the root, if none was clicked).
    """

    BINDINGS = [("escape", "dismiss(None)", "Cancel")]

    CSS = """
    DataDirPickerScreen { align: center middle; }
    #dialog { width: 92; height: 40; padding: 1 2; border: solid $primary; background: $panel; }
    #tree { height: 1fr; border: solid $primary; margin: 1 0; }
    #current { color: $text-muted; }
    """

    def __init__(self, start_dir: str) -> None:
        super().__init__()
        start = Path(start_dir).expanduser()
        while not start.is_dir() and start != start.parent:
            start = start.parent
        if not start.is_dir():
            start = Path.home()
        self._start = start.resolve()
        self._selected = self._start

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Data root directory", classes="field-label")
            yield Input(value=str(self._start), id="root_input")
            yield _DirOnlyTree(str(self._start), id="tree")
            yield Static(str(self._selected), id="current")
            with Horizontal():
                yield Button("Use this folder", id="use", variant="success")
                yield Button("Cancel", id="cancel")

    def _set_selected(self, path: Path) -> None:
        self._selected = path
        self.query_one("#current", Static).update(str(path))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        path = Path(event.value).expanduser()
        if path.is_dir():
            tree = self.query_one("#tree", _DirOnlyTree)
            tree.path = path.resolve()
            self._set_selected(path.resolve())

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        self._set_selected(Path(event.path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(str(self._selected) if event.button.id == "use" else None)


def demo() -> None:
    assert validate_directory("")[1]
    assert validate_directory("relative/path")[1]
    assert validate_directory(str(Path(__file__)))[1]          # a file, not a dir
    assert validate_directory("/no/such/dir/here")[0]          # warning, not error
    assert validate_directory("/no/such/dir/here")[1] is None
    w, e = validate_directory(str(Path(__file__).parent))
    assert w is None and e is None
    print("ok")


if __name__ == "__main__":
    demo()
