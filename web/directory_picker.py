"""
Local directory picker for bridge/web
=========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Lets a run's save directory be chosen freely anywhere on disk, rather than
only a sub-folder name under a hardcoded data/ root. bridge/web runs
locally on the same machine as the lab instruments (localhost-only, no
auth — see app.py), so a server-side local-filesystem browser is safe and
appropriate here.

Adapted from NiceGUI's own upstream `local_file_picker` example
(zauberzeug/nicegui, examples/local_file_picker/local_file_picker.py),
narrowed to directories only:
  - the grid lists sub-directories, not files (nothing to "open" here)
  - double-click descends into a directory rather than submitting it
  - a breadcrumb (not just a synthetic ".." row) lets you jump back up
    multiple levels in one click
  - the footer button submits the CURRENT directory itself, not a selected
    grid row — the key behavioral divergence from the file-picker original,
    since a directory is a destination here, not a leaf to select
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

from nicegui import events, ui


class LocalDirectoryPicker(ui.dialog):

    def __init__(self, start_dir: str, *, upper_limit: Optional[str] = None,
                 show_hidden: bool = False) -> None:
        """
        :param start_dir: directory to open in (falls back to the user's
            home directory if it doesn't exist/isn't a directory).
        :param upper_limit: directory to stop breadcrumb/".." navigation at
            (None: no limit — can browse all the way to the filesystem root).
        :param show_hidden: whether to show dot-directories.
        """
        super().__init__()

        initial = Path(start_dir).expanduser()
        if not initial.is_dir():
            initial = Path.home()
        self.path = initial.resolve()
        self.upper_limit = Path(upper_limit).expanduser().resolve() if upper_limit else None
        self.show_hidden = show_hidden

        with self, ui.card().classes("w-[36rem]"):
            self.breadcrumb_row = ui.row().classes("w-full items-center gap-1 flex-wrap")
            self._add_drives_toggle()
            self.grid = ui.aggrid({
                "columnDefs": [{"field": "name", "headerName": "Folder"}],
                "rowSelection": {"mode": "singleRow"},
                "headerHeight": 0,
            }, html_columns=[0]).classes("w-full h-80").on("cellDoubleClicked", self._handle_double_click)
            with ui.row().classes("w-full justify-between items-center"):
                ui.label().bind_text_from(self, "path", lambda p: str(p)).classes("text-xs text-grey")
                with ui.row().classes("gap-2"):
                    ui.button("Cancel", on_click=lambda: self.submit(None)).props("outline")
                    ui.button("Select this folder", on_click=lambda: self.submit(str(self.path))) \
                        .props("color=primary")
        self._refresh()

    def _add_drives_toggle(self) -> None:
        # Lab-instrument control PCs are frequently Windows even though this
        # codebase is developed on macOS — cheap to support, no cost if unused.
        if platform.system() != "Windows":
            return
        import win32api  # type: ignore[import-not-found]
        drives = win32api.GetLogicalDriveStrings().split("\000")[:-1]
        ui.toggle(drives, value=drives[0], on_change=lambda e: self._jump(Path(e.value))) \
            .classes("mb-1")

    def _jump(self, path: Path) -> None:
        self.path = path
        self._refresh()

    def _handle_double_click(self, e: events.GenericEventArguments) -> None:
        candidate = Path(e.args["data"]["path"])
        if candidate.is_dir():
            self._jump(candidate)

    def _refresh(self) -> None:
        self._render_breadcrumb()
        subdirs = [p for p in self.path.glob("*") if p.is_dir()]
        if not self.show_hidden:
            subdirs = [p for p in subdirs if not p.name.startswith(".")]
        subdirs.sort(key=lambda p: p.name.lower())

        row_data = [{"name": f"📁 {p.name}", "path": str(p)} for p in subdirs]
        can_go_up = self.path != self.path.parent and \
            (self.upper_limit is None or self.path != self.upper_limit)
        if can_go_up:
            row_data.insert(0, {"name": "📁 ..", "path": str(self.path.parent)})
        self.grid.options["rowData"] = row_data
        self.grid.update()

    def _render_breadcrumb(self) -> None:
        self.breadcrumb_row.clear()
        with self.breadcrumb_row:
            parts = self.path.parts
            for i, _ in enumerate(parts):
                segment_path = Path(*parts[: i + 1])
                is_last = i == len(parts) - 1
                ui.button(parts[i], on_click=lambda p=segment_path: self._jump(p)) \
                    .props("flat dense" + (" color=primary" if is_last else "")) \
                    .classes("text-xs px-1 min-w-0")
                if not is_last:
                    ui.label("/").classes("text-xs text-grey")


def validate_directory(raw: str) -> tuple[Optional[str], Optional[str]]:
    """
    Validate a save-directory string typed/pasted/picked into a
    directory_field(). Returns (warning, error) — at most one is not None.

    Never errors on "doesn't exist yet": every run_measurement() already
    does Path(output_file).parent.mkdir(parents=True, exist_ok=True), so a
    not-yet-existing directory is fine, just worth a heads-up. Only a
    missing/relative/non-directory path is a genuine blocking error.
    """
    text = raw.strip()
    if not text:
        return None, "Save directory is required."
    path = Path(text).expanduser()
    if not path.is_absolute():
        return None, f"'{text}' is not an absolute path."
    if path.exists() and not path.is_dir():
        return None, f"'{path}' exists but is not a directory."
    if not path.exists():
        return f"'{path}' does not exist yet — it will be created when the run starts.", None
    return None, None


def directory_field(label: str, default: str) -> ui.input:
    """
    The composite "choose save directory" widget every page uses: a
    typeable/pasteable absolute-path ui.input (no dialog round-trip needed
    for the common "same folder as last time" case) plus a Browse… button
    opening a LocalDirectoryPicker. Returns the ui.input element — pages
    read `.value` in parse_state() the same way as every other field.
    """
    with ui.row().classes("w-full items-end gap-2"):
        inp = ui.input(label, value=default).props("outlined dense").classes("flex-grow")

        async def browse() -> None:
            start = inp.value.strip() or default
            result = await LocalDirectoryPicker(start)
            if result:
                inp.value = result

        ui.button("Browse…", on_click=browse).props("outline")
    return inp
