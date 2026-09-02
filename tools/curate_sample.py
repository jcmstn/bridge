#!/usr/bin/env python3
"""
curate_sample.py — interactive TUI for deciding which runs of a sample go
into paper figures.

Reads {data_root}/{sample}/index.csv, groups runs by
(device, type, T_setpoint_K, series), and lets you page through groups and
individual runs with quick-look matplotlib plots. Decisions are written
back as three new index.csv columns, through the same finalize_index_row()
path the acquisition scripts already use — nothing in raw/ is touched, and
the CSV is the only thing that changes, one row at a time:

    paper_include    True / False / unset — your keep/exclude decision
    figure_ref       free text, e.g. "Fig2b"
    curation_note    free text, e.g. "cleaner repeat of #0044"

Usage:
    python curate_sample.py A
    python curate_sample.py A --data-root /path/to/data

Keybindings (also shown in the footer):
    up/down/left/right, enter   navigate / expand-collapse the tree
    k        mark current run  -> paper_include = True
    x        mark current run  -> paper_include = False
    u        clear the decision on the current run
    f        set figure_ref (opens a text prompt)
    n        set curation_note (opens a text prompt)
    c        cycle which numeric column is used as the x-axis
    a        toggle showing error/aborted runs (hidden by default)
    q        quit

Selecting a *run* leaf shows that run's own traces (one subplot per
measured column). Selecting a *group* node shows all its runs overlaid
on one axes, for comparing repeats at a glance.

CAVEAT: matplotlib needs an actual display. If you run this over SSH,
you'll need X11 forwarding (ssh -X) or a VNC session — the Textual side
works fine headless, but the plot window will not.

This script assumes the data_naming.py API described in your data storage
convention doc: ensure_sample, list_samples, finalize_index_row,
TEST_SAMPLE, and (optionally) BASE_COLUMNS. Adjust the import block below
if your actual module lives somewhere else or names things differently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, Static, Tree
from textual.widgets.tree import TreeNode

# --------------------------------------------------------------------------
# Wiring into your repo. This mirrors the "_DATA_DIR = three .parents up"
# convention from the other scripts — adjust if this file doesn't live at
# the same depth (e.g. bridge/tools/curate_sample.py).
# --------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from instruments.data_naming import (
        TEST_SAMPLE,
        ensure_sample,
        finalize_index_row,
        list_samples,
    )
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"Could not import instruments.data_naming ({exc}).\n"
        "Run this from inside the bridge repo, or fix _REPO_ROOT above."
    )

try:
    from instruments.data_naming import BASE_COLUMNS
except ImportError:
    BASE_COLUMNS = [
        "run", "timestamp", "sample", "device", "type",
        "T_setpoint_K", "T_K", "cooldown", "status", "comment", "series",
    ]

_DATA_DIR = _REPO_ROOT / "data"

CURATION_COLUMNS = ["paper_include", "figure_ref", "curation_note"]
BAD_STATUSES = {"error", "aborted"}
GROUP_COLUMNS = [c for c in ("device", "type", "T_setpoint_K", "series") if c in BASE_COLUMNS]


# --------------------------------------------------------------------------
# Data loading / grouping helpers — plain functions, no Textual/matplotlib
# dependency, so they're easy to test or reuse from a notebook.
# --------------------------------------------------------------------------

def normalize_bool(value) -> Optional[bool]:
    """Coerce whatever pandas read for paper_include into True/False/None."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def load_index(sample_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_dir / "index.csv")
    for col in CURATION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["paper_include"] = df["paper_include"].map(normalize_bool)
    for col in ("figure_ref", "curation_note"):
        df[col] = df[col].fillna("")
    df["run"] = df["run"].astype(int)
    return df


def extra_columns(df: pd.DataFrame) -> list[str]:
    """Measurement-specific columns beyond BASE_COLUMNS/CURATION_COLUMNS —
    this is where angle_deg / field_T / gate_V / current_A etc. show up,
    without us having to hardcode their names."""
    known = set(BASE_COLUMNS) | set(CURATION_COLUMNS)
    return [c for c in df.columns if c not in known]


def extra_summary(row: pd.Series, extra_cols: list[str]) -> str:
    bits = []
    for col in extra_cols:
        val = row.get(col)
        if val is None or val == "" or (isinstance(val, float) and pd.isna(val)):
            continue
        bits.append(f"{col}={val}")
    return ", ".join(bits)


class RunGroup:
    def __init__(self, key: tuple, run_numbers: list[int]):
        self.key = key
        self.run_numbers = run_numbers


def build_groups(df: pd.DataFrame) -> list[RunGroup]:
    work = df.copy()
    for col in GROUP_COLUMNS:
        work[col] = work[col].fillna("")
    groups = []
    for key, sub in work.groupby(GROUP_COLUMNS, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        sub = sub.sort_values("run")
        groups.append(RunGroup(key=key, run_numbers=sub["run"].tolist()))
    return groups


def format_group_label(group: RunGroup, df: pd.DataFrame) -> str:
    parts = []
    for col, val in zip(GROUP_COLUMNS, group.key):
        if val == "":
            continue
        parts.append(f"T={val}K" if col == "T_setpoint_K" else str(val))
    sub = df[df["run"].isin(group.run_numbers)]
    kept = int((sub["paper_include"] == True).sum())  # noqa: E712
    excluded = int((sub["paper_include"] == False).sum())  # noqa: E712
    return f"{' · '.join(parts)}  ({len(group.run_numbers)} runs, {kept} kept, {excluded} excluded)"


def run_label(row: pd.Series, extra_cols: list[str]) -> str:
    icon = {True: "\u2713", False: "\u2717"}.get(row.get("paper_include"), "\u00b7")
    run = int(row["run"])
    status = str(row.get("status") or "")
    fig_ref = str(row.get("figure_ref") or "")
    bits = [f"{icon} #{run:04d}", status]
    extras = extra_summary(row, extra_cols)
    if extras:
        bits.append(extras)
    if fig_ref:
        bits.append(f"\u2192{fig_ref}")
    return "  ".join(b for b in bits if b)


def find_raw_file(sample_dir: Path, sample: str, run: int) -> Optional[Path]:
    matches = sorted((sample_dir / "raw").glob(f"{sample}_{run:04d}_*.csv"))
    return matches[0] if matches else None


def load_run_dataframe(raw_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(raw_path, comment="#")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------
# matplotlib side: one persistent figure, redrawn in place. plt.pause()
# briefly hands control to the GUI backend so the window actually updates
# — called synchronously from Textual's event handlers, on the same
# (main) thread, which avoids the cross-thread GUI restrictions some
# backends (esp. macOS) impose.
# --------------------------------------------------------------------------

class PlotWindow:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 6))
        plt.show(block=False)
        self.x_col_override: Optional[str] = None

    def plot_run(self, sample_dir: Path, sample: str, row: pd.Series) -> None:
        run = int(row["run"])
        raw_path = find_raw_file(sample_dir, sample, run)
        self.fig.clear()
        if raw_path is None:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"raw file for run #{run:04d} not found", ha="center", va="center")
            self._flush(f"#{run:04d} — missing raw file")
            return

        df = load_run_dataframe(raw_path)
        cols = numeric_columns(df)
        if not cols:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "no numeric columns in this file", ha="center", va="center")
            self._flush(f"#{run:04d}")
            return

        x_col = self.x_col_override if self.x_col_override in cols else cols[0]
        y_cols = [c for c in cols if c != x_col] or [x_col]
        axes = self.fig.subplots(len(y_cols), 1, sharex=True, squeeze=False)[:, 0]
        for ax, y_col in zip(axes, y_cols):
            ax.plot(df[x_col], df[y_col], marker=".", lw=1, ms=3)
            ax.set_ylabel(y_col)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel(x_col)
        title = f"#{run:04d}  {row.get('device','')}  {row.get('type','')}  status={row.get('status','')}"
        self._flush(title)

    def plot_group(self, sample_dir: Path, sample: str, group: RunGroup, df: pd.DataFrame) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        x_label, y_label = None, None
        for run in group.run_numbers:
            raw_path = find_raw_file(sample_dir, sample, run)
            if raw_path is None:
                continue
            run_df = load_run_dataframe(raw_path)
            cols = numeric_columns(run_df)
            if len(cols) < 2:
                continue
            x_col = self.x_col_override if self.x_col_override in cols else cols[0]
            y_col = next((c for c in cols if c != x_col), cols[0])
            ax.plot(run_df[x_col], run_df[y_col], marker=".", lw=1, ms=3, label=f"#{run:04d}")
            x_label, y_label = x_col, y_col
        ax.set_xlabel(x_label or "")
        ax.set_ylabel(y_label or "")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        label_bits = [str(v) for v in group.key if v != ""]
        self._flush(f"group: {' · '.join(label_bits)}  ({len(group.run_numbers)} runs)")

    def available_columns(self, sample_dir: Path, sample: str, run: int) -> list[str]:
        raw_path = find_raw_file(sample_dir, sample, run)
        if raw_path is None:
            return []
        return numeric_columns(load_run_dataframe(raw_path))

    def _flush(self, title: str) -> None:
        self.fig.suptitle(title)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


# --------------------------------------------------------------------------
# Small modal for text entry (figure_ref / curation_note).
# --------------------------------------------------------------------------

class TextInputScreen(ModalScreen[Optional[str]]):
    CSS = """
    TextInputScreen {
        align: center middle;
    }
    #dialog {
        width: 60;
        height: auto;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self, prompt: str, initial: str = ""):
        super().__init__()
        self.prompt = prompt
        self.initial = initial

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.prompt)
            yield Input(value=self.initial, id="value")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------

class CurationApp(App):
    CSS = """
    Tree {
        width: 50%;
        border-right: solid $accent;
    }
    #details {
        width: 50%;
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("k", "mark_keep", "Keep"),
        Binding("x", "mark_exclude", "Exclude"),
        Binding("u", "mark_unset", "Unset"),
        Binding("f", "set_figure_ref", "Fig ref"),
        Binding("n", "set_note", "Note"),
        Binding("c", "cycle_xaxis", "Cycle x-axis"),
        Binding("a", "toggle_show_bad", "Show/hide bad"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, data_root: Path, sample: str):
        super().__init__()
        self.data_root = data_root
        self.sample = sample
        self.sample_dir = data_root / sample
        self.df = load_index(self.sample_dir)
        self.extra_cols = extra_columns(self.df)
        self.show_bad = False
        self.plot_window = PlotWindow()
        self.current_run: Optional[int] = None
        self.current_group: Optional[RunGroup] = None
        self.run_nodes: dict[int, TreeNode] = {}
        self.group_nodes: dict[tuple, TreeNode] = {}

    # -- data helpers --------------------------------------------------

    def visible_df(self) -> pd.DataFrame:
        if self.show_bad:
            return self.df
        return self.df[~self.df["status"].isin(BAD_STATUSES)]

    def row_for_run(self, run: int) -> Optional[pd.Series]:
        matches = self.df[self.df["run"] == run]
        return matches.iloc[0] if len(matches) else None

    # -- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield Tree(f"Sample: {self.sample}", id="tree")
            with Vertical(id="details"):
                yield Static(id="meta")
        yield Footer()

    def on_mount(self) -> None:
        self.rebuild_tree()
        self.refresh_header_counts()
        self.query_one(Tree).focus()

    def rebuild_tree(self) -> None:
        tree = self.query_one(Tree)
        tree.clear()
        tree.root.expand()
        self.run_nodes.clear()
        self.group_nodes.clear()
        vis = self.visible_df()
        for group in build_groups(vis):
            label = format_group_label(group, self.df)
            gnode = tree.root.add(label, data={"kind": "group", "group": group})
            self.group_nodes[group.key] = gnode
            for run in group.run_numbers:
                row = self.row_for_run(run)
                leaf = gnode.add_leaf(run_label(row, self.extra_cols), data={"kind": "run", "run": run})
                self.run_nodes[run] = leaf

    def refresh_header_counts(self) -> None:
        n_runs = len(self.df)
        n_kept = int((self.df["paper_include"] == True).sum())  # noqa: E712
        n_excl = int((self.df["paper_include"] == False).sum())  # noqa: E712
        self.sub_title = f"{n_runs} runs · {n_kept} kept · {n_excl} excluded"

    # -- navigation -> plotting ------------------------------------------

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        data = event.node.data
        if not data:
            return
        if data["kind"] == "run":
            self.current_run = data["run"]
            self.current_group = None
            row = self.row_for_run(self.current_run)
            self.plot_window.plot_run(self.sample_dir, self.sample, row)
            self.update_meta_panel(row)
        elif data["kind"] == "group":
            self.current_group = data["group"]
            self.current_run = None
            self.plot_window.plot_group(self.sample_dir, self.sample, self.current_group, self.df)
            self.update_meta_panel(None, group=self.current_group)

    def update_meta_panel(self, row: Optional[pd.Series], group: Optional[RunGroup] = None) -> None:
        meta = self.query_one("#meta", Static)
        if row is not None:
            lines = [f"[b]Run #{int(row['run']):04d}[/b]", ""]
            for col in BASE_COLUMNS + self.extra_cols:
                if col in row.index:
                    lines.append(f"{col}: {row[col]}")
            lines.append("")
            for col in CURATION_COLUMNS:
                lines.append(f"{col}: {row.get(col)}")
            meta.update("\n".join(lines))
        elif group is not None:
            lines = [f"[b]Group[/b]  {' · '.join(str(v) for v in group.key if v != '')}", ""]
            lines.append(f"{len(group.run_numbers)} runs: " + ", ".join(f"#{r:04d}" for r in group.run_numbers))
            meta.update("\n".join(lines))

    # -- decisions ---------------------------------------------------------

    def action_mark_keep(self) -> None:
        self.set_curation(paper_include=True)

    def action_mark_exclude(self) -> None:
        self.set_curation(paper_include=False)

    def action_mark_unset(self) -> None:
        self.set_curation(paper_include=None)

    def action_set_figure_ref(self) -> None:
        if self.current_run is None:
            return
        row = self.row_for_run(self.current_run)
        current = str(row.get("figure_ref") or "") if row is not None else ""
        self.push_screen(TextInputScreen("Figure reference (e.g. Fig2b):", current), self._apply_figure_ref)

    def _apply_figure_ref(self, value: Optional[str]) -> None:
        if value is not None:
            self.set_curation(figure_ref=value.strip())

    def action_set_note(self) -> None:
        if self.current_run is None:
            return
        row = self.row_for_run(self.current_run)
        current = str(row.get("curation_note") or "") if row is not None else ""
        self.push_screen(TextInputScreen("Curation note:", current), self._apply_note)

    def _apply_note(self, value: Optional[str]) -> None:
        if value is not None:
            self.set_curation(curation_note=value.strip())

    def action_cycle_xaxis(self) -> None:
        run = self.current_run
        cols: list[str] = []
        if run is not None:
            cols = self.plot_window.available_columns(self.sample_dir, self.sample, run)
        elif self.current_group is not None and self.current_group.run_numbers:
            cols = self.plot_window.available_columns(
                self.sample_dir, self.sample, self.current_group.run_numbers[0]
            )
        if not cols:
            return
        current = self.plot_window.x_col_override
        idx = cols.index(current) + 1 if current in cols else 0
        self.plot_window.x_col_override = cols[idx % len(cols)]
        # redraw with the new axis
        if run is not None:
            self.plot_window.plot_run(self.sample_dir, self.sample, self.row_for_run(run))
        elif self.current_group is not None:
            self.plot_window.plot_group(self.sample_dir, self.sample, self.current_group, self.df)

    def action_toggle_show_bad(self) -> None:
        self.show_bad = not self.show_bad
        self.rebuild_tree()

    def set_curation(self, **updates) -> None:
        run = self.current_run
        if run is None:
            self.notify("Select a run (not a group) to tag it.", severity="warning", timeout=2)
            return
        idx = self.df.index[self.df["run"] == run]
        if len(idx) == 0:
            return
        i = idx[0]
        for key, val in updates.items():
            self.df.at[i, key] = val
        row = self.df.loc[i]
        header_fields = {
            k: ("" if isinstance(v, float) and pd.isna(v) else v) for k, v in row.to_dict().items()
        }
        finalize_index_row(self.data_root, self.sample, int(run), header_fields)

        leaf = self.run_nodes.get(run)
        if leaf is not None:
            leaf.set_label(run_label(row, self.extra_cols))
        group = next((g for g in build_groups(self.visible_df()) if run in g.run_numbers), None)
        if group is not None:
            gnode = self.group_nodes.get(group.key)
            if gnode is not None:
                gnode.set_label(format_group_label(group, self.df))
        self.update_meta_panel(row)
        self.refresh_header_counts()
        self.notify(f"Saved run #{run:04d}", timeout=1)


# --------------------------------------------------------------------------

def resolve_samples(selection: str, samples: list[str]) -> list[str]:
    """Turn user input (name, number, or comma-separated numbers/names) into sample names."""
    chosen = []
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            i = int(part)
            if not (1 <= i <= len(samples)):
                sys.exit(f"No sample numbered {i} (have 1-{len(samples)})")
            chosen.append(samples[i - 1])
        elif part in samples:
            chosen.append(part)
        else:
            sys.exit(f"Unknown sample {part!r}")
    if not chosen:
        sys.exit("No sample selected")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample", nargs="?", default=None, help="Sample name, or number(s) from the list, e.g. A or 1,3")
    parser.add_argument("--data-root", type=Path, default=None, help="Override the data root directory")
    args = parser.parse_args()

    data_root = args.data_root or _DATA_DIR
    selection = args.sample

    samples = [s for s in list_samples(data_root) if s != TEST_SAMPLE]
    if not samples:
        sys.exit(f"No samples found under {data_root}")

    if selection is None:
        for i, s in enumerate(samples, 1):
            print(f"{i}. {s}")
        selection = input("Sample(s) to curate (number, comma-separated numbers, or name): ").strip()

    chosen = resolve_samples(selection, samples)

    for sample in chosen:
        ensure_sample(data_root, sample, create=False)  # raises SampleNotFoundError if missing
        app = CurationApp(data_root, sample)
        app.run()
        plt.close("all")


if __name__ == "__main__":
    main()
