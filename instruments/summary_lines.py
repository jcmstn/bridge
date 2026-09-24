"""
Summary-sidebar line format, shared by both front ends
=======================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-24

Every program's ``build_summary()`` returns ``info`` lines written as

    "Key: value — note"      e.g. "Sense current: 10 µA — reversed ±I each rep"

``split_info`` cuts one into (key, value, note) so the TUI (``summary_markup``,
Rich markup) and the web (``web.run_controller.render_summary``) can show the
value bold and the key/note muted. A line without ": " is all note.

Pure (Rich markup strings only, no Textual/NiceGUI).

Usage example:
    from instruments.summary_lines import summary_markup
    self.query_one("#summary", Static).update(summary_markup(info, warnings, errors))
"""
from __future__ import annotations

from rich.markup import escape


def split_info(line: str) -> tuple[str, str, str]:
    """(key, value, note) of one "Key: value — note" line; missing parts are ""."""
    key, sep, rest = line.partition(": ")
    if not sep:
        return "", "", line
    value, _, note = rest.partition(" — ")
    return key, value, note


def summary_markup(info: list[str], warnings: list[str], errors: list[str]) -> str:
    """The TUI sidebar: blocking issues, warnings, then the key/value lines."""
    lines: list[str] = []
    if errors:
        lines.append("[bold red]Blocking issues[/bold red]")
        lines += [f"  [red]✗ {escape(e)}[/red]" for e in errors]
    if warnings:
        lines.append("[bold yellow]Warnings[/bold yellow]")
        lines += [f"  [yellow]⚠ {escape(w)}[/yellow]" for w in warnings]
        lines.append("")
    for line in info:
        if not line:
            continue
        key, value, note = (escape(p) for p in split_info(line))
        if key:
            lines.append(f"[dim]{key}[/dim]  [bold $accent]{value}[/bold $accent]")
        if note:
            lines.append(f"{'  ' if key else ''}[dim]{note}[/dim]")
    return "\n".join(lines)


def demo() -> None:
    assert split_info("Sense current: 10 µA — ±I each rep") == ("Sense current", "10 µA", "±I each rep")
    assert split_info("Sweep: 3 row(s), 41 points") == ("Sweep", "3 row(s), 41 points", "")
    assert split_info("Magnet untouched.") == ("", "", "Magnet untouched.")
    assert "[bold $accent]10 µA" in summary_markup(["Sense current: 10 µA"], [], [])
    assert "\\[b]" in summary_markup(["Pulse: [b] 1 V"], [], [])   # user text can't inject markup


if __name__ == "__main__":
    demo()
    print("ok")
