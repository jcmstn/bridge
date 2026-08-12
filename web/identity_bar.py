"""
Shared "File & run identity" bar for bridge/web
====================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-12

NiceGUI analog of the Textual TUIs' #identity_bar/#identity_fields (see
DC/dc_hall_measurement_tui.py etc.): the filename preview plus the fields
that change every run (sample, device, cooldown, temperature setpoint),
pinned above the parameter grid instead of buried in a mid-form section.
Implemented once here rather than duplicated across 7 pages, same as
web/directory_picker.py and web/sample_picker.py.

The web pages have one field the TUI doesn't: a free-choice save directory
(directory_picker.py) — it sits on its own full-width row above the
identity_fields-equivalent 4-column grid, since a filesystem path doesn't
fit a narrow grid column and the TUI has no equivalent to align with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from nicegui import ui

from web.directory_picker import directory_field
from web.run_controller import optional_num_field, text_field
from web.sample_picker import sample_select


@dataclass
class IdentityBar:
    data_dir_input: ui.input
    sample_dropdown: ui.select
    refresh_sample_options: Callable[..., None]
    device_input: ui.input
    cooldown_input: ui.input
    temperature_input: ui.number
    filename_label: ui.label


def identity_bar(*, default_data_dir: str, default_sample: str,
                  default_device: str, default_cooldown: str,
                  default_temperature_K: Optional[float]) -> IdentityBar:
    """Renders the bar and returns handles for reading field values in
    collect_raw()/parse_state() and for updating the filename preview from
    refresh_summary() (`.filename_label.set_text(...)`)."""
    with ui.card().classes("w-full mb-3"):
        filename_label = ui.label().classes("font-bold")
        data_dir_input = directory_field("Data root directory", default_data_dir)
        with ui.grid(columns=4).classes("w-full gap-2"):
            sample_dropdown, refresh_sample_options = sample_select(
                lambda: data_dir_input.value, default=default_sample)
            device_input = text_field("Device (e.g. HB3, SV2)", default_device)
            cooldown_input = text_field("Cooldown (optional)", default_cooldown)
            temperature_input = optional_num_field(
                "Temp. setpoint (K, optional)", default_temperature_K,
                hint="Filename's T###K token only.")
    data_dir_input.on_value_change(lambda: refresh_sample_options())

    return IdentityBar(
        data_dir_input=data_dir_input,
        sample_dropdown=sample_dropdown,
        refresh_sample_options=refresh_sample_options,
        device_input=device_input,
        cooldown_input=cooldown_input,
        temperature_input=temperature_input,
        filename_label=filename_label,
    )
