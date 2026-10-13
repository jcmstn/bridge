# bridge

Instrument control suite for DC transport and MFLI lock-in
measurements in a spintronics lab: Keithley 6221/2182/2400/2450 current
sources, nanovoltmeters and gate sources, a Kepco BOP-GL electromagnet, a
Lake Shore 475 Gaussmeter, an Oxford Instruments MercuryiTC, and Zurich
Instruments MFLI lock-ins — driven from a terminal UI (Textual).

## Install

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

This installs the project's dependencies and editable-installs `bridge`
itself, so every module below is importable as a package (`instruments.*`,
`dc.*`, `mfli.*`, `sot.*`) regardless of which script you run
or from where.

## Running it

One menu opens every measurement program, and each program is also directly
runnable on its own:

```
uv run python bridge_tui.py            # TUI menu: DC / MFLI / SOT programs + recent-runs history
uv run python sot/sot_pulsed_switching_tui.py   # any single program's TUI, standalone
```

The TUI menu has one column per suite, each
program a card with its description, a collapsible wiring schematic and a
Launch button; quitting a program returns to the menu. Every run is listed
in the "Recent runs" table (`runs.db` next to the data folder).

SOT pulsed switching is one form with two toggles — write
pulse from the 4200A PMU or the 6221, read as DC R_xy or lock-in harmonic —
covering the former SOTPS / SOT2H / SOT1I programs; the 4200A modes need a
KULT user module compiled on the 4200A — see
[`instruments/kult/`](instruments/kult/). The dual-harmonic program likewise
has an AC-source toggle (MFLI output or Keithley 6221; formerly HARM and
HARM6). Each mode still saves exactly the file format and type code it did
as a separate program. The nonlocal spin-current switching program (6221 +
2182A only) has its own TUI.

## Layout

```
bridge_tui.py  The TUI menu (every program, one entry point)
dc/            Keithley 6221/2182/2400 DC measurement programs + TUIs
mfli/          Zurich Instruments MFLI lock-in measurement programs + TUIs
sot/           Spin-orbit-torque / nonlocal pulsed-switching programs + TUIs
instruments/   Shared instrument drivers, data convention, run history, TUI helpers
docs/          Physics/theory background for specific measurements
```

## docs/

- [`docs/architecture.md`](docs/architecture.md) — cross-cutting layout:
  the three-layer (TUI / measurement / instruments) pattern, the
  `run_measurement()` contract, the instrument layer, and a "change X →
  touch these files" map. Start here to navigate the code.
- [`docs/data_convention.md`](docs/data_convention.md) — how every run is
  named and saved, and the `instruments/data_naming.py` writer/reader API.
- [`docs/current-reversal.md`](docs/current-reversal.md) — the V_odd/V_even
  current-reversal decomposition used by every DC measurement program.

## License

MIT — see [LICENSE](LICENSE).
