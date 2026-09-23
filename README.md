# bridge

Instrument control suite for DC transport and MFLI lock-in
measurements in a spintronics lab: Keithley 6221/2182/2400/2450 current
sources, nanovoltmeters and gate sources, a Kepco BOP-GL electromagnet, a
Lake Shore 475 Gaussmeter, an Oxford Instruments MercuryiTC, and Zurich
Instruments MFLI lock-ins — driven either from a terminal UI (Textual) or a
browser (NiceGUI).

## Install

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

This installs the project's dependencies and editable-installs `bridge`
itself, so every module below is importable as a package (`instruments.*`,
`dc.*`, `mfli.*`, `web.*`) regardless of which script you run
or from where.

## Running it

One menu opens every measurement program, and each program is also directly
runnable on its own:

```
uv run python bridge_tui.py            # TUI menu: DC / MFLI / SOT programs + recent-runs history
uv run python sot/sot_pulsed_switching_tui.py   # any single program's TUI, standalone
uv run python web/app.py               # Browser front end, DC + MFLI + SOT suites, http://localhost:8080
```

The TUI menu mirrors the web landing page: one column per suite, each
program a card with its description, a collapsible wiring schematic and a
Launch button; quitting a program returns to the menu. Runs from both front
ends are listed in the same "Recent runs" table (`runs.db` next to the data
folder).

SOT pulsed switching is one form (TUI and web page) with two toggles — write
pulse from the 4200A PMU or the 6221, read as DC R_xy or lock-in harmonic —
covering the former SOTPS / SOT2H / SOT1I programs; the 4200A modes need a
KULT user module compiled on the 4200A — see
[`instruments/kult/`](instruments/kult/). The dual-harmonic program likewise
has an AC-source toggle (MFLI output or Keithley 6221; formerly HARM and
HARM6). Each mode still saves exactly the file format and type code it did
as a separate program. The nonlocal spin-current switching program (6221 +
2182A only) has a TUI and a web page.

The web front end runs alongside the TUI, not instead of it — pick whichever
is more convenient for a given session. Its port defaults to 8080; override
with `BRIDGE_WEB_PORT` if that's taken:

```
BRIDGE_WEB_PORT=8090 uv run python web/app.py
```

## Layout

```
bridge_tui.py  The TUI menu (every program, one entry point)
dc/            Keithley 6221/2182/2400 DC measurement programs + TUIs
mfli/          Zurich Instruments MFLI lock-in measurement programs + TUIs
sot/           Spin-orbit-torque / nonlocal pulsed-switching programs + TUIs
instruments/   Shared instrument drivers, data convention, run history, TUI helpers
web/           NiceGUI browser front end for the dc/, mfli/ and sot/ programs
docs/          Physics/theory background for specific measurements
```

## docs/

- [`docs/architecture.md`](docs/architecture.md) — cross-cutting layout:
  the three-layer (measurement / TUI / web) pattern, the
  `run_measurement()` contract, the instrument layer, and a "change X →
  touch these files" map. Start here to navigate the code.
- [`docs/data_convention.md`](docs/data_convention.md) — how every run is
  named and saved, and the `instruments/data_naming.py` writer/reader API.
- [`docs/current-reversal.md`](docs/current-reversal.md) — the V_odd/V_even
  current-reversal decomposition used by every DC measurement program.

## Troubleshooting

**Windows: `web/app.py` fails to bind its port with `OSError WinError 10013`**
("forbidden by its access protection") rather than the usual "address
already in use" — this is almost always Hyper-V/WSL2 reserving the port in
its dynamic port-exclusion range, not a conflicting process. Diagnose with
(elevated `cmd`):

```
netsh interface ipv4 show excludedportrange protocol=tcp
```

If port 8080 falls inside a listed range, either set `BRIDGE_WEB_PORT` to a
port outside all listed ranges, or free it with `net stop winnat` followed
by `net start winnat` (this regenerates the exclusion list; it may reassign
a new range on restart, so re-check afterwards).

## License

MIT — see [LICENSE](LICENSE).
