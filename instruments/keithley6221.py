"""
Keithley 6221 DC Current Source — shared connect/shutdown/acquisition helpers
===============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-06

The 6221 driver itself ships with pymeasure (pymeasure.instruments.keithley.
Keithley6221) — this module holds the connect/shutdown/acquisition wrapper
functions shared by the DC measurement programs.

Usage example (fixed sense-current case):
    from instruments.keithley6221 import SourceConfig, connect_source, shutdown_source

    src_cfg = SourceConfig(visa_resource="GPIB0::20::INSTR", sense_current_A=1e-3)
    source = connect_source(src_cfg)
    ...
    shutdown_source(source)

A program that instead sweeps the current itself (dc_iv_curve.py) keeps its
own SourceConfig shape (a sweep range, not a fixed sense current) and calls
the lower-level `connect()` directly with an explicit `initial_current_A`.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymeasure.instruments.keithley import Keithley2182, Keithley6221

from instruments.run_time import ARM_S, GPIB_TXN_S

log = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Keithley 6221 as a fixed DC sense-current source — the common case used
    by dc_hall_measurement.py, dc_gate_sweep.py and dc_spin_valve.py."""
    visa_resource: str     = "GPIB0::20::INSTR"
    sense_current_A: float = 1e-3    # Sense current magnitude [A]
    compliance_V: float    = 2.0     # Voltage compliance [V]
    source_delay_s: float  = 0.05    # Settle time after each current step [s]


@dataclass
class ACSourceConfig:
    """Keithley 6221 as an AC (sine) current source with a hardware phase
    marker on the Trigger Link, for external-reference lock-in detection
    (used by sot/sot_pulsed_switching_2h.py: an MFLI locks to this marker
    via its Aux Input, per the Zurich Instruments external-reference guide).

    `phasemarker_line` is the Trigger Link output line (1-6) the marker
    pulse (one edge per excitation cycle) appears on — set via SCPI
    (`:SOUR:WAVE:PMAR:OLIN`, confirmed against the Model 6220/6221
    Reference Manual, 622x-901-01 Rev. B) and read back by
    connect_ac_source() to confirm it took; no front-panel step needed. The
    Trigger Link's 8-pin DIN connector is a flat 1:1 map — DIN pin N =
    Trigger Link line N (pins 7/8 are ground) — so "line N" and "pin N" are
    the same thing. The 6221's own factory default is line 3 (same manual,
    p.7-10), and that's also what Zurich Instruments' own MFLI ↔ 6221
    external-reference guide wires (pin 3) — but default here is 1,
    matching this lab's actual cable (which only brings out pin 1 to the
    MFLI). Getting this wrong doesn't error — the 6221 happily outputs a
    clean marker on whichever line you ask for — it just means the real
    signal comes out on a different pin than whatever your cable taps, and
    what you see on the pin you're actually wired to is crosstalk, not the
    marker. Confirm against your own rig's cabling before trusting either
    default."""
    visa_resource: str        = "GPIB0::20::INSTR"
    amplitude_A: float        = 1e-4     # AC current amplitude, peak [A]
    offset_A: float           = 0.0      # DC offset added to the sine [A] (harmonic-Hall 2f sign calibration)
    frequency_Hz: float       = 977.0    # Excitation frequency [Hz] — avoid 50/60 Hz harmonics
    compliance_V: float       = 2.0      # Voltage compliance [V]
    ranging: str              = "best"   # "best" or "fixed"
    phasemarker_line: int     = 1        # Trigger Link line (1-6) the phase marker appears on — this lab's cable taps pin 1


def apply_sine_wave(source: Keithley6221, cfg: ACSourceConfig) -> None:
    """Write every WAVE parameter of the continuous sine + phase-marker read
    (not yet armed — a WAVE write after `waveform_arm()` only takes effect
    on the next arm). Shared by connect_ac_source() and any caller that
    re-arms the read after fire_wave_pulse() has overwritten the WAVE state
    with its one-cycle square pulse (marker off) — re-arming without this
    would fire that pulse again instead of the sine."""
    source.source_compliance = cfg.compliance_V
    source.waveform_function = "sine"
    source.waveform_amplitude = cfg.amplitude_A
    source.waveform_offset = cfg.offset_A
    source.waveform_frequency = cfg.frequency_Hz
    source.waveform_ranging = cfg.ranging
    source.waveform_use_phasemarker = True
    source.waveform_phasemarker_line = cfg.phasemarker_line
    source.waveform_duration_set_infinity()


def connect_ac_source(cfg: ACSourceConfig) -> Keithley6221:
    """Open and arm a Keithley 6221 as an AC current source with its phase
    marker enabled. Every WAVE parameter (function/amplitude/frequency/
    ranging/phase-marker state+line) must be set before `waveform_arm()` — a
    write after arming does not take effect until the next arm(), per the
    pymeasure driver's WAVE-mode example. `waveform_arm()` + `waveform_start()`
    is the official sequence for starting continuous sourcing (no separate
    `enable_source()` call needed — arming brings the output up itself); the
    duration is set to infinite so the wave runs until `waveform_abort()`.

    Reads the phase-marker line back after arming and raises if it didn't
    take — the usual cause is another Trigger Link function already owning
    that pin.
    """
    source = Keithley6221(cfg.visa_resource)
    source.reset()
    apply_sine_wave(source, cfg)
    source.waveform_arm()
    readback = source.waveform_phasemarker_line
    if int(readback) != cfg.phasemarker_line:
        # Leave nothing armed behind a raise — the caller never gets a
        # handle back to shut down otherwise, and this program's whole
        # safety story is "the 6221 is quiet unless we say otherwise".
        source.waveform_abort()
        source.shutdown()
        raise RuntimeError(
            f"6221 phase-marker line rejected: asked for {cfg.phasemarker_line}, "
            f"device reports {readback}. Another Trigger Link function likely "
            "already owns that pin.")
    source.waveform_start()
    log.info("Keithley 6221 connected: %s  AC I=%.4g A peak  f=%.4g Hz  "
              "compliance=%.2f V  phase marker → Trigger Link pin %d",
              cfg.visa_resource, cfg.amplitude_A, cfg.frequency_Hz,
              cfg.compliance_V, cfg.phasemarker_line)
    return source


def shutdown_ac_source(source: Keithley6221) -> None:
    """Stop the AC wave and disable the 6221's output."""
    source.waveform_abort()
    source.shutdown()
    log.info("Keithley 6221 AC wave stopped, output disabled")


def ac_source_restart_s() -> float:
    """Modelled wall time of one 6221 rebuild between files: shutdown_ac_source()
    (ABORt + shutdown = 2 writes) + connect_ac_source() (*RST, 9 WAVE writes,
    ARM, phase-marker readback, START = 13 transactions + ARM_S) + the caller's
    output-off (ABORt + OUTPUT OFF = 2)."""
    return (2 + 13 + 2) * GPIB_TXN_S + ARM_S


@dataclass
class PulseWaveConfig:
    """A single HARDWARE-TIMED current pulse via WAVE mode's square function,
    run for exactly one cycle — the 6221's real, verified, no-2182-required
    single-pulse mechanism. Confirmed against the Model 6220/6221 User's
    Manual (622x-900-01 Rev. C, Section 7 "Wave Functions"):

      * A square wave swings between (offset - amplitude) and
        (offset + amplitude) (manual Fig. 7-1). fire_wave_pulse() below sets
        offset = amplitude = pulse_current_A / 2, so the wave sits at 0 and
        rises to pulse_current_A for the duty-cycle "high" fraction of one
        period, then returns to 0 — a clean unipolar pulse, not the
        alternating-polarity triplet Pulse Delta mode uses.
      * "The output will turn off after the currently set duration period
        has expired" (manual, Section 7 "Duration") — with
        waveform_duration_cycles = 1, that is real hardware-timed
        auto-termination after exactly one pulse, not a software sleep
        guess. fire_wave_pulse() polls the OUTPUT state for this rather than
        trusting elapsed wall time.
      * This is DIFFERENT from Pulse Delta / Differential Conductance
        (`:SOUR:PDEL`, `:SOUR:DCON`), which require a 2182/2182A wired via
        the rear-panel Trigger Link cable and are architecturally built
        around alternating-polarity offset-cancelling measurement, not a
        single deliberate-polarity write pulse. Nothing here touches a 2182.

    ``pulse_current_A`` may be negative. The wave then swings between
    -|I| and 0 with the 0 A half first, so a negative pulse is expected to
    arrive one ``width_s`` later than a positive one (scope-check once).

    Duty cycle is fixed at 50% internally so ``width_s`` alone determines
    the frequency this programs (frequency = 1 / (2 x width_s)) — pick
    ``width_s`` so that frequency stays inside the WAVE subsystem's 1 mHz to
    100 kHz range (width_s below ~5 µs pushes frequency past 100 kHz and the
    instrument will clip it).

    Spec floor: manual quotes "Settable to 1 µs min. pulse duration" for
    square-wave duty cycle, footnoted "minimum realizable duty cycle is
    limited by current range response and load impedance" — the datasheet's
    own headline number is 5 µs. Both are best-case; the true floor at your
    actual pulse amplitude/range is whatever fire_wave_pulse() measures and
    returns as ``pulse_width_measured_s`` — check that, not this docstring,
    before trusting a sub-10 µs pulse.
    """
    pulse_current_A: float = 5e-3    # peak pulse current [A] — pulse rises from 0 to this
    width_s: float          = 1e-3    # pulse width (duty-cycle "high" time) [s]
    compliance_V: float     = 5.0     # pulse voltage compliance [V]
    ranging: str             = "best"  # "best" or "fixed"


def fire_wave_pulse(source: Keithley6221, cfg: PulseWaveConfig,
                     stop_event: Optional[threading.Event] = None,
                     timeout_margin_s: float = 0.5) -> dict:
    """Fire ONE hardware-timed current pulse — see PulseWaveConfig's
    docstring for the mechanism and why it's the 6221's real supported
    single-pulse capability, not Pulse Delta and not a software sleep loop.

    Every WAVE parameter must be set before ``waveform_arm()`` (same
    pymeasure/SCPI ordering rule as connect_ac_source() — a write after
    arming doesn't take effect until the next arm). Disables the phase
    marker for the pulse (it's only meaningful during the continuous AC
    read elsewhere in this codebase) and always sets every parameter
    explicitly — never assumes a retained level from a prior AC-read or
    pulse cycle.

    Completion is detected by polling ``source_enabled`` (``OUTPUT?``)
    rather than trusting elapsed wall time or SRQ (SRQ needs GPIB; this
    driver's configs default to GPIB but nothing stops a TCPIP/LAN
    resource string) — bounded by the pulse's own period plus
    ``timeout_margin_s`` so a stuck poll can't hang a run forever; a
    timeout still returns rather than raising; the caller sees it via a
    ``pulse_width_measured_s`` that doesn't shrink back down. ``waveform_
    abort()`` + ``disable_source()`` run unconditionally afterward so the
    instrument is left in the same disarmed state the caller's next action
    (another pulse, or connect_ac_source()'s continuous read) expects.

    Returns ``{"pulse_width_measured_s": <elapsed to the instrument's own
    auto-off, or the timeout bound>}``.
    """
    half = cfg.pulse_current_A / 2.0
    period_s = 2.0 * cfg.width_s
    frequency_Hz = 1.0 / period_s

    source.source_compliance = cfg.compliance_V
    source.waveform_function = "square"
    # pymeasure's waveform_amplitude truncates to [2e-12, 0.105]: a negative
    # value would silently become ~0 (a constant half-height, double-width
    # "pulse"). Amplitude is therefore always |I|/2; the sign lives in the offset.
    source.waveform_amplitude = abs(half)
    source.waveform_offset = half
    source.waveform_dutycycle = 50.0
    source.waveform_frequency = frequency_Hz
    source.waveform_ranging = cfg.ranging
    source.waveform_use_phasemarker = False
    source.waveform_duration_cycles = 1
    source.waveform_arm()
    t0 = time.monotonic()
    source.waveform_start()

    timeout_s = period_s + timeout_margin_s
    poll_interval_s = min(0.001, timeout_s)
    while time.monotonic() - t0 < timeout_s:
        if stop_event is not None and stop_event.is_set():
            break
        if not source.source_enabled:
            break
        time.sleep(poll_interval_s)
    elapsed = time.monotonic() - t0

    source.waveform_abort()
    source.disable_source()
    return {"pulse_width_measured_s": elapsed}


def wave_pulse_s(width_s: float) -> float:
    """Modelled wall time of one ``fire_wave_pulse()``: 9 WAVE config writes, ARM,
    START, >= 1 OUTPUT? poll, ABORt, OUTPUT OFF (14 transactions) + ARM_S, and the
    hardware-timed one-cycle wave itself (period = 2 x width_s)."""
    return 14 * GPIB_TXN_S + ARM_S + 2.0 * width_s


def connect(visa_resource: str, compliance_V: float, source_delay_s: float,
            initial_current_A: float = 0.0) -> Keithley6221:
    """Open and configure a Keithley 6221 as a DC current source."""
    source = Keithley6221(visa_resource)
    source.reset()
    source.source_auto_range = True
    source.source_compliance = compliance_V
    source.source_delay = source_delay_s
    source.source_current = initial_current_A
    source.enable_source()
    log.info("Keithley 6221 connected: %s  I=%.4g A  compliance=%.2f V",
              visa_resource, initial_current_A, compliance_V)
    return source


def connect_source(cfg: SourceConfig) -> Keithley6221:
    """connect(), parameterized by a SourceConfig — the fixed sense-current case."""
    return connect(cfg.visa_resource, cfg.compliance_V, cfg.source_delay_s,
                    initial_current_A=cfg.sense_current_A)


def shutdown_source(source: Keithley6221, zero_first: bool = True) -> None:
    """
    Disable the 6221's output.

    `zero_first`, if True (the default), sets the current to 0 A immediately
    before disabling — pass False if the current was already ramped down
    gently (e.g. via ramp_current_to_zero()) beforehand, so it isn't jumped
    a second time.
    """
    if zero_first:
        source.source_current = 0.0
    source.shutdown()
    log.info("Keithley 6221 output disabled")


def ramp_current_to_zero(source: Keithley6221, step_A: float = 1e-4, delay_s: float = 0.02) -> None:
    """Step the sourced current back to 0 A gradually rather than jumping — gentler on the DUT."""
    current = source.source_current
    log.info("Ramping current from %.4g A to 0 A ...", current)
    n_steps = max(1, int(abs(current) / step_A))
    for i in np.linspace(current, 0.0, n_steps + 1)[1:]:
        source.source_current = float(i)
        time.sleep(delay_s)


def reversal_avg_s(n_reversals: int, source_delay_s: float, read_s: float,
                   n_channels: int = 1, channel_settle_s: float = 0.0) -> float:
    """Modelled wall time of one ``acquire_reversal_averaged_voltage()`` call.

    Per polarity flip: one source write + ``source_delay_s`` sleep, then
    `n_channels` reads of `read_s` each (``keithley2182.read_time_s``), each
    preceded by a channel-mux write + ``channel_settle_s`` when n_channels > 1;
    two flips per reversal, plus the final write leaving the source at +I.
    The delay and the read ADD -- they are sequential, not overlapping."""
    per_read = read_s + ((GPIB_TXN_S + channel_settle_s) if n_channels > 1 else 0.0)
    half = GPIB_TXN_S + source_delay_s + n_channels * per_read
    return n_reversals * 2 * half + GPIB_TXN_S


def acquire_reversal_averaged_voltage(
    source: Keithley6221,
    voltmeter: Keithley2182,
    sense_current_A: float,
    n_reversals: int,
    stop_event: Optional[threading.Event] = None,
    source_delay_s: float = 0.0,
    channels: tuple = (1,),
    channel_settle_s: float = 0.0,
) -> dict:
    """
    Reverse the sense current n_reversals times and decompose the resulting
    voltage into its odd part (V_odd, the resistive signal) and even part
    (V_even, offset plus any genuinely even-in-current physics) — both are
    returned rather than just V_odd; see docs/current-reversal.md for why.

    `source_delay_s`, if given, is slept after each polarity flip before the
    voltmeter is read — a plain property write does not itself wait for the
    reversed current to settle.

    `channels`, if more than one (e.g. `(1, 2)` to read R_xy and R_xx off
    the same 2182 alongside each other), interleaves a channel-mux read
    within each polarity rather than doubling the number of current
    reversals: `+I -> read ch1 -> read ch2 -> -I -> read ch1 -> read ch2`.
    `channel_settle_s` is slept after each `active_channel` switch — this
    is the 2182's channel-mux settle time, distinct from `source_delay_s`
    (which settles the *current source* after a polarity flip). The
    single-channel default (`channels=(1,)`) never touches
    `voltmeter.active_channel` at all, so existing single-channel callers
    are unaffected.

    Leaves the source at +sense_current_A on return. If `stop_event` fires
    partway through, returns the mean/sem of whatever pairs were already
    collected. ``mean``/``even_mean`` are the odd/even components; ``sem``/
    ``even_sem`` are the standard error of each of those means (sample
    stdev over the reversal pairs / sqrt(n); ``nan`` if only one pair was
    collected). The raw pair-to-pair scatter is ``sem * sqrt(n_reversals)``.

    Returns the flat ``{"mean","sem","even_mean","even_sem","n_reversals"}``
    dict (unchanged shape) when `channels` has exactly one entry; otherwise
    a ``{channel: {"mean","sem","even_mean","even_sem"}, ...,
    "n_reversals": n}`` dict keyed by channel number.
    """
    multi = len(channels) > 1
    samples_odd = {ch: np.empty(n_reversals) for ch in channels}
    samples_even = {ch: np.empty(n_reversals) for ch in channels}
    n_used = 0

    def _read_channels() -> dict:
        v = {}
        for ch in channels:
            if multi:
                voltmeter.active_channel = ch
                if channel_settle_s > 0:
                    time.sleep(channel_settle_s)
            v[ch] = voltmeter.voltage
        return v

    for i in range(n_reversals):
        source.source_current = sense_current_A
        if source_delay_s > 0:
            time.sleep(source_delay_s)
        v_plus = _read_channels()

        source.source_current = -sense_current_A
        if source_delay_s > 0:
            time.sleep(source_delay_s)
        v_minus = _read_channels()

        for ch in channels:
            samples_odd[ch][i] = (v_plus[ch] - v_minus[ch]) / 2.0
            samples_even[ch][i] = (v_plus[ch] + v_minus[ch]) / 2.0
        n_used = i + 1

        if stop_event is not None and stop_event.is_set():
            break

    source.source_current = sense_current_A

    def _decompose(ch: int) -> dict:
        used_odd = samples_odd[ch][:n_used]
        used_even = samples_even[ch][:n_used]
        if n_used >= 2:
            sqrt_n = np.sqrt(n_used)
            sem_odd = float(np.std(used_odd, ddof=1) / sqrt_n)
            sem_even = float(np.std(used_even, ddof=1) / sqrt_n)
        else:
            sem_odd = sem_even = float("nan")
        return {
            "mean": float(np.mean(used_odd)),
            "sem": sem_odd,
            "even_mean": float(np.mean(used_even)),
            "even_sem": sem_even,
        }

    if not multi:
        result = _decompose(channels[0])
        result["n_reversals"] = n_used
        return result

    result = {ch: _decompose(ch) for ch in channels}
    result["n_reversals"] = n_used
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pulse Delta write pulse + Delta read  (6221 + 2182A on the 6221's RS-232)
# ─────────────────────────────────────────────────────────────────────────────
# Both modes need the 2182A on a null-modem RS-232 cable to the 6221 plus the
# Trigger Link (6221 User's Manual 622x-900-01, Section 5, Fig. 5-2/5-4).
# Selecting RS-232 on the 2182A disables its GPIB, so the PC only ever talks
# to the 6221 here; the 2182A is reached through SYST:COMM:SER:SEND. pymeasure
# has no Delta / Pulse Delta support — these are raw SCPI from Table 5-2/5-3.

# ponytail: 50 Hz mains (lab is in Sweden); Pulse Delta cycles are line-synced, 60 Hz → 1/60
LINE_PERIOD_S = 0.02
PDEL_WIDTH_RANGE_S = (50e-6, 12e-3)     # SOUR:PDEL:WIDT limits (manual Table 5-3)
DELTA_NPLC_RANGE = (1, 50)              # 2182A rate must be an integer PLC for Delta, else 1 PLC


@dataclass
class PulseDeltaConfig:
    """ONE write pulse via Pulse Delta: count 1, I-Lo = 0, so the low "pulses"
    of the low-high-low cycle are 0 A and the output is a single 0 → I → 0
    pulse, synchronised to the power line. Pulse Delta has no rise-time
    setting; the only edge controls are the output response speed
    (``OUTP:RESP FAST`` ≈ 2-3 µs / ``SLOW`` ≈ 100 µs settling, manual
    Appendix A) and the analog filter (``CURR:FILT``, 33 pF across the output
    — only slows a high-impedance load, manual "Analog filter")."""
    width_s: float          = 1e-3    # 50 µs - 12 ms [s]
    compliance_V: float     = 5.0     # pulse voltage compliance [V]
    response: str           = "FAST"  # OUTP:RESP FAST | SLOW
    analog_filter: bool     = False   # CURR:FILT ON | OFF
    source_delay_s: float   = 16e-6   # SOUR:PDEL:SDEL — when the (discarded) 2182A read triggers
    interval_plc: int       = 5       # SOUR:PDEL:INT — cycle length; 5 PLC is the minimum


def check_2182a_on_serial(source: Keithley6221) -> None:
    """Raise unless the 6221 sees a Delta-capable 2182A on its RS-232 port."""
    if source.ask("SOUR:PDEL:NVPR?").strip() != "1":
        raise RuntimeError(
            "6221 does not see a 2182A on its RS-232 port (SOUR:PDEL:NVPR? = 0). Set the "
            "2182A interface to RS-232, connect it to the 6221 with a null-modem cable + the "
            "Trigger Link, and make sure the 6221 itself is on GPIB/LAN, not RS-232.")


def _wait_buffer(source: Keithley6221, n: int, timeout_s: float,
                 stop_event: Optional[threading.Event]) -> bool:
    """Poll TRAC:POIN:ACT? until the 6221 buffer holds `n` readings. True once
    it does; False on timeout or abort (the caller disarms either way)."""
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if int(float(source.ask("TRAC:POIN:ACT?"))) >= n:
            return True
        if stop_event is not None and stop_event.is_set():
            return False
        time.sleep(0.01)
    return False


def _arm(source: Keithley6221, mode: str) -> None:
    """SOUR:<mode>:ARM, check it took (a settings conflict leaves it unarmed), INIT."""
    source.write(f"SOUR:{mode}:ARM")
    if source.ask(f"SOUR:{mode}:ARM?").strip() != "1":
        source.write("SOUR:SWE:ABOR")
        raise RuntimeError(f"6221 refused to arm {mode} — check SYST:ERR? (settings conflict?)")
    source.write("INIT:IMM")


def fire_pdelta_pulse(source: Keithley6221, cfg: PulseDeltaConfig, pulse_current_A: float,
                      stop_event: Optional[threading.Event] = None,
                      timeout_margin_s: float = 0.5) -> dict:
    """Fire one Pulse Delta write pulse of `pulse_current_A` (signed, ±105 mA)
    and wait until the cycle is done. The Pulse Delta reading the 2182A takes
    is never fetched — this is a write, not a measurement. Leaves the 6221
    disarmed (SOUR:SWE:ABOR, output off). Returns ``{"pulse_completed": bool}``
    — False means the buffer never saw the cycle's reading within one cycle +
    `timeout_margin_s` (or an abort)."""
    source.write("SOUR:SWE:ABOR")
    source.write(f"CURR:COMP {cfg.compliance_V:g}")
    source.write(f"OUTP:RESP {cfg.response}")
    source.write(f"CURR:FILT {'ON' if cfg.analog_filter else 'OFF'}")
    for cmd in (f"HIGH {pulse_current_A:g}", "LOW 0", f"WIDT {cfg.width_s:g}",
                f"SDEL {cfg.source_delay_s:g}", "COUN 1", "RANG BEST",
                f"INT {cfg.interval_plc:d}", "SWE OFF", "LME 1"):
        source.write(f"SOUR:PDEL:{cmd}")
    source.write("TRAC:CLE")
    source.write("TRAC:POIN 1")
    _arm(source, "PDEL")
    done = _wait_buffer(source, 1, cfg.interval_plc * LINE_PERIOD_S + timeout_margin_s, stop_event)
    source.write("SOUR:SWE:ABOR")
    if not done:
        log.warning("Pulse Delta cycle did not complete (I=%.4g A) — the pulse may not have fired.",
                    pulse_current_A)
    return {"pulse_completed": done}


def acquire_delta_voltage(source: Keithley6221, sense_current_A: float, n_readings: int,
                          delay_s: float, nplc: int,
                          stop_event: Optional[threading.Event] = None,
                          timeout_margin_s: float = 2.0) -> dict:
    """The R_xy read in Delta mode: the 6221 alternates ±`sense_current_A`, the
    2182A (rate `nplc`, an integer) is triggered over the Trigger Link after
    `delay_s` each flip, and the 6221 builds `n_readings` 3-point delta
    voltages into its buffer — the instrument's own current reversal (cancels
    thermal EMF / offset). Waits for every reading, then disarms.

    Returns ``{"mean", "sem", "n_readings"}`` (V_odd only — Delta mode has no
    V_even). Consecutive delta readings share A/D samples (moving average),
    so ``sem`` is optimistic. Raises if no reading arrived."""
    source.write("SOUR:SWE:ABOR")
    # Pulse Delta's arm set its own 2182A integration rate — restore ours.
    source.write(f'SYST:COMM:SER:SEND "VOLT:NPLC {nplc:d}"')
    source.write(f"SOUR:DELT:HIGH {sense_current_A:g}")     # I-Lo follows as -HIGH
    source.write(f"SOUR:DELT:DEL {delay_s:g}")
    source.write(f"SOUR:DELT:COUN {n_readings:d}")
    source.write("SOUR:DELT:CAB ON")
    source.write("TRAC:CLE")
    source.write(f"TRAC:POIN {n_readings:d}")
    source.write("FORM:ELEM READ")
    _arm(source, "DELT")
    _wait_buffer(source, n_readings, delta_read_s(n_readings, delay_s, nplc) * 2 + timeout_margin_s,
                 stop_event)
    raw = source.ask("TRAC:DATA?").strip()
    source.write("SOUR:SWE:ABOR")
    v = np.array([float(x) for x in raw.split(",") if x.strip()])
    if v.size == 0:
        raise RuntimeError("Delta read returned no readings — check the 2182A link / compliance.")
    sem = float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size >= 2 else float("nan")
    return {"mean": float(np.mean(v)), "sem": sem, "n_readings": int(v.size)}


def pdelta_pulse_s(cfg: PulseDeltaConfig) -> float:
    """Modelled wall time of one ``fire_pdelta_pulse()``: 18 writes/queries + the
    arm (the 6221 configures the 2182A over RS-232) + one line-synced cycle."""
    return 18 * GPIB_TXN_S + ARM_S + cfg.interval_plc * LINE_PERIOD_S


def delta_read_s(n_readings: int, delay_s: float, nplc: int) -> float:
    """Modelled wall time of the Delta acquisition itself: n + 2 A/D conversions
    (the first delta needs 3 points), each after one polarity flip + `delay_s`."""
    return (n_readings + 2) * (delay_s + nplc * LINE_PERIOD_S)
