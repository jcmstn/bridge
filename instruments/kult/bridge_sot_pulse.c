/* USRLIB MODULE INFORMATION

	MODULE NAME: bridge_sot_pulse
	MODULE RETURN TYPE: int
	NUMBER OF PARMS: 20
	ARGUMENTS:
		PulseWidth,	double,	Input,	100e-9,	60e-9,	.999999
		RiseTime,	double,	Input,	20e-9,	20e-9,	.033
		FallTime,	double,	Input,	20e-9,	20e-9,	.033
		Period,	double,	Input,	1e-3,	120e-9,	1
		Delay,	double,	Input,	0,	0,	.999999
		SampleRate,	double,	Input,	200E6,	1000,	200E6
		MeasStartPerc,	double,	Input,	.75,	0,	1
		MeasStopPerc,	double,	Input,	.90,	0,	1
		PulseAverage,	int,	Input,	1,	1,	10000
		DUTRes,	double,	Input,	1E3,	1,	1E6
		VRange,	double,	Input,	10,	5,	40
		IRange,	double,	Input,	.01,	100e-9,	.8
		AmplitudeV,	double,	Input,	0.5,	-40,	40
		BaseV,	double,	Input,	0,	-40,	40
		Chan,	int,	Input,	1,	1,	2
		PMU_ID,	char *,	Input,	"PMU1",	,
		V_Ampl,	double,	Output,	,	,
		I_Ampl,	double,	Output,	,	,
		V_Base,	double,	Output,	,	,
		I_Base,	double,	Output,	,	,
	INCLUDES:
#include "keithley.h"
BOOL LPTIsInCurrentConfiguration(char* hrid);

	END USRLIB MODULE INFORMATION
*/
/* USRLIB MODULE HELP DESCRIPTION
<!--MarkdownExtra-->
<link rel="stylesheet" type="text/css" href="http://clariusweb/HelpPane/stylesheet.css">

Module: bridge_sot_pulse
========================

Description
-----------

Fire **one** pulse burst at **one** amplitude on a single 4225-PMU channel and return
the voltage and current spot means for the pulse top and the pulse base. Written for
the `bridge` SOT pulsed-switching program (`sot/sot_pulsed_switching.py`), driven over
KXCI in UL mode.

Derived from Keithley's `PMU_1Chan_Sweep_Example`, with four deliberate differences:

1. **No sweep.** One `AmplitudeV` per call (`pulse_vhigh`/`pulse_vlow`), not
   Start/Stop/Step. The Python side owns the amplitude loop, so a sweep here would be a
   second loop fighting the first.
2. **Scalar outputs**, not `D_ARRAY_T`. Four doubles come back cleanly through KXCI
   `GN`; arrays would drag in the six array-size arguments for nothing.
3. **The RPM pathway is routed back to the SMU on every exit path**, including every
   error. This is the important one — see below.
4. **Self-contained includes.** Only `keithley.h`, so the module drops into a brand new
   user library without needing `PMU_examples_ulib_internal.h` on the include path.

Why the route-back matters
--------------------------

`PMU_1Chan_Sweep_Example` calls
`rpm_config(InstId, Chan, KI_RPM_PATHWAY, KI_RPM_PULSE)` on entry and never routes back,
and leaves `pulse_output` enabled. After it runs, the RPM is parked on the pulse pathway
and **the SMU wired through that RPM cannot reach the DUT at all** — a DC read straight
after a pulse returns an open circuit.

The `bridge` measurement pulses and then reads R_xy with SMU1 through the same RPM, so
this module always finishes with:

    pulse_output(InstId, Chan, 0);
    rpm_config(InstId, Chan, KI_RPM_PATHWAY, KI_RPM_SMU);

Doing it here rather than from Python is deliberate: a Python-side switch cannot cover
the case where the module itself errors out partway.

Inputs
------
PulseWidth
: Pulse width (FWHM), seconds. Range 60 ns to 999.999 ms.

RiseTime, FallTime
: 0-100% transition times, seconds. 10 V range: 20 ns to 33 ms.
  40 V range: 100 ns to 33 ms. Slow transitions are slew-rate limited.

Period
: Pulse period, seconds. 10 V range: 120 ns to 1 s. 40 V range: 280 ns to 1 s.
  Must be at least Delay + RiseTime + PulseWidth + FallTime.

Delay
: Time before the rise, seconds.

SampleRate
: Samples per second, in steps of 200e6/n. Keep the sample count under ~1e6 per
  waveform.

MeasStartPerc, MeasStopPerc
: Spot-mean measure window as a fraction of the pulse top, where the pulse top is
  `PulseWidth - 0.5*RiseTime - 0.5*FallTime`. Typical 0.75 and 0.90.

PulseAverage
: Number of pulses output per call and averaged into one spot mean. This is the burst
  length: `PulseAverage = 5` outputs 5 pulses and means them together.

DUTRes
: DUT resistance in ohms, used for the load-line correction of the PMU's 50 ohm output.
  Set it near the real channel resistance (run `sot_dc_characterization` first).
  Not meaningful on RPM current ranges of 1 mA or below.

VRange
: PMU voltage range. Valid: 10 or 40.

IRange
: Current measure range. **With a 4225-RPM on the 10 V range the maximum is 0.01 A** —
  a pulse current above that reads back overflowed, not errored, because this module
  sets `KI_LIM_MODE` to `KI_VALUE`. Valid ranges:
  PMU 10 V: 0.01, 0.2. PMU 40 V: 100e-6, 0.01, 0.8.
  RPM 10 V: 100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 0.01.

AmplitudeV
: Pulse top voltage. This is a *forced voltage at the PMU output*, not at the DUT: on a
  2-wire path the drop across cables and contacts is included, which is exactly why
  I_Ampl is the physically meaningful number.

BaseV
: Quiescent voltage between pulses. Normally 0.

Chan
: PMU channel, 1 or 2.

PMU_ID
: PMU card name, e.g. "PMU1" (lowest-numbered slot).

Outputs
-------
V_Ampl, I_Ampl
: Voltage and current spot means measured during the pulse top.

V_Base, I_Base
: Voltage and current spot means measured during the base level between pulses. A free
  leakage / drift check — these should sit near (BaseV, 0).

Return values
-------------

Value  | Description
------ | -----------
0      | OK.
-122   | pulse_ranges(): illegal parameter. IRange is not valid for VRange (see IRange above).
-824   | pulse_exec(): invalid pulse timing for the chosen VRange. Increase the timing parameters.
-829   | Base + amplitude exceeds the present voltage range.
-17001 | PMU_ID is not in the system configuration. Check the card name in KCON.
-17002 | Failed to get a card handle for PMU_ID.

Any non-zero return still leaves the RPM routed back to the SMU and the pulse output
disabled.

	END USRLIB MODULE HELP DESCRIPTION */
/* USRLIB MODULE PARAMETER LIST */
#include "keithley.h"
BOOL LPTIsInCurrentConfiguration(char* hrid);

/* Error codes, matching the values Keithley's PMU examples use. Defined here rather
   than pulled from PMU_examples_ulib_internal.h so this module compiles standalone in
   a new user library. */
#define BRIDGE_ERR_WRONGCARDID      17001
#define BRIDGE_ERR_CARDHANDLEFAIL   17002

int bridge_sot_pulse( double PulseWidth, double RiseTime, double FallTime, double Period, double Delay, double SampleRate, double MeasStartPerc, double MeasStopPerc, int PulseAverage, double DUTRes, double VRange, double IRange, double AmplitudeV, double BaseV, int Chan, char *PMU_ID, double *V_Ampl, double *I_Ampl, double *V_Base, double *I_Base )
{
/* USRLIB MODULE CODE */
    int status = 0;
    int InstId = 0;
    int rpm_routed = 0;
    double elapsedt;
    int verbose = 0;

    /* pulse_fetch returns two records for one amplitude with both levels
       acquired: [0] = pulse top, [1] = pulse base. */
    double Vbuf[2], Ibuf[2], Tbuf[2];
    unsigned long Sbuf[2];

    /* Blank the outputs first, so a failed call never looks like data. */
    *V_Ampl = 0.0;
    *I_Ampl = 0.0;
    *V_Base = 0.0;
    *I_Base = 0.0;

    if ( !LPTIsInCurrentConfiguration(PMU_ID) )
    {
        printf("bridge_sot_pulse: instrument %s is not in the system configuration", PMU_ID);
        return -BRIDGE_ERR_WRONGCARDID;
    }

    getinstid(PMU_ID, &InstId);
    if ( -1 == InstId )
        return BRIDGE_ERR_CARDHANDLEFAIL;

//    verbose = 1;      //Enable printf messages to msgcon for troubleshooting

    /* Route the 4225-RPM (if fitted) onto the pulse pathway. From here on EVERY
       exit goes through cleanup:, which routes it back to the SMU -- the DC R_xy
       read that follows this pulse reaches the DUT through that same RPM. */
    status = rpm_config(InstId, Chan, KI_RPM_PATHWAY, KI_RPM_PULSE);
    if ( status )
        goto cleanup;
    rpm_routed = 1;

    status = pg2_init(InstId, PULSE_MODE_PULSE);
    if ( status )
        goto cleanup;

    /* Return the actual value on measurement overflow rather than an error --
       an overflowed I_Ampl is visible in the data, a failed pulse is not. */
    status = setmode(InstId, KI_LIM_MODE, KI_VALUE);
    if ( status )
        goto cleanup;

    status = pulse_sample_rate(InstId, SampleRate);
    if ( status )
        goto cleanup;

    status = pulse_ranges(InstId, Chan, VRange, PULSE_MEAS_FIXED, VRange, PULSE_MEAS_FIXED, IRange);
    if ( status )
        goto cleanup;

    status = pulse_load(InstId, Chan, DUTRes);
    if ( status )
        goto cleanup;

    /* No sweep: the amplitude is set directly. pulse_vhigh is the amplitude
       setter whenever the amplitude is not the swept parameter. */
    status = pulse_vlow(InstId, Chan, BaseV);
    if ( status )
        goto cleanup;

    status = pulse_vhigh(InstId, Chan, AmplitudeV);
    if ( status )
        goto cleanup;

    status = pulse_burst_count(InstId, Chan, 1);
    if ( status )
        goto cleanup;

    status = pulse_source_timing(InstId, Chan, Period, Delay, PulseWidth, RiseTime, FallTime);
    if ( status )
        goto cleanup;

    /* PulseAverage pulses are output and meaned into one spot mean. */
    status = pulse_meas_timing(InstId, Chan, MeasStartPerc, MeasStopPerc, PulseAverage);
    if ( status )
        goto cleanup;

    /* pulse_meas_sm(Card, ch, Measuretype, acqVHigh, acqVLow, acqIHigh, acqILow,
       acqTimeStamp, LLEComp). LLE comp is off: PULSE_MODE_SIMPLE below does not
       support it. */
    status = pulse_meas_sm(InstId, Chan, PULSE_ACQ_PBURST, TRUE, TRUE, TRUE, TRUE, TRUE, 0);
    if ( status )
        goto cleanup;

    status = pulse_output(InstId, Chan, 1);
    if ( status )
        goto cleanup;

    if ( verbose )
        printf("bridge_sot_pulse: chan=%d ampl=%g V base=%g V width=%g s", Chan, AmplitudeV, BaseV, PulseWidth);

    /* Simple mode: fixed current ranges, no LLE comp, no IVP thresholds -- the
       shortest test time, which matters when this runs once per write/read cycle. */
    status = pulse_exec(PULSE_MODE_SIMPLE);
    if ( status )
        goto cleanup;

    while ( pulse_exec_status(&elapsedt) == 1 )
        Sleep(1);

    status = pulse_fetch(InstId, Chan, 0, 1, Vbuf, Ibuf, Tbuf, Sbuf);
    if ( status )
        goto cleanup;

    *V_Ampl = Vbuf[0];
    *I_Ampl = Ibuf[0];
    *V_Base = Vbuf[1];
    *I_Base = Ibuf[1];

cleanup:
    /* Leave the channel quiet and the RPM back on the SMU pathway. Both calls are
       deliberately unchecked: this runs on the error path too, and the original
       status is what the caller needs to see. */
    pulse_output(InstId, Chan, 0);
    if ( rpm_routed )
        rpm_config(InstId, Chan, KI_RPM_PATHWAY, KI_RPM_SMU);

    return status;

/* USRLIB MODULE END  */
} 		/* End bridge_sot_pulse.c */
