"""
build_run_metadata()'s measure_rxx flag and run_measurement()'s demod2_label
-- the R_xx toggle (mfli_dual_harmonic_6221.py) must (a) record which mode
produced a run's follower columns, and (b) label those columns rxx_1f_*
instead of 2f_* when the caller passes a non-"2f" demod2_label.

Hardware-free -- a fake DAQ that only answers the two getDouble() reads
build_run_metadata() actually makes (oscillator frequency, demod phaseshift).
"""

from __future__ import annotations

from instruments.keithley6221 import ACSourceConfig
from mfli.mfli_dual_harmonic import DemodConfig
from mfli.mfli_dual_harmonic_6221 import ExtRefConfig, build_run_metadata


class _DAQ:
    def getDouble(self, path):
        if path.endswith("/freq"):
            return 317.3
        if path.endswith("/phaseshift"):
            return 0.0
        raise AssertionError(f"unexpected getDouble({path!r})")


def _cfgs():
    leader_extref = ExtRefConfig(device="dev1", osc_index=0)
    demod1 = DemodConfig(device="dev1", demod_index=0, harmonic=1)
    demod2 = DemodConfig(device="dev2", demod_index=0, harmonic=2)
    return leader_extref, demod1, demod2


def test_build_run_metadata_default_measure_rxx_false():
    leader_extref, demod1, demod2 = _cfgs()
    meta = build_run_metadata(_DAQ(), ACSourceConfig(), leader_extref, demod1, demod2)
    assert meta["measure_rxx"] is False


def test_build_run_metadata_records_measure_rxx_true():
    leader_extref, demod1, demod2 = _cfgs()
    meta = build_run_metadata(_DAQ(), ACSourceConfig(), leader_extref, demod1, demod2,
                               measure_rxx=True)
    assert meta["measure_rxx"] is True
