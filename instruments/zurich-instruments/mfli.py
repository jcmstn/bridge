from dataclasses import dataclass
from typing import Optional, Dict, Any

from pymeasure.instruments import Instrument
from zhinst.toolkit import Session


@dataclass
class MFLIConfig:
    server_host: str = "localhost"
    master_serial: str = "dev0000"
    slave_serial: str = "dev0001"
    excitation_freq: float = 1_000.0
    excitation_amp: float = 0.01
    input_range: float = 1.0
    output_range: float = 1.0
    demod_rate: float = 10_000.0
    time_constant: float = 0.01
    settling_cycles: int = 10
    harmonic_1: int = 1
    harmonic_2: int = 2


class DualMFLIHarmonicLockin(Instrument):
    """
    PyMeasure wrapper for two Zurich Instruments MFLIs on one LabOne server.

    Master device provides the AC current excitation.
    Master and slave each demodulate one harmonic of the same drive frequency.
    """

    def __init__(
        self,
        server_host: str,
        master_serial: str,
        slave_serial: str,
        name: str = "DualMFLIHarmonicLockin",
        **kwargs,
    ):
        super().__init__(adapter=None, name=name, includeSCPI=False, **kwargs)
        self.cfg = MFLIConfig(
            server_host=server_host,
            master_serial=master_serial,
            slave_serial=slave_serial,
        )
        self.session: Optional[Session] = None
        self.master = None
        self.slave = None

    def connect(self):
        self.session = Session(self.cfg.server_host)
        self.master = self.session.connect_device(self.cfg.master_serial)
        self.slave = self.session.connect_device(self.cfg.slave_serial)
        return self

    def disconnect(self):
        self.shutdown()
        self.session = None
        self.master = None
        self.slave = None

    def setup(
        self,
        excitation_freq: Optional[float] = None,
        excitation_amp: Optional[float] = None,
        input_range: Optional[float] = None,
        output_range: Optional[float] = None,
        demod_rate: Optional[float] = None,
        time_constant: Optional[float] = None,
    ):
        if self.session is None:
            self.connect()

        f = excitation_freq if excitation_freq is not None else self.cfg.excitation_freq
        a = excitation_amp if excitation_amp is not None else self.cfg.excitation_amp
        ir = input_range if input_range is not None else self.cfg.input_range
        orng = output_range if output_range is not None else self.cfg.output_range
        dr = demod_rate if demod_rate is not None else self.cfg.demod_rate
        tc = time_constant if time_constant is not None else self.cfg.time_constant

        self.cfg.excitation_freq = f
        self.cfg.excitation_amp = a
        self.cfg.input_range = ir
        self.cfg.output_range = orng
        self.cfg.demod_rate = dr
        self.cfg.time_constant = tc

        self._configure_master_source()
        self._configure_demodulators()
        self._arm_devices()

        return self

    def _configure_master_source(self):
        d = self.master
        d.sigins[0].range(self.cfg.input_range)
        d.sigouts[0].range(self.cfg.output_range)

        d.oscs[0].freq(self.cfg.excitation_freq)
        d.sigouts[0].on(1)
        d.sigouts[0].enables[0](1)

        d.demods[0].enable(1)
        d.demods[0].adcselect(0)
        d.demods[0].order(4)
        d.demods[0].rate(self.cfg.demod_rate)
        d.demods[0].timeconstant(self.cfg.time_constant)
        d.demods[0].oscselect(0)
        d.demods[0].harmonic(self.cfg.harmonic_1)

    def _configure_demodulators(self):
        m = self.master
        s = self.slave

        m.demods[0].enable(1)
        m.demods[0].harmonic(self.cfg.harmonic_1)
        m.demods[0].oscselect(0)

        s.sigins[0].range(self.cfg.input_range)
        s.demods[0].enable(1)
        s.demods[0].adcselect(0)
        s.demods[0].order(4)
        s.demods[0].rate(self.cfg.demod_rate)
        s.demods[0].timeconstant(self.cfg.time_constant)
        s.demods[0].oscselect(0)
        s.demods[0].harmonic(self.cfg.harmonic_2)

    def _arm_devices(self):
        self.master.sync()
        self.slave.sync()

    def measure(self) -> Dict[str, Any]:
        if self.session is None:
            self.connect()

        m = self.master.demods[0].sample()
        s = self.slave.demods[0].sample()

        m_x = float(m["x"])
        m_y = float(m["y"])
        s_x = float(s["x"])
        s_y = float(s["y"])

        return {
            "master": {
                "x": m_x,
                "y": m_y,
                "r": (m_x**2 + m_y**2) ** 0.5,
                "phi": m["phase"] if "phase" in m else None,
            },
            "slave": {
                "x": s_x,
                "y": s_y,
                "r": (s_x**2 + s_y**2) ** 0.5,
                "phi": s["phase"] if "phase" in s else None,
            },
        }

    def shutdown(self):
        try:
            if self.master is not None:
                self.master.sigouts[0].on(0)
                self.master.demods[0].enable(0)
        except Exception:
            pass

        try:
            if self.slave is not None:
                self.slave.demods[0].enable(0)
        except Exception:
            pass

    @property
    def excitation_freq(self):
        return self.cfg.excitation_freq

    @excitation_freq.setter
    def excitation_freq(self, value):
        self.cfg.excitation_freq = float(value)
        if self.master is not None:
            self.master.oscs[0].freq(self.cfg.excitation_freq)

    @property
    def excitation_amp(self):
        return self.cfg.excitation_amp

    @excitation_amp.setter
    def excitation_amp(self, value):
        self.cfg.excitation_amp = float(value)
