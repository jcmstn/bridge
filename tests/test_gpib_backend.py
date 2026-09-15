"""
resolve_visa_resource()'s platform branch and GPIB-address parsing — pure
logic, no serial port or hardware ever opened (the Prologix controller
construction is monkeypatched out).
"""

from __future__ import annotations

import pytest

import instruments.gpib_backend as gb


def test_non_macos_returns_resource_unchanged(monkeypatch):
    monkeypatch.setattr(gb, "_IS_MACOS", False)
    assert gb.resolve_visa_resource("GPIB0::6::INSTR") == "GPIB0::6::INSTR"


def test_macos_non_gpib_resource_untouched(monkeypatch):
    monkeypatch.setattr(gb, "_IS_MACOS", True)
    resource = "TCPIP0::192.168.1.5::7020::SOCKET"
    assert gb.resolve_visa_resource(resource) == resource


def test_macos_gpib_resource_routes_through_prologix(monkeypatch):
    monkeypatch.setattr(gb, "_IS_MACOS", True)

    class FakeController:
        def gpib(self, address):
            return ("prologix-adapter-for", address)

    monkeypatch.setattr(gb, "_prologix_controller", lambda: FakeController())
    assert gb.resolve_visa_resource("GPIB0::6::INSTR") == ("prologix-adapter-for", 6)
    assert gb.resolve_visa_resource("GPIB1::20::INSTR") == ("prologix-adapter-for", 20)


def test_macos_missing_port_env_raises(monkeypatch):
    monkeypatch.setattr(gb, "_IS_MACOS", True)
    monkeypatch.setattr(gb, "_controller", None)
    monkeypatch.delenv("BRIDGE_PROLOGIX_PORT", raising=False)
    with pytest.raises(RuntimeError, match="BRIDGE_PROLOGIX_PORT"):
        gb.resolve_visa_resource("GPIB0::6::INSTR")
