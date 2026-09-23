"""Suite-wide isolation: no test ever writes the real <repo>/../data/runs.db."""

from __future__ import annotations

import pytest

from instruments import run_index


@pytest.fixture(autouse=True)
def _isolated_run_history(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(run_index, "_DB_PATH", tmp_path_factory.mktemp("runs") / "runs.db")
