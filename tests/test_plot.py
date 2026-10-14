import pandas as pd
import pytest

from plot import derive, pick_x, y_columns


def test_lockin_and_dc_resistance_derived():
    df = pd.DataFrame({
        "magnet_field_mT": [0.0, 10.0, 20.0],
        "1f_X_V": [2e-3, 4e-3, 6e-3], "1f_Y_V": [0.0, 0.0, 1e-6], "1f_R_V": [2e-3, 4e-3, 6e-3],
        "1f_R_sem_V": [1e-6] * 3,
        "excitation_current_A_rms": [1e-3] * 3,
        "voltage_even_V": [1e-4, 2e-4, 3e-4], "sense_current_A": [1e-4] * 3,
    })
    out, hidden = derive(df)
    assert out["1f_R_ohm"].tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert out["1f_R_sem_ohm"].tolist() == pytest.approx([1e-3] * 3)
    assert out["resistance_even_ohm"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert {"1f_R_V", "voltage_even_V"} <= hidden  # constant current: Ω replaces V
    ys = y_columns(out, "magnet_field_mT", hidden)
    assert "1f_R_ohm" in ys and "1f_R_V" not in ys and "1f_R_sem_ohm" not in ys


def test_iv_keeps_voltage_and_adds_dvdi():
    df = pd.DataFrame({"current_A": [-1e-3, 0.0, 1e-3], "voltage_V": [-2e-3, 0.0, 2e-3],
                       "resistance_ohm": [2.0, float("nan"), 2.0]})
    out, hidden = derive(df)
    assert "voltage_V" not in hidden
    assert out["dVdI_ohm"].tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert pick_x(out, "IV") == "current_A"


def test_pick_x_falls_back_when_field_constant():
    df = pd.DataFrame({"point_index": [1, 2, 3], "magnet_field_mT": [5.0] * 3,
                       "magnet_current_A": [0.1] * 3})
    assert pick_x(df, "BSWP") == "point_index"
    df["magnet_field_mT"] = [0.0, 1.0, 2.0]
    assert pick_x(df, "BSWP") == "magnet_field_mT"


def test_find_raw_nested_user_folders(tmp_path):
    from plot import find_raw
    for user in ("alice", "bob"):
        raw = tmp_path / user / "S1" / "raw"
        raw.mkdir(parents=True)
        (raw / "S1_0007_HB1_IV_20260101T000000.csv").write_text("x\n")
    with pytest.raises(SystemExit, match="alice/S1/raw"):
        find_raw(tmp_path, 7, None)
    with pytest.raises(SystemExit):
        find_raw(tmp_path, 7, "S1")          # name alone still ambiguous
    assert find_raw(tmp_path, 7, "bob/S1").parts[-4] == "bob"
