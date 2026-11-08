"""C library (float) against the NumPy reference (float64) on simulated shots. Needs build/libtrgctl.so."""
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "ref"), str(ROOT / "tools")]

from shooter_sim import make_set  # noqa: E402
from trgctl_ref import Cfg, analyze  # noqa: E402

trgctl_c = pytest.importorskip("trgctl_c")
if not trgctl_c.LIB_PATH.exists():
    pytest.skip("build/libtrgctl.so missing: run scripts/build.sh", allow_module_level=True)


@pytest.fixture(scope="module")
def lib():
    return trgctl_c.TrgctlC()


def test_defaults_match(lib):
    c = lib.default_cfg().as_dict()
    r = asdict(Cfg())
    assert c.keys() == r.keys()
    for k in r:
        assert c[k] == pytest.approx(r[k], rel=1e-6), k


def test_c_matches_reference(lib):
    shots = make_set(100, seed=7)  # 400 shots, a seed used nowhere else
    cfg_c, cfg_r = lib.default_cfg(), Cfg()
    disagree = 0
    for s in shots:
        c = lib.analyze(cfg_c, s)
        r = analyze(cfg_r, s["t_us"], s["gyro"].astype(float), s["acc"].astype(float), s["t_break_us"], s["trig"])
        assert c is not None and r is not None
        for k in ("dip_mrad", "left_mrad", "disp_mrad", "sway_rms_mrad", "push_g"):
            assert c[k] == pytest.approx(r[k], abs=0.02), k
        assert c["squeeze_ms"] == r["squeeze_ms"]
        assert c["peak_rate_pm_per_ms"] == pytest.approx(r["peak_rate_pm_per_ms"], abs=1e-5)
        assert c["imu_samples"] == r["imu_samples"]
        disagree += c["label"] != r["label"] or abs(c["onset_ms"] - r["onset_ms"]) > 0.01
    assert disagree <= 2  # float vs float64 can flip a shot that sits exactly on a threshold


def test_thin_window_rejected(lib):
    s = make_set(1, seed=3)[0]
    s = dict(s, t_us=s["t_us"][-20:], gyro=s["gyro"][-20:], acc=s["acc"][-20:])
    assert lib.analyze(lib.default_cfg(), s) is None
    assert analyze(Cfg(), s["t_us"], s["gyro"].astype(float), s["acc"].astype(float), s["t_break_us"],
                   s["trig"]) is None


def test_simulator_truth_is_consistent():
    for s in make_set(20, seed=5):
        t = s["truth"]
        assert len(s["trig"]) == 1000 and s["trig"][-1] == 800
        assert np.all(np.diff(s["t_us"].astype(np.int64)) > 0)
        if t["kind"] in ("GOOD", "UNSTEADY"):
            assert t["fault_dip_mrad"] == 0.0 and t["fault_left_mrad"] == 0.0
        else:
            assert 0 < np.hypot(t["fault_dip_mrad"], t["fault_left_mrad"]) <= t["magnitude_mrad"] + 1e-9
