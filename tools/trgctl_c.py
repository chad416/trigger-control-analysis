"""ctypes wrapper around build/libtrgctl.so (scripts/build.sh), so Python can run the C code on simulated shots."""
from __future__ import annotations

import ctypes as C
from pathlib import Path

import numpy as np

LIB_PATH = Path(__file__).resolve().parents[1] / "build" / "libtrgctl.so"
LABELS = ["GOOD", "UNSTEADY", "JERK", "FLINCH", "DISTURBED"]


class ImuSample(C.Structure):
    _fields_ = [("t_us", C.c_uint32)] + [(n, C.c_float) for n in ("gx", "gy", "gz", "ax", "ay", "az")]


class Cfg(C.Structure):
    _fields_ = [
        ("disp_window_us", C.c_uint32), ("ref_from_us", C.c_uint32), ("ref_to_us", C.c_uint32),
        ("hold_from_us", C.c_uint32), ("hold_to_us", C.c_uint32),
        ("disp_thr_mrad", C.c_float), ("disp_sway_k", C.c_float), ("sway_thr_mrad", C.c_float),
        ("onset_k", C.c_float), ("onset_floor_dps", C.c_float), ("jerk_rate_pm_per_ms", C.c_float),
        ("jerk_onset_us", C.c_uint32), ("flinch_dip_share", C.c_float), ("wall_pm", C.c_uint16),
    ]

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name, _ in self._fields_}


class Result(C.Structure):
    _fields_ = [("label", C.c_int)] + [(n, C.c_float) for n in (
        "dip_mrad", "left_mrad", "disp_mrad", "onset_ms", "sway_rms_mrad", "push_g", "squeeze_ms",
        "peak_rate_pm_per_ms")] + [("imu_samples", C.c_uint16)]


class TrgctlC:
    def __init__(self, path: Path = LIB_PATH):
        self.lib = C.CDLL(str(path))
        self.lib.trgctl_analyze.restype = C.c_bool
        self.lib.trgctl_analyze.argtypes = [C.POINTER(Cfg), C.POINTER(ImuSample), C.c_uint16, C.c_uint32,
                                            C.POINTER(C.c_uint16), C.c_uint16, C.POINTER(Result)]

    def default_cfg(self) -> Cfg:
        cfg = Cfg()
        self.lib.trgctl_default_cfg(C.byref(cfg))
        return cfg

    def analyze(self, cfg: Cfg, shot: dict) -> dict | None:
        n = len(shot["t_us"])
        imu = (ImuSample * n)()
        for i in range(n):
            g, a = shot["gyro"][i], shot["acc"][i]
            imu[i] = ImuSample(int(shot["t_us"][i]), *(float(v) for v in g), *(float(v) for v in a))
        trig = np.ascontiguousarray(shot["trig"], dtype=np.uint16)
        res = Result()
        ok = self.lib.trgctl_analyze(C.byref(cfg), imu, n, shot["t_break_us"],
                                     trig.ctypes.data_as(C.POINTER(C.c_uint16)), len(trig), C.byref(res))
        if not ok:
            return None
        out = {name: getattr(res, name) for name, _ in Result._fields_}
        out["label"] = LABELS[res.label]
        return out
