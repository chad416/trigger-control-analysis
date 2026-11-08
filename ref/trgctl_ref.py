"""Reference implementation of trgctl in NumPy (float64), for cross-checking the C library.

Same windows, rules and order of operations as trgctl/trgctl.c; see that file for the
meaning of each step.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

DEG_TO_MRAD = 17.45329252
LABELS = ["GOOD", "UNSTEADY", "JERK", "FLINCH", "DISTURBED"]
PEAK_RATE_WINDOW = 50
PEAK_RATE_STEP = 5


@dataclass
class Cfg:
    disp_window_us: int = 150_000
    ref_from_us: int = 160_000
    ref_to_us: int = 320_000
    hold_from_us: int = 200_000
    hold_to_us: int = 600_000
    disp_thr_mrad: float = 1.5
    disp_sway_k: float = 6.0
    sway_thr_mrad: float = 0.3
    onset_k: float = 4.0
    onset_floor_dps: float = 0.3
    jerk_rate_pm_per_ms: float = 2.5
    jerk_onset_us: int = 25_000
    flinch_dip_share: float = 0.5
    wall_pm: int = 620

    def as_dict(self) -> dict:
        return asdict(self)


def analyze(cfg: Cfg, t_us: np.ndarray, gyro: np.ndarray, acc: np.ndarray, t_break_us: int,
            trig: np.ndarray) -> dict | None:
    """t_us: (n,) uint32; gyro, acc: (n, 3) deg/s and g; trig: (m,) permille at 1 kHz, last = break."""
    age = (np.int64(t_break_us) - t_us.astype(np.int64))
    gy, gz, ax = gyro[:, 1], gyro[:, 2], acc[:, 0]
    n = len(t_us)

    # rectangle rule: each sample holds until the next one (or the break)
    until = np.full(n, t_break_us, dtype=np.int64)
    nxt_ok = np.zeros(n, dtype=bool)
    nxt_ok[:-1] = age[1:] >= 0
    until[:-1] = np.where(nxt_ok[:-1], t_us[1:].astype(np.int64), t_break_us)
    dt = (until - t_us.astype(np.int64)) * 1e-6

    ref = (age >= cfg.ref_from_us) & (age <= cfg.ref_to_us)
    hold = (age >= cfg.hold_from_us) & (age <= cfg.hold_to_us)
    disp = (age >= 0) & (age <= cfg.disp_window_us)
    if ref.sum() < 3 or hold.sum() < 10 or disp.sum() < 3 or len(trig) < 2:
        return None

    ref_y, ref_z, ref_ax = gy[ref].mean(), gz[ref].mean(), ax[ref].mean()
    hold_y, hold_z = gy[hold].mean(), gz[hold].mean()

    # straight-line fit of the rate over the reference window, extrapolated to the break
    x = -age * 1e-6
    ref_x = x[ref].mean()
    dxr = x[ref] - ref_x
    sxx = np.sum(dxr ** 2)
    slope_y = np.sum(dxr * (gy[ref] - ref_y)) / sxx if sxx > 0 else 0.0
    slope_z = np.sum(dxr * (gz[ref] - ref_z)) / sxx if sxx > 0 else 0.0
    ry = gy - (ref_y + slope_y * (x - ref_x))
    rz = gz - (ref_z + slope_z * (x - ref_x))

    dip = np.sum(ry[disp] * dt[disp]) * DEG_TO_MRAD
    left = np.sum(rz[disp] * dt[disp]) * DEG_TO_MRAD
    push = ax[disp].mean() - ref_ax

    ang_y = np.cumsum((gy[hold] - hold_y) * dt[hold] * DEG_TO_MRAD)
    ang_z = np.cumsum((gz[hold] - hold_z) * dt[hold] * DEG_TO_MRAD)
    var = np.mean(ang_y ** 2 + ang_z ** 2) - ang_y.mean() ** 2 - ang_z.mean() ** 2
    sway = float(np.sqrt(var)) if var > 0 else 0.0

    res = np.sqrt(ry ** 2 + rz ** 2)
    thr = max(cfg.onset_k * np.sqrt(np.mean(res[ref] ** 2)), cfg.onset_floor_dps)
    out = disp & (res > thr)
    onset_age = int(age[out].max()) if out.sum() >= 2 else -1

    # trigger
    last = len(trig) - 1
    i = last
    while i > 0 and trig[i] >= cfg.wall_pm:
        i -= 1
    squeeze = float(len(trig)) if trig[i] >= cfg.wall_pm else float(last - i)
    peak = 0.0
    for k in range(PEAK_RATE_WINDOW):
        j = last - k
        if j - PEAK_RATE_STEP < 0:
            break
        peak = max(peak, (float(trig[j]) - float(trig[j - PEAK_RATE_STEP])) / PEAK_RATE_STEP)

    disp_mag = float(np.hypot(dip, left))
    feat = {
        "dip_mrad": float(dip),
        "left_mrad": float(left),
        "disp_mrad": disp_mag,
        "onset_ms": onset_age * 1e-3 if onset_age >= 0 else 0.0,
        "sway_rms_mrad": sway,
        "push_g": float(push),
        "squeeze_ms": squeeze,
        "peak_rate_pm_per_ms": peak,
        "imu_samples": int(hold.sum()),
        "onset_us": onset_age,
    }
    feat["label"] = classify(cfg, feat)
    return feat


def classify(cfg: Cfg, f: dict) -> str:
    """Decision rules on the features; separate so thresholds can be tuned without re-analysing."""
    if f["disp_mrad"] < max(cfg.disp_thr_mrad, cfg.disp_sway_k * f["sway_rms_mrad"]):
        return "UNSTEADY" if f["sway_rms_mrad"] > cfg.sway_thr_mrad else "GOOD"
    if f["peak_rate_pm_per_ms"] >= cfg.jerk_rate_pm_per_ms or (0 <= f["onset_us"] <= cfg.jerk_onset_us):
        return "JERK"
    if f["dip_mrad"] >= cfg.flinch_dip_share * f["disp_mrad"]:
        return "FLINCH"
    return "DISTURBED"
