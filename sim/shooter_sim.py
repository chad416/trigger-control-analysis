"""Synthetic shots for an instrumented training weapon: muzzle motion, IMU samples and trigger travel.

Every shot is the second before the trigger breaks (t = 0) plus a little after it. Four kinds:

  GOOD      ordinary hold sway, clean press (some presses are fast but still clean)
  UNSTEADY  large hold sway, clean press
  JERK      fast final press that drags the muzzle, mostly down and left (right-handed
            shooter), the movement starting 10-35 ms before the break
  FLINCH    muzzle pushed down in anticipation of recoil, starting 50-150 ms before the
            break, with a forward push on the accelerometer; the press itself is normal

All of this is a model, not recorded shooters: the sway, fault sizes and timings below are
assumptions chosen to be plausible, and the sensor noise is from the LSM6DSO datasheet
class (rate noise density 3.8 mdps/sqrt(Hz), zero-rate level within +-0.5 deg/s after
calibration, 8.75 mdps/LSB at +-250 deg/s).

Angles are in mrad: dip (+ = muzzle down), left (+ = muzzle left). Weapon frame as in the
trainer firmware: x forward, y left, z up; positive gy dips the muzzle, positive gz moves
it left.
"""
from __future__ import annotations

import numpy as np

KINDS = ["GOOD", "UNSTEADY", "JERK", "FLINCH"]
DEG_TO_MRAD = 17.45329252

FINE_HZ = 1000           # motion is built on a 1 ms grid
T0, T1 = -1.2, 0.1       # seconds around the break
IMU_HZ = 208.0
GYRO_NOISE_DPS = 3.8e-3 * np.sqrt(IMU_HZ / 2)   # white noise per sample
GYRO_BIAS_DPS = 0.5
GYRO_LSB_DPS = 8.75e-3
ACC_NOISE_G = 70e-6 * np.sqrt(IMU_HZ / 2)
ACC_LSB_G = 0.061e-3
WALL_PM, BREAK_PM = 610, 800


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sway(rng: np.random.Generator, t: np.ndarray, rms_2d: float) -> tuple[np.ndarray, np.ndarray]:
    """Low-frequency hold sway (0.2-1 Hz) plus a small 8-12 Hz tremor, per axis."""
    out = []
    for _ in range(2):
        f = rng.uniform(0.2, 1.0, 3)
        a = rng.uniform(0.5, 1.0, 3)
        ph = rng.uniform(0, 2 * np.pi, 3)
        s = sum(a[k] * np.sin(2 * np.pi * f[k] * t + ph[k]) for k in range(3))
        s *= (rms_2d / np.sqrt(2)) / np.sqrt(np.mean(s ** 2))
        s += rng.uniform(0.03, 0.15) * np.sin(2 * np.pi * rng.uniform(8, 12) * t + rng.uniform(0, 2 * np.pi))
        out.append(s)
    return out[0], out[1]


def _trigger(rng: np.random.Generator, press_s: float) -> np.ndarray:
    """Travel in permille at 1 kHz for t = -0.999 .. 0 s; the last sample is the break."""
    t = np.arange(-999, 1) / 1000.0
    take_start = rng.uniform(-0.95, -0.80)
    take_len = rng.uniform(0.10, 0.20)
    travel = WALL_PM * smoothstep((t - take_start) / take_len)
    wall = WALL_PM + 5.0 * smoothstep((t - take_start - take_len) / 0.2)
    travel = np.where(t > take_start + take_len, wall, travel)
    press = (t > -press_s)
    frac = np.clip((t + press_s) / press_s, 0, 1)
    travel = np.where(press, (WALL_PM + 5.0) + (BREAK_PM - WALL_PM - 5.0) * frac, travel)
    travel = travel + rng.integers(-2, 3, size=t.size)
    travel = np.clip(np.round(travel), 0, 1000).astype(np.uint16)
    travel[-1] = BREAK_PM
    return travel


def make_shot(rng: np.random.Generator, kind: str) -> dict:
    t = np.arange(int(T0 * FINE_HZ), int(T1 * FINE_HZ) + 1) / FINE_HZ
    sway_rms = rng.uniform(3.5, 7.0) if kind == "UNSTEADY" else rng.uniform(0.4, 2.0)
    dip_s, left_s = _sway(rng, t, sway_rms)
    dip_f = np.zeros_like(t)
    left_f = np.zeros_like(t)
    push = np.zeros_like(t)
    truth = {"kind": kind, "sway_rms_mrad": sway_rms, "magnitude_mrad": 0.0, "onset_ms": 0.0}

    if kind == "JERK":
        mag = rng.uniform(2.0, 10.0)
        ang = np.deg2rad(rng.normal(35.0, 20.0))        # from straight down towards left
        lead = rng.uniform(0.010, 0.035)                # movement starts this long before the break
        prof = smoothstep((t + lead) / (1.5 * lead))
        dip_f, left_f = mag * np.cos(ang) * prof, mag * np.sin(ang) * prof
        press_s = rng.uniform(0.008, 0.040)
        truth.update(magnitude_mrad=mag, onset_ms=lead * 1e3)
    elif kind == "FLINCH":
        mag = rng.uniform(2.0, 12.0)
        ang = np.deg2rad(rng.normal(0.0, 15.0))
        lead = rng.uniform(0.050, 0.150)
        rise = lead * rng.uniform(0.8, 1.3)
        prof = smoothstep((t + lead) / rise)
        dip_f, left_f = mag * np.cos(ang) * prof, mag * np.sin(ang) * prof
        push = rng.uniform(0.03, 0.15) * prof
        press_s = rng.uniform(0.12, 0.60)
        truth.update(magnitude_mrad=mag, onset_ms=lead * 1e3)
    else:
        press_s = rng.uniform(0.03, 0.08) if rng.random() < 0.2 else rng.uniform(0.12, 0.60)

    dip = dip_s + dip_f
    left = left_s + left_f
    i0 = int(round(-T0 * FINE_HZ))                       # index of t = 0
    hold = (t >= -0.6) & (t <= -0.2)
    truth["fault_dip_mrad"] = float(dip_f[i0])
    truth["fault_left_mrad"] = float(left_f[i0])
    truth["aim_err_dip_mrad"] = float(dip[i0] - dip[hold].mean())
    truth["aim_err_left_mrad"] = float(left[i0] - left[hold].mean())
    truth["press_ms"] = press_s * 1e3

    # IMU: rates from the motion, sampled at 208 Hz with a random phase, noise, bias, quantisation
    rate_dip = np.gradient(dip, 1.0 / FINE_HZ) / DEG_TO_MRAD      # deg/s
    rate_left = np.gradient(left, 1.0 / FINE_HZ) / DEG_TO_MRAD
    period = 1.0 / IMU_HZ
    t_imu = np.arange(-0.65 + rng.uniform(0, period), 0.0, period)
    t_imu = t_imu[t_imu <= 0.0]
    gy = np.interp(t_imu, t, rate_dip)
    gz = np.interp(t_imu, t, rate_left)
    gx = np.zeros_like(t_imu)
    gyro = np.stack([gx, gy, gz], axis=1)
    gyro += rng.uniform(-GYRO_BIAS_DPS, GYRO_BIAS_DPS, 3)
    gyro += rng.normal(0.0, GYRO_NOISE_DPS, gyro.shape)
    gyro = np.round(gyro / GYRO_LSB_DPS) * GYRO_LSB_DPS

    cant = np.deg2rad(rng.uniform(-3.0, 3.0))
    ax = np.interp(t_imu, t, push)
    acc = np.stack([ax, np.full_like(t_imu, np.sin(cant)), np.full_like(t_imu, np.cos(cant))], axis=1)
    acc += rng.normal(0.0, ACC_NOISE_G, acc.shape)
    acc = np.round(acc / ACC_LSB_G) * ACC_LSB_G

    t_break_us = 10_000_000                               # arbitrary clock
    t_us = (t_break_us + np.round(t_imu * 1e6)).astype(np.uint32)

    return {
        "truth": truth,
        "t_us": t_us,
        "gyro": gyro.astype(np.float32),
        "acc": acc.astype(np.float32),
        "t_break_us": t_break_us,
        "trig": _trigger(rng, press_s),
        "motion": {"t": t, "dip": dip, "left": left},
    }


def make_set(n_per_kind: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    shots = [make_shot(rng, k) for k in KINDS for _ in range(n_per_kind)]
    return shots
