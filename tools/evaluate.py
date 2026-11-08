"""Tune the decision thresholds on one simulated set, then score the C library on another.

  python tools/evaluate.py --tune   grid search on the training set (seed 1), prints the best thresholds
  python tools/evaluate.py          runs the C library with its built-in defaults on the test set (seed 2),
                                    cross-checks it against the NumPy reference, writes results/ and plots

The training and test sets are generated with different seeds; the test set is never used for tuning.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "ref"), str(ROOT / "tools")]

from shooter_sim import KINDS, make_set  # noqa: E402
from trgctl_ref import Cfg, analyze, classify  # noqa: E402

TRAIN_SEED, TEST_SEED = 1, 2
N_TRAIN, N_TEST = 500, 1000
PRED = ["GOOD", "UNSTEADY", "JERK", "FLINCH", "DISTURBED"]
RESULTS = ROOT / "results"

# palette (validated categorical slots 1-4 and text inks)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def features(cfg: Cfg, shots: list[dict]) -> list[dict]:
    return [analyze(cfg, s["t_us"], s["gyro"].astype(float), s["acc"].astype(float), s["t_break_us"], s["trig"])
            for s in shots]


def balanced_accuracy(truth: list[str], pred: list[str]) -> float:
    return float(np.mean([np.mean([p == k for t, p in zip(truth, pred) if t == k]) for k in KINDS]))


def tune() -> Cfg:
    shots = make_set(N_TRAIN, TRAIN_SEED)
    base = Cfg()
    feats = features(base, shots)
    truth = [s["truth"]["kind"] for s in shots]
    grid = {
        "disp_thr_mrad": [1.0, 1.2, 1.5, 2.0],
        "disp_sway_k": [3.0, 4.0, 5.0, 6.0, 8.0],
        "sway_thr_mrad": [0.25, 0.3, 0.35],
        "jerk_rate_pm_per_ms": [1.5, 2.0, 2.5, 3.0],
        "jerk_onset_us": [15_000, 20_000, 25_000, 35_000],
        "flinch_dip_share": [0.2, 0.3, 0.4, 0.5, 0.6],
    }
    best = (-1.0, None)
    for values in itertools.product(*grid.values()):
        cfg = replace(base, **dict(zip(grid.keys(), values)))
        score = balanced_accuracy(truth, [classify(cfg, f) for f in feats])
        if score > best[0]:
            best = (score, cfg)
    print(f"training set: {len(shots)} shots, best balanced accuracy {best[0]:.4f}")
    for k in grid:
        print(f"  {k} = {getattr(best[1], k)}")
    return best[1]


def ref_cfg_from_c(c_cfg) -> Cfg:
    d = c_cfg.as_dict()
    return Cfg(**{k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()})


def evaluate() -> None:
    from trgctl_c import TrgctlC

    lib = TrgctlC()
    c_cfg = lib.default_cfg()
    r_cfg = ref_cfg_from_c(c_cfg)
    shots = make_set(N_TEST, TEST_SEED)
    truth = [s["truth"]["kind"] for s in shots]
    c_out = [lib.analyze(c_cfg, s) for s in shots]
    r_out = features(r_cfg, shots)
    assert all(o is not None for o in c_out) and all(o is not None for o in r_out)

    # cross-check C (float) against the reference (float64)
    keys = ["dip_mrad", "left_mrad", "disp_mrad", "sway_rms_mrad", "onset_ms", "squeeze_ms", "peak_rate_pm_per_ms"]
    max_diff = {k: float(max(abs(c[k] - r[k]) for c, r in zip(c_out, r_out))) for k in keys}
    label_agree = sum(c["label"] == r["label"] for c, r in zip(c_out, r_out))

    pred = [c["label"] for c in c_out]
    conf = {k: {p: sum(1 for t, q in zip(truth, pred) if t == k and q == p) for p in PRED} for k in KINDS}
    recall = {k: conf[k][k] / N_TEST for k in KINDS}
    precision = {k: conf[k][k] / max(1, sum(conf[t][k] for t in KINDS)) for k in KINDS}

    # movement estimate against the injected fault at the break
    est_err = {}
    for kind in ("JERK", "FLINCH"):
        idx = [i for i, s in enumerate(shots) if s["truth"]["kind"] == kind]
        e = [np.hypot(c_out[i]["dip_mrad"] - shots[i]["truth"]["fault_dip_mrad"],
                      c_out[i]["left_mrad"] - shots[i]["truth"]["fault_left_mrad"]) for i in idx]
        true_mag = [np.hypot(shots[i]["truth"]["fault_dip_mrad"], shots[i]["truth"]["fault_left_mrad"]) for i in idx]
        est_err[kind] = {"rms_mrad": float(np.sqrt(np.mean(np.square(e)))),
                         "p95_mrad": float(np.percentile(e, 95)),
                         "median_true_mrad": float(np.median(true_mag))}
    onset_err = [c_out[i]["onset_ms"] - shots[i]["truth"]["onset_ms"]
                 for i in range(len(shots)) if truth[i] == "FLINCH" and c_out[i]["onset_ms"] > 0]

    summary = {
        "test_set": {"seed": TEST_SEED, "shots_per_kind": N_TEST},
        "config": c_cfg.as_dict(),
        "balanced_accuracy": balanced_accuracy(truth, pred),
        "recall": recall,
        "precision": precision,
        "confusion": conf,
        "movement_error": est_err,
        "flinch_onset_error_ms": {"median": float(np.median(onset_err)),
                                  "p5": float(np.percentile(onset_err, 5)),
                                  "p95": float(np.percentile(onset_err, 95))},
        "c_vs_reference": {"labels_agree": label_agree, "shots": len(shots), "max_abs_diff": max_diff},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("balanced_accuracy", "recall", "precision", "movement_error",
                                              "flinch_onset_error_ms", "c_vs_reference")}, indent=2))
    plots(shots, c_out, conf)


def _style(ax) -> None:
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plots(shots: list[dict], out: list[dict], conf: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})

    # 1. one example shot per kind: muzzle motion and trigger travel around the break
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]},
                             facecolor=SURF)
    for col, kind in enumerate(KINDS):
        i = next(j for j, s in enumerate(shots) if s["truth"]["kind"] == kind and out[j]["label"] == kind
                 and (kind in ("GOOD", "UNSTEADY") or s["truth"]["magnitude_mrad"] > 5))
        s, o = shots[i], out[i]
        m = s["motion"]
        sel = (m["t"] >= -0.6) & (m["t"] <= 0.0)
        t_ms = m["t"][sel] * 1e3
        ax = axes[0, col]
        _style(ax)
        ax.axvspan(-150, 0, color=GRID, alpha=0.55, linewidth=0)
        ax.plot(t_ms, m["dip"][sel] - m["dip"][sel][0], color=BLUE, linewidth=2, label="dip (+ down)")
        ax.plot(t_ms, m["left"][sel] - m["left"][sel][0], color=ORANGE, linewidth=2, label="left (+ left)")
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(f"{kind}  ->  detector: {o['label']}", fontsize=10, color=INK, loc="left", pad=30)
        ax.text(0.0, 1.03, f"measured at break: dip {o['dip_mrad']:+.1f}, left {o['left_mrad']:+.1f} mrad\n"
                f"onset {o['onset_ms']:.0f} ms, hold RMS {o['sway_rms_mrad']:.2f} mrad",
                transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
        if col == 0:
            ax.set_ylabel("muzzle angle (mrad)")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax2 = axes[1, col]
        _style(ax2)
        tt = np.arange(-len(s["trig"]) + 1, 1)
        keep = tt >= -600
        ax2.plot(tt[keep], s["trig"][keep], color=AQUA, linewidth=2)
        ax2.axvline(0, color=INK2, linewidth=1)
        ax2.set_ylim(550, 820)
        ax2.text(0.02, 0.08, f"press {o['squeeze_ms']:.0f} ms, peak {o['peak_rate_pm_per_ms']:.1f} pm/ms",
                 transform=ax2.transAxes, fontsize=8, color=INK2)
        ax2.set_xlabel("time to break (ms)")
        if col == 0:
            ax2.set_ylabel("trigger travel (permille)")
    fig.suptitle("One simulated shot of each kind (shaded: the 150 ms window measured at the break)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "examples.png", dpi=130, facecolor=SURF)
    plt.close(fig)

    # 2. confusion matrix, row-normalised
    mat = np.array([[conf[k][p] for p in PRED] for k in KINDS], dtype=float)
    pct = mat / mat.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.6, 3.9), facecolor=SURF)
    ax.imshow(pct, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f2f6fc", BLUE]), vmin=0,
              vmax=100, aspect="auto")
    for r in range(len(KINDS)):
        for c in range(len(PRED)):
            ax.text(c, r, f"{pct[r, c]:.1f} %\n({int(mat[r, c])})", ha="center", va="center", fontsize=8,
                    color="#ffffff" if pct[r, c] > 55 else INK)
    ax.set_xticks(range(len(PRED)), PRED, fontsize=8, color=INK2)
    ax.set_yticks(range(len(KINDS)), KINDS, fontsize=8, color=INK2)
    ax.set_xlabel("detector label (C library)")
    ax.set_ylabel("simulated shot")
    ax.set_title(f"Test set: {N_TEST} shots of each kind, thresholds tuned on a separate set",
                 fontsize=10, color=INK, loc="left")
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion.png", dpi=130, facecolor=SURF)
    plt.close(fig)

    # 3. movement estimate vs truth, and recall against fault size
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    for ax in (a1, a2):
        _style(ax)
    for kind, colour in (("JERK", BLUE), ("FLINCH", ORANGE)):
        idx = [i for i, s in enumerate(shots) if s["truth"]["kind"] == kind]
        tm = [np.hypot(shots[i]["truth"]["fault_dip_mrad"], shots[i]["truth"]["fault_left_mrad"]) for i in idx]
        em = [out[i]["disp_mrad"] for i in idx]
        a1.scatter(tm, em, s=9, color=colour, alpha=0.5, linewidths=0, label=kind)
        mag = np.array([shots[i]["truth"]["magnitude_mrad"] for i in idx])
        hit = np.array([out[i]["label"] == kind for i in idx])
        edges = np.arange(2, 13, 1.0)
        centres, rec = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (mag >= lo) & (mag < hi)
            if sel.sum() >= 20:
                centres.append((lo + hi) / 2)
                rec.append(hit[sel].mean() * 100)
        a2.plot(centres, rec, color=colour, linewidth=2, marker="o", markersize=5, label=kind)
    lim = [0, 13]
    a1.plot(lim, lim, color=INK2, linewidth=1, linestyle="--")
    a1.set_xlim(lim)
    a1.set_ylim(lim)
    a1.set_xlabel("true movement at the break (mrad)")
    a1.set_ylabel("measured movement (mrad)")
    a1.set_title("Measured vs true muzzle movement at the break", fontsize=10, color=INK, loc="left")
    a1.legend(fontsize=8, frameon=False)
    a2.set_ylim(0, 102)
    a2.set_xlabel("full size of the fault (mrad)")
    a2.set_ylabel("labelled correctly (%)")
    a2.set_title("Recall against fault size", fontsize=10, color=INK, loc="left")
    a2.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "movement.png", dpi=130, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    if args.tune:
        tune()
    else:
        evaluate()
