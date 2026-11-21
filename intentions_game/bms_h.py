"""Add the hierarchical self-other model H to the model comparison.

Combines results/laplace_g*.npz (Beta, M2, M1, M3, M4) with results/laplace_g*_H.npz (H).
Reports, per group:
  - random-effects BMS over all six models (expected frequencies, exceedance probabilities);
  - the informative head-to-head, H vs. M1 (H is the exact version of M1): per-participant
    log Bayes factors (median, share > 0, share with |log BF| > 1) and a two-model RFX-BMS;
  - H vs. M3 and H vs. M4 for reference;
  - fitted coupling (population spread tau) by group.

Usage: python -I bms_h.py
"""
import sys
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bms import rfx_bms, RES  # noqa: E402
from data import FIT_ORDER  # noqa: E402

MODELS = FIT_ORDER + ["H"]


def load(group):
    z5 = onp.load(RES / f"laplace_{group}.npz")
    zh = onp.load(RES / f"laplace_{group}_H.npz")
    assert int(zh["done"]) == zh["log_ev"].shape[0], f"{group}: H fit incomplete"
    assert (z5["ids"] == zh["ids"]).all()
    ev = onp.concatenate([z5["log_ev"], zh["log_ev"]], axis=1)
    ok = onp.concatenate([z5["ok"], zh["ok"]], axis=1)
    return ev, ok, zh["params_H"]


def head_to_head(ev, a, b):
    d = ev[:, MODELS.index(a)] - ev[:, MODELS.index(b)]
    freq, xp, _ = rfx_bms(ev[:, [MODELS.index(a), MODELS.index(b)]])
    return (f"{a} vs {b}: median log BF = {onp.median(d):+.2f}; {a} better for {(d > 0).mean():.0%} "
            f"(|log BF| > 1: {(d > 1).mean():.0%} for {a}, {(d < -1).mean():.0%} for {b}); "
            f"RFX freq {a} = {freq[0]:.2f}, xp = {xp[0]:.3f}; summed log BF = {d.sum():+.1f}")


if __name__ == "__main__":
    fmt = lambda v: "  ".join(f"{m}={x:.3f}" for m, x in zip(MODELS, v))
    taus = {}
    for g, label in (("g1", "BPD"), ("g2", "Controls")):
        ev, ok, ph = load(g)
        freq, xp, _ = rfx_bms(ev)
        print(f"\n=== {label} (n = {ev.shape[0]}); H fits ok: {ok[:, -1].mean():.3f}")
        print("  6-model RFX-BMS  frequency:", fmt(freq))
        print("                   xp:       ", fmt(xp))
        print("  best per participant:      ", "  ".join(
            f"{m}={c}" for m, c in zip(MODELS, onp.bincount(ev.argmax(axis=1), minlength=len(MODELS)))))
        for b in ("M1", "M3", "M4"):
            print("  " + head_to_head(ev, "H", b))
        taus[label] = onp.exp(ph[:, 4:6])
        print(f"  fitted tau (median [IQR]): alpha {onp.median(taus[label][:, 0]):.2f} "
              f"[{onp.percentile(taus[label][:, 0], 25):.2f}, {onp.percentile(taus[label][:, 0], 75):.2f}]; "
              f"beta {onp.median(taus[label][:, 1]):.2f} "
              f"[{onp.percentile(taus[label][:, 1], 25):.2f}, {onp.percentile(taus[label][:, 1], 75):.2f}]")
    from scipy.stats import mannwhitneyu
    for j, dim in enumerate(("alpha", "beta")):
        u = mannwhitneyu(taus["BPD"][:, j], taus["Controls"][:, j])
        print(f"\nBPD vs controls, log tau_{dim}: Mann-Whitney U = {u.statistic:.0f}, p = {u.pvalue:.3f}")
