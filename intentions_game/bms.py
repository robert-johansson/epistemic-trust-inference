"""Model comparison on the memo-port Laplace fits, compared with the authors' HBI.

Reads results/laplace_g1.npz (BPD) and results/laplace_g2.npz (controls) from fit.py.
Reports, per group:
  - random-effects Bayesian model selection (Stephan et al., 2009): expected model
    frequencies and exceedance probabilities, from the Laplace log evidences;
  - the number of participants best fit by each model;
  - the authors' HBI model frequencies and exceedance probabilities (Piray et al., 2019);
  - agreement between our MAP parameters and the authors' HBI parameters for each group's
    winning model (Spearman rho per parameter; HBI estimates are shrunk to group means).

Usage: python -I bms.py
"""
import sys
from pathlib import Path

import numpy as onp
from scipy.special import digamma
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_hbi, FIT_ORDER  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
PARAM_NAMES = {
    "M1": ["alpha", "beta", "alpha_v", "beta_v", "alpha_v_ref", "beta_v_ref"],
    "M2": ["alpha", "beta", "alpha_v", "beta_v", "alpha_v_ref", "beta_v_ref"],
    "M3": ["alpha", "beta", "alpha_v", "beta_v", "alpha_par", "beta_par", "alpha_v_ref", "beta_v_ref"],
    "M4": ["alpha", "beta", "alpha_v", "beta_v", "alpha_par", "beta_par", "alpha_v_ref", "beta_v_ref"],
    "Beta": ["beta", "beta_v", "beta_v_ref"],
}


def rfx_bms(log_ev, n_samples=1_000_000, seed=0):
    """Variational random-effects BMS (Stephan et al., 2009). Returns (expected freq, xp, alpha)."""
    n, k = log_ev.shape
    alpha0 = onp.ones(k)
    alpha = alpha0.copy()
    for _ in range(10_000):
        lu = log_ev + digamma(alpha) - digamma(alpha.sum())
        lu -= lu.max(axis=1, keepdims=True)
        g = onp.exp(lu)
        g /= g.sum(axis=1, keepdims=True)
        new = alpha0 + g.sum(axis=0)
        if onp.max(onp.abs(new - alpha)) < 1e-12:
            alpha = new
            break
        alpha = new
    r = onp.random.default_rng(seed).dirichlet(alpha, size=n_samples)
    xp = onp.bincount(r.argmax(axis=1), minlength=k) / n_samples
    return alpha / alpha.sum(), xp, alpha


if __name__ == "__main__":
    fmt = lambda v: "  ".join(f"{m}={x:.3f}" for m, x in zip(FIT_ORDER, v))
    for g, label in (("g1", "BPD"), ("g2", "Controls")):
        z = onp.load(RES / f"laplace_{g}.npz")
        ev, ok = z["log_ev"], z["ok"]
        n = int(z["done"])
        assert n == ev.shape[0], f"{g}: fit incomplete ({n}/{ev.shape[0]})"
        hbi = load_hbi(g)
        freq, xp, _ = rfx_bms(ev)
        best = onp.bincount(ev.argmax(axis=1), minlength=len(FIT_ORDER))
        print(f"\n=== {label} (n = {n}); converged with positive-definite Hessian: {ok.mean():.3f}")
        print("  Laplace + RFX-BMS  expected frequency:", fmt(freq))
        print("                     exceedance prob.:   ", fmt(xp))
        print("  best model per participant (counts):   ", "  ".join(f"{m}={c}" for m, c in zip(FIT_ORDER, best)))
        print("  authors' HBI       model frequency:    ", fmt(hbi["model_frequency"]))
        print("                     exceedance prob.:   ", fmt(hbi["exceedance_prob"]))
        print("  fixed-effects summed log evidence:     ", fmt(ev.sum(axis=0)))
        win = FIT_ORDER[int(onp.argmax(hbi["exceedance_prob"]))]
        ours = z[f"params_{win}"]
        theirs = hbi["parameters"][win]
        rhos = [spearmanr(ours[:, j], theirs[:, j])[0] for j in range(ours.shape[1])]
        print(f"  MAP vs HBI parameters for {win} (Spearman rho):",
              "  ".join(f"{p}={r:.2f}" for p, r in zip(PARAM_NAMES[win], rhos)))


def family_bms(log_ev, families, n_samples=1_000_000, seed=0):
    """Family-level RFX-BMS (Penny et al., 2010): uniform prior over models within each family,
    family evidence = log mean of member evidences. Returns {family: (expected freq, xp)}."""
    names = list(families)
    fam_ev = onp.stack([
        onp.logaddexp.reduce(log_ev[:, [FIT_ORDER.index(m) for m in families[f]]], axis=1)
        - onp.log(len(families[f])) for f in names], axis=1)
    freq, xp, _ = rfx_bms(fam_ev, n_samples, seed)
    return dict(zip(names, zip(freq, xp)))


FAMILIES = {
    "contagion (personally relevant)": {"yes": ["M1", "M3", "Beta"], "no": ["M2", "M4"]},
    "self-insertion (generalizable)": {"yes": ["M1", "M2", "Beta"], "no": ["M3", "M4"]},
}

if __name__ == "__main__":
    print("\n=== Family-level comparison (Laplace + RFX-BMS)")
    for g, label in (("g1", "BPD"), ("g2", "Controls")):
        ev = onp.load(RES / f"laplace_{g}.npz")["log_ev"]
        for fname, fam in FAMILIES.items():
            res = family_bms(ev, fam)
            print(f"  {label:8s} {fname:32s} " + "  ".join(
                f"{k}: freq={v[0]:.2f} xp={v[1]:.3f}" for k, v in res.items()))


def protected_bms(log_ev, n_samples=1_000_000, seed=0):
    """Random-effects BMS with protected exceedance probabilities (Rigoux et al., 2014), following
    SPM's spm_BMS: BOR = 1 / (1 + exp(F1 - F0)), where F1 is the variational free energy of the RFX
    model and F0 the log evidence of the null (all models equally frequent); PXP = (1 - BOR) * XP + BOR / K.
    Returns (expected frequency, xp, pxp, bor)."""
    n, k = log_ev.shape
    freq, xp, alpha = rfx_bms(log_ev, n_samples, seed)
    alpha0 = onp.ones(k)
    elogr = digamma(alpha) - digamma(alpha.sum())
    lu = log_ev + elogr
    lu = lu - lu.max(axis=1, keepdims=True)
    g = onp.exp(lu)
    g /= g.sum(axis=1, keepdims=True)
    from scipy.special import gammaln
    sqf = gammaln(alpha).sum() - gammaln(alpha.sum()) - ((alpha - 1) * elogr).sum()     # entropy of q(r)
    sqm = -(g * onp.log(g + 1e-300)).sum()                                               # entropy of q(m)
    elj = (gammaln(alpha0.sum()) - gammaln(alpha0).sum() + ((alpha0 - 1) * elogr).sum()
           + (g * (elogr[None, :] + log_ev)).sum())
    f1 = elj + sqf + sqm
    f0 = (onp.logaddexp.reduce(log_ev, axis=1) - onp.log(k)).sum()
    bor = 1.0 / (1.0 + onp.exp(f1 - f0))
    pxp = (1 - bor) * xp + bor / k
    return freq, xp, pxp, bor
