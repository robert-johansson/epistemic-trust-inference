"""Family-level tests and contagion contrasts reported in Section 4 of the manuscript.

Family evidence = log mean of member evidences (uniform prior over models within a family, as in
Penny et al., 2010); random-effects BMS with protected exceedance probabilities (Rigoux et al., 2014)
on the family evidences. Contagion contrasts compare a model with contagion against the same partner
representation without it.

Usage: python -I family_report.py
"""
import sys
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bms import protected_bms  # noqa: E402
from pxp_report import load, family  # noqa: E402

FAMILIES = {
    "representation: intent {I,C} vs competitiveness {Beta,M1-M4,H}":
        (["intent", "competitiveness"], [["I", "C"], ["Beta", "M1", "M2", "M3", "M4", "H"]]),
    "coupling within competitiveness: {M1,H} vs M4":
        (["coupled", "uncoupled"], [["M1", "H"], ["M4"]]),
    "coupling within intent: C vs I":
        (["coupled", "uncoupled"], [["C"], ["I"]]),
}

CONTAGION = [("M3 vs M4 (competitiveness, no insertion)", "M3", "M4"),
             ("M1 vs M2 (competitiveness, insertion)", "M1", "M2"),
             ("C vs I (intent)", "C", "I")]

if __name__ == "__main__":
    for g, label in (("g1", "BPD"), ("g2", "Controls")):
        ev = load(g)
        n = len(ev["M1"])
        print(f"\n=== {label} (n = {n})")
        for name, (labels, members) in FAMILIES.items():
            freq, xp, pxp, bor = protected_bms(onp.column_stack([family(ev, m) for m in members]))
            print(f"  {name:66s} " + "  ".join(
                f"{lab}: freq {freq[i]:.2f} xp {xp[i]:.3f} pxp {pxp[i]:.3f}" for i, lab in enumerate(labels))
                + f"  BOR {bor:.3f}")
        print("  contagion contrasts (values for the model WITH contagion):")
        for name, yes, no in CONTAGION:
            freq, xp, pxp, bor = protected_bms(onp.column_stack([ev[yes], ev[no]]))
            lbf = ev[yes] - ev[no]
            print(f"    {name:44s} freq {freq[0]:.2f}  pxp {pxp[0]:.3f}  BOR {bor:.3f}  "
                  f"sum log BF {lbf.sum():+.1f}  median log BF {onp.median(lbf):+.2f}  "
                  f"share log BF > 0 {onp.mean(lbf > 0):.0%}")


# ---------------- between-group comparison (Rigoux et al., 2014) ----------------
# Do BPD and controls have different model frequencies? Compare the evidence for separate random-effects
# analyses per group (frequencies may differ) with that for one pooled analysis (same frequencies):
# P(differ) = 1 / (1 + exp(logZ_pooled - logZ_separate)), equal prior odds. This is the formula of the VBA
# toolbox's VBA_groupBMC_btwGroups (p_same = 1 / (1 + exp(Fd - Fe))). VBA uses variational free energies;
# because every comparison here has two models or families, the evidence is also computed EXACTLY, as a
# 1-D integral over the frequency r under the flat Dirichlet(1,1) prior. The variational bound loses slack in
# each separate analysis, which biases P(differ) downward; the paper reports the exact values.
def free_energy(log_ev):
    from scipy.special import digamma, gammaln
    from bms import rfx_bms
    n, k = log_ev.shape
    _, _, alpha = rfx_bms(log_ev)
    a0 = onp.ones(k)
    el = digamma(alpha) - digamma(alpha.sum())
    lu = log_ev + el
    lu = lu - lu.max(axis=1, keepdims=True)
    g = onp.exp(lu)
    g /= g.sum(axis=1, keepdims=True)
    sqf = gammaln(alpha).sum() - gammaln(alpha.sum()) - ((alpha - 1) * el).sum()
    sqm = -(g * onp.log(g + 1e-300)).sum()
    elj = gammaln(a0.sum()) - gammaln(a0).sum() + ((a0 - 1) * el).sum() + (g * (el[None, :] + log_ev)).sum()
    return elj + sqf + sqm


def exact_log_evidence(log_ev, n_grid=400000):
    """log integral_0^1 prod_n [r exp(L_n1) + (1 - r) exp(L_n2)] dr (flat prior on r), midpoint rule."""
    from scipy.special import logsumexp
    assert log_ev.shape[1] == 2
    r = (onp.arange(n_grid) + 0.5) / n_grid
    a = onp.log(r)[:, None] + log_ev[None, :, 0]
    b = onp.log1p(-r)[:, None] + log_ev[None, :, 1]
    return logsumexp(onp.logaddexp(a, b).sum(axis=1)) - onp.log(n_grid)


BETWEEN = {
    "contagion: M3 vs M4": lambda e: onp.column_stack([e["M3"], e["M4"]]),
    "contagion: M1 vs M2": lambda e: onp.column_stack([e["M1"], e["M2"]]),
    "contagion: C vs I": lambda e: onp.column_stack([e["C"], e["I"]]),
    "contagion family {M1,M3} vs {M2,M4}": lambda e: onp.column_stack([family(e, ["M1", "M3"]), family(e, ["M2", "M4"])]),
    "representation {I,C} vs {Beta,M1-M4,H}": lambda e: onp.column_stack([family(e, ["I", "C"]), family(e, ["Beta", "M1", "M2", "M3", "M4", "H"])]),
    "coupling {M1,H} vs M4": lambda e: onp.column_stack([family(e, ["M1", "H"]), family(e, ["M4"])]),
    "insertion {M1,M2,Beta} vs {M3,M4}": lambda e: onp.column_stack([family(e, ["M1", "M2", "Beta"]), family(e, ["M3", "M4"])]),
}

if __name__ == "__main__":
    e1, e2 = load("g1"), load("g2")
    print("\n=== Insertion family including Beta (the beta-only version of M1): {M1,M2,Beta} vs {M3,M4}")
    for label, e in (("BPD", e1), ("Controls", e2)):
        freq, xp, pxp, bor = protected_bms(onp.column_stack([family(e, ["M1", "M2", "Beta"]), family(e, ["M3", "M4"])]))
        print(f"  {label:8s} insertion freq {freq[0]:.4f} pxp {pxp[0]:.4f} | no insertion freq {freq[1]:.4f} pxp {pxp[1]:.4f} | BOR {bor:.4f}")
    print("\n=== Between-group: posterior probability that BPD and controls differ in frequencies (Rigoux et al., 2014)")
    print(f"  {'comparison':42s} {'exact':>8s} {'variational':>12s}")
    for name, fn in BETWEEN.items():
        a, b = fn(e1), fn(e2)
        exact = 1.0 / (1.0 + onp.exp(exact_log_evidence(onp.vstack([a, b])) - (exact_log_evidence(a) + exact_log_evidence(b))))
        vb = 1.0 / (1.0 + onp.exp(free_energy(onp.vstack([a, b])) - (free_energy(a) + free_energy(b))))
        print(f"  {name:42s} {exact:8.4f} {vb:12.4f}")
