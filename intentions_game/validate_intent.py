"""Check the memo intent model (intent.py, model "I") against an independent NumPy implementation
of the same equations: partner P(option 1 | omega, rho) = logistic(omega*(s1-s2) + rho*(o1-o2)),
sequential normalised Bayesian updating over the (omega, rho) grid, probability-matching
predictions, and the two attributions P(rho < 0) and E[omega] after phase 2.

Usage: python -I validate_intent.py
"""
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, matlab_order  # noqa: E402
import memo_models as mm  # noqa: E402
import intent  # noqa: E402

EPS = onp.finfo(float).eps
OM, RH = onp.asarray(mm.AG), onp.asarray(mm.BG)


def numpy_phase2(params, data):
    q = mm.unpack("I", np.asarray(params))
    q = {k: float(v) for k, v in q.items()}
    prior = norm.pdf(OM, q["alpha_par"], q["alpha_v_ref"])[:, None] * norm.pdf(RH, q["beta_par"], q["beta_v_ref"])[None, :] + EPS
    b = prior / prior.sum()
    p2 = onp.zeros(54)
    for k, t in enumerate(range(36, 90)):
        s1, o1, s2, o2 = data[t, 2:6] / 10.0
        p1 = 1 / (1 + onp.exp(-(OM[:, None] * (s1 - s2) + RH[None, :] * (o1 - o2))))
        p2[k] = (b * p1).sum()
        lik = p1 if data[t, 7] == 1 else 1 - p1
        b = b * lik
        b = b / b.sum()
    harm = (b.sum(axis=0) * onp.maximum(-RH, 0)).sum()
    e_omega = (b.sum(axis=1) * OM).sum()
    p_neg = b[:, RH < 0].sum()
    return p2, (harm, e_omega, p_neg), prior / prior.sum(), b


if __name__ == "__main__":
    P = load_participants()
    ids = matlab_order(P, "g1")[:2] + matlab_order(P, "g2")[:2]
    settings = [onp.zeros(8), onp.array([0.5, -5, 0, 0, 0.3, -2.0, 0.5, 1.5]),
                onp.array([1.0, 2.0, -0.5, 0.5, -1.0, 4.0, 1.0, 0.5])]
    netp = jax.jit(partial(mm.model_netp, "I"))
    worst_p, worst_a = 0.0, 0.0
    for pid in ids:
        d = P[pid]["data"]
        for prm in settings:
            ours, post = netp(np.asarray(prm), np.asarray(d))
            p2_np, attr_np, prior_np, post_np = numpy_phase2(prm, d)
            # attributions from memo, with the evidence = likelihood of all 54 partner choices
            L = intent.intent_choice(np.asarray(d[36:90, 2:6] / 10.0))
            lik = L[np.arange(54), :, :, np.asarray(d[36:90, 7].astype(int) - 1)]
            logL = np.log(lik).sum(axis=0)
            attr = intent.attribute_intent(np.asarray(prior_np), np.exp(logL - logL.max()))
            e_p = float(onp.abs(onp.asarray(ours)[36:90] - p2_np).max())
            e_a = max(abs(float(a) - b) for a, b in zip(attr, attr_np))
            worst_p, worst_a = max(worst_p, e_p), max(worst_a, e_a)
            print(f"{pid:12s} max|p2 diff| = {e_p:.2e}   E[harm] {float(attr[0]):.3f}/{attr_np[0]:.3f}   "
                  f"E[omega] {float(attr[1]):.3f}/{attr_np[1]:.3f}   P(rho<0) {float(attr[2]):.4f}/{attr_np[2]:.4f}  (memo/numpy)")
    print(f"\nWORST: predictions {worst_p:.2e}, attributions {worst_a:.2e} -> "
          f"{'AGREE' if worst_p < 1e-10 and worst_a < 1e-8 else 'DISAGREE'}")
