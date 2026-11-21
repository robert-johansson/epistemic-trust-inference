"""Check the factorised hierarchical model (hier_core.py) against its memo specification
(hier_spec.py) on the specification's coarse grid, using real participants' payoffs and
partner choices and a range of parameter settings, including strong and weak coupling.

Usage: python -I validate_hier.py
"""
import sys
from pathlib import Path

import jax.numpy as np
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, matlab_order  # noqa: E402
import hier_spec as spec  # noqa: E402
import hier_core as core  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def delta_u(ag, bg, pay):
    """U(option 1) - U(option 2) on the grid, per trial: [t, alpha, beta]."""
    s1, o1, s2, o2 = (pay[:, i][:, None, None] for i in range(4))
    return (ag[None, :, None] * (s1 - s2)
            + bg[None, None, :] * (np.maximum(s1 - o1, 0) - np.maximum(s2 - o2, 0)))


def fast(q, pay1, pay2, pay3, partner_choices, ag, bg):
    Ja, Jb, Wa, Wb, b1, pi = core.beliefs(ag, bg, q)
    p1 = (sigmoid(delta_u(ag, bg, pay1)) * b1).sum(axis=(1, 2))
    du2 = delta_u(ag, bg, pay2)
    lik = np.where(partner_choices[:, None, None] == 0, sigmoid(du2), sigmoid(-du2))   # [t, a, b]
    hist = np.exp(np.cumsum(np.log(lik), axis=0) - np.log(lik))
    post = pi[None] * hist
    post = post / post.sum(axis=(1, 2), keepdims=True)
    p2 = (sigmoid(du2) * post).sum(axis=(1, 2))
    b3 = core.contagion_belief(Ja, Jb, Wa, Wb, np.prod(lik, axis=0))
    p3 = (sigmoid(delta_u(ag, bg, pay3)) * b3).sum(axis=(1, 2))
    return np.concatenate([p1, p2, p3])


if __name__ == "__main__":
    P = load_participants()
    ids = matlab_order(P, "g1")[:2] + matlab_order(P, "g2")[:2]
    settings = [
        (15.0, 0.0, 5.0, 8.0, 3.0, 6.0),       # moderate coupling
        (10.0, -10.0, 4.0, 6.0, 1.0, 1.0),     # strong coupling (people alike)
        (20.0, 5.0, 6.0, 10.0, 60.0, 60.0),    # weak coupling (people independent)
        (5.0, 15.0, 2.0, 3.0, 8.0, 15.0),
    ]
    worst = 0.0
    for pid in ids:
        d = P[pid]["data"]
        pay = np.asarray(d[:, 2:6] / 10.0)
        partner = np.asarray(d[36:90, 7].astype(int) - 1)
        for q in settings:
            a = spec.run(q, pay[:36], pay[36:90], pay[90:], partner)
            b = fast(q, pay[:36], pay[36:90], pay[90:], partner, spec.AG, spec.BG)
            err = float(np.max(np.abs(a - b)))
            worst = max(worst, err)
            print(f"{pid:12s} q={q}: max |spec - factorised| = {err:.2e}")
    print(f"\nWORST: {worst:.2e}  ->  {'AGREE' if worst < 1e-10 else 'DISAGREE'}")
