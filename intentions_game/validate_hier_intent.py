"""Validate the coupled-intent model C.

(1) Coarse grid: the memo specification (hier_intent_spec.py, explicit population) against the
    factorised computation (hier_core beliefs + direct JAX choice probabilities).
(2) Full grid: model "C" in memo_models (hier_core + memo agent models own_choice_intent,
    predict_intent, intent_choice) against a direct JAX computation of the same equations.

Usage: python -I validate_hier_intent.py
"""
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, matlab_order  # noqa: E402
import hier_intent_spec as spec  # noqa: E402
import hier_core as core  # noqa: E402
import memo_models as mm  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def delta_u(om, rh, pay):
    """U(option 1) - U(option 2) = omega*(S1 - S2) + rho*(O1 - O2), per trial: [t, omega, rho]."""
    ds = (pay[:, 0] - pay[:, 2])[:, None, None]
    do = (pay[:, 1] - pay[:, 3])[:, None, None]
    return om[None, :, None] * ds + rh[None, None, :] * do


def direct(q, pay1, pay2, pay3, partner, om, rh):
    Ja, Jb, Wa, Wb, b1, pi = core.beliefs(om, rh, q)
    p1 = (sigmoid(delta_u(om, rh, pay1)) * b1).sum(axis=(1, 2))
    du2 = delta_u(om, rh, pay2)
    lik = np.where(partner[:, None, None] == 0, sigmoid(du2), sigmoid(-du2))
    hist = np.exp(np.cumsum(np.log(lik), axis=0) - np.log(lik))
    post = pi[None] * hist
    post = post / post.sum(axis=(1, 2), keepdims=True)
    p2 = (sigmoid(du2) * post).sum(axis=(1, 2))
    logL = np.log(lik).sum(axis=0)
    b3 = core.contagion_belief(Ja, Jb, Wa, Wb, np.exp(logL - logL.max()))
    p3 = (sigmoid(delta_u(om, rh, pay3)) * b3).sum(axis=(1, 2))
    return np.concatenate([p1, p2, p3])


if __name__ == "__main__":
    P = load_participants()
    ids = matlab_order(P, "g1")[:2] + matlab_order(P, "g2")[:2]
    settings = [(15.0, 0.0, 5.0, 8.0, 3.0, 6.0), (10.0, -10.0, 4.0, 6.0, 1.0, 1.0),
                (20.0, 5.0, 6.0, 10.0, 60.0, 60.0), (5.0, 15.0, 2.0, 3.0, 8.0, 15.0)]
    worst_coarse = 0.0
    for pid in ids:
        d = P[pid]["data"]
        pay = np.asarray(d[:, 2:6] / 10.0)
        partner = np.asarray(d[36:90, 7].astype(int) - 1)
        for q in settings:
            a = spec.run(q, pay[:36], pay[36:90], pay[90:], partner)
            b = direct(q, pay[:36], pay[36:90], pay[90:], partner, spec.OMEGA, spec.RHO)
            worst_coarse = max(worst_coarse, float(np.max(np.abs(a - b))))
    print(f"(1) coarse grid, memo specification vs factorised: worst |diff| = {worst_coarse:.2e}")

    netp = jax.jit(partial(mm.model_netp, "C"))
    raw = [np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), np.array([0.5, -5.0, 0.5, 1.5, 0.0, 2.0]),
           np.array([-0.5, 3.0, 1.0, 0.5, 2.5, -0.5])]
    worst_full = 0.0
    for pid in ids:
        d = P[pid]["data"]
        pay = np.asarray(d[:, 2:6] / 10.0)
        partner = np.asarray(d[36:90, 7].astype(int) - 1)
        for p in raw:
            ours, _ = netp(p, np.asarray(d))
            q = mm.unpack("C", p)
            ref = direct((q["a"], q["b"], q["v_a"], q["v_b"], q["tau_a"], q["tau_b"]),
                         pay[:36], pay[36:90], pay[90:], partner, mm.AG, mm.BG)
            worst_full = max(worst_full, float(np.max(np.abs(ours - ref))))
    print(f"(2) full grid, memo model C vs direct JAX: worst |diff| = {worst_full:.2e}")
    print("AGREE" if worst_coarse < 1e-10 and worst_full < 1e-10 else "DISAGREE")
