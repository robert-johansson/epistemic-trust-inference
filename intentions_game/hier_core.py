"""Factorised computation of the hierarchical self-other model (see hier_spec.py).

The specification enumerates the joint space (norm x own profile x partner profile), which
is |grid|^3 and infeasible on the authors' 241 x 241 grid. Because the population norm is
uniform over the grid and `resembles` factorises over alpha and beta, the norm can be summed
out first, giving one |grid_d| x |grid_d| matrix per dimension:

    J_d[s, q] = sum_mu P(mu) P(s | mu) P(q | mu),   P(s | mu) = N(s; mu, tau_d), normalised over s.

All beliefs then follow from matrix products (layout [alpha index, beta index]):

    own preferences, phase 1:   b1(s) ∝ W(s) * sum_q J(s, q)
    partner, insertion prior:   pi(q) ∝ sum_s J(s, q) W(s)
    partner, trial t:           pi_t(q) ∝ pi(q) * prod_{k<t} lik_k(q)
    own preferences, phase 3:   b3(s) ∝ W(s) * sum_q J(s, q) L(q)       (contagion)

with W(s) = N(alpha_s; a, v_a) N(beta_s; b, v_b) (introspection) and L the likelihood of all
the partner's choices. This is the same discrete model as the specification, computed in a
different order; validate_hier.py checks agreement.
"""
import jax.numpy as np
from jax.scipy.stats import norm


def _scaled(logw, axis=None):
    """exp(logw) rescaled by its maximum: same normalised distribution, no underflow."""
    return np.exp(logw - np.max(logw, axis=axis, keepdims=axis is not None))


def coupling_kernel(grid, tau):
    """J[s, q] = sum_mu P(mu) P(s|mu) P(q|mu) with mu uniform on the grid."""
    D = _scaled(norm.logpdf(grid[None, :], grid[:, None], tau), axis=1)   # [mu, s]
    D = D / D.sum(axis=1, keepdims=True)
    return D.T @ D / grid.size


def introspection_weights(ag, bg, a, b, v_a, v_b):
    """W up to a constant per dimension (all beliefs built from it are normalised)."""
    return _scaled(norm.logpdf(ag, a, v_a)), _scaled(norm.logpdf(bg, b, v_b))


def beliefs(ag, bg, q):
    """Return (Ja, Jb, Wa, Wb, b1, pi) for natural-scale q = (a, b, v_a, v_b, tau_a, tau_b)."""
    a, b, v_a, v_b, tau_a, tau_b = q
    Ja, Jb = coupling_kernel(ag, tau_a), coupling_kernel(bg, tau_b)
    Wa, Wb = introspection_weights(ag, bg, a, b, v_a, v_b)
    b1 = (Wa * Ja.sum(axis=1))[:, None] * (Wb * Jb.sum(axis=1))[None, :]
    b1 = b1 / b1.sum()
    pi = (Ja.T @ Wa)[:, None] * (Jb.T @ Wb)[None, :]
    pi = pi / pi.sum()
    return Ja, Jb, Wa, Wb, b1, pi


def contagion_belief(Ja, Jb, Wa, Wb, L):
    """Own preferences after the partner: W(s) * (Ja @ L @ Jb.T)(s), normalised. L in [alpha, beta]."""
    b3 = Wa[:, None] * Wb[None, :] * (Ja @ L @ Jb.T)
    return b3 / b3.sum()
