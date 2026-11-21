"""Line-by-line NumPy (float64) translation of the authors' MATLAB models
(Models/ABA_M1.m ... ABA_M4.m, ABA_Beta.m and Models/Utilities/*.m).

This file is the *reference* for the memo port: it reproduces MATLAB's
arithmetic, including the +eps on every grid prior and the per-trial
normalisation, so the memo models can be checked against it and it can be
checked against the MATLAB outputs saved in the repository.

Returns, per model call, the per-trial probability that the participant
chooses / predicts option 1 ("netp"), the log-likelihood of the observed
choices per phase, and the belief marginals the MATLAB code saves.
"""
import numpy as onp
from scipy.stats import norm

EPS = onp.finfo(float).eps   # MATLAB eps = 2.2204e-16
RES, INC = 30.0, 0.25
T1, T2 = 36, 90              # phase boundaries (T3 = 126)

AG = onp.arange(0, RES + 1e-9, INC / 2)       # alpha grid 0:0.125:30   (241)
BG = onp.arange(-RES, RES + 1e-9, INC)        # beta grid  -30:0.25:30  (241)
ALPHA_GRID, BETA_GRID = onp.meshgrid(AG, BG)  # MATLAB meshgrid: rows = beta, cols = alpha


def sigmoid(x):
    return 1.0 / (1.0 + onp.exp(-x))


def payoffs(data, t):
    return data[t, 2] / 10, data[t, 3] / 10, data[t, 4] / 10, data[t, 5] / 10


def utilities(alpha_grid, beta_grid, s1, o1, s2, o2):
    val1 = alpha_grid * s1 + beta_grid * max(s1 - o1, 0)
    val2 = alpha_grid * s2 + beta_grid * max(s2 - o2, 0)
    return val1, val2


def netp(val1, val2, belief):
    return float(onp.sum(sigmoid(val1 - val2) * belief))


def update(val1, val2, actual_choice, belief):
    lik = sigmoid(val1 - val2) if actual_choice == 1 else sigmoid(val2 - val1)
    b = lik * belief
    return b / b.sum()


def grid_prior(a_mu, a_sd, b_mu, b_sd, beta_only=False):
    if beta_only:
        p = norm.pdf(BG, b_mu, b_sd) + EPS
    else:
        p = norm.pdf(ALPHA_GRID, a_mu, a_sd) * norm.pdf(BETA_GRID, b_mu, b_sd) + EPS
    return p / p.sum()


def convolve(self_m, self_v, self_v_ref, sigma_par, mu_par):
    """Utilities/convolve_distributions.m: precision-weighted self/partner combination."""
    prec = self_v ** -2 + (2 * self_v_ref ** 2 + sigma_par ** 2) ** -1
    v2 = 1 / prec
    m = v2 * (self_v ** -2 * self_m + (2 * self_v_ref ** 2 + sigma_par ** 2) ** -1 * mu_par)
    return onp.sqrt(v2), m


def unpack(model, parms):
    p = onp.asarray(parms, dtype=float)
    if model == "Beta":
        return dict(alpha=None, beta=p[0], alpha_v=None, beta_v=onp.exp(p[1]),
                    alpha_par=None, beta_par=p[0], alpha_v_ref=None, beta_v_ref=onp.exp(p[2]))
    alpha = RES * sigmoid(p[0])
    out = dict(alpha=alpha, beta=p[1], alpha_v=onp.exp(p[2]), beta_v=onp.exp(p[3]))
    if model in ("M1", "M2"):          # self-insertion: partner prior centred on own preferences
        out.update(alpha_par=alpha, beta_par=p[1], alpha_v_ref=onp.exp(p[4]), beta_v_ref=onp.exp(p[5]))
    else:                              # M3, M4: free partner-prior centre
        out.update(alpha_par=RES * sigmoid(p[4]), beta_par=p[5],
                   alpha_v_ref=onp.exp(p[6]), beta_v_ref=onp.exp(p[7]))
    return out


def run(model, parms, data):
    """Evaluate one model on one participant's (126, 9) data. Returns dict."""
    q = unpack(model, parms)
    beta_only = model == "Beta"
    contagion = model in ("M1", "M3", "Beta")

    if beta_only:
        A = 1.0                         # calculate_utilities([], ...) -> alpha fixed at 1
        Bg = BG
    else:
        A, Bg = ALPHA_GRID, BETA_GRID

    self_b = grid_prior(q["alpha"], q["alpha_v"], q["beta"], q["beta_v"], beta_only)
    par_b = grid_prior(q["alpha_par"], q["alpha_v_ref"], q["beta_par"], q["beta_v_ref"], beta_only)

    p = onp.zeros(126)
    for t in range(T1):
        v1, v2 = utilities(A, Bg, *payoffs(data, t))
        p[t] = netp(v1, v2, self_b)
    for t in range(T1, T2):
        v1, v2 = utilities(A, Bg, *payoffs(data, t))
        p[t] = netp(v1, v2, par_b)
        par_b = update(v1, v2, data[t, 7], par_b)

    if beta_only:
        a_marg2b, b_marg2b = None, par_b
    else:
        a_marg2b, b_marg2b = par_b.sum(axis=0), par_b.sum(axis=1)

    if contagion:
        mu_b = onp.sum(BG * b_marg2b)
        sd_b = onp.sqrt(onp.sum((BG - mu_b) ** 2 * (b_marg2b / b_marg2b.sum())))
        cbv, cbm = convolve(q["beta"], q["beta_v"], q["beta_v_ref"], sd_b, mu_b)
        if beta_only:
            phase3_b = norm.pdf(BG, cbm, cbv) + EPS
        else:
            mu_a = onp.sum(AG * a_marg2b)
            sd_a = onp.sqrt(onp.sum((AG - mu_a) ** 2 * (a_marg2b / a_marg2b.sum())))
            cav, cam = convolve(q["alpha"], q["alpha_v"], q["alpha_v_ref"], sd_a, mu_a)
            phase3_b = norm.pdf(ALPHA_GRID, cam, cav) * norm.pdf(BETA_GRID, cbm, cbv) + EPS
        phase3_b = phase3_b / phase3_b.sum()
    else:
        phase3_b = self_b

    for t in range(T2, 126):
        v1, v2 = utilities(A, Bg, *payoffs(data, t))
        p[t] = netp(v1, v2, phase3_b)

    ch = data[:, 6]
    ll = onp.where(ch == 1, onp.log(p), onp.log(1 - p))
    return {
        "netp": p,
        "lik": (ll[:T1].sum(), ll[T1:T2].sum(), ll[T2:].sum()),
        "F": ll.sum() + EPS,
        "self_belief": self_b, "partner_posterior": par_b, "phase3_belief": phase3_b,
        "alpha_marg2b": a_marg2b, "beta_marg2b": b_marg2b,
    }
