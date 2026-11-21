"""memo port of Barnby et al.'s (2025, eLife 14:RP104008) Intentions Game models.

The theory, in memo's terms
---------------------------
A person's social preferences are (alpha, beta): alpha weights their own points,
beta weights *advantageous inequality* (positive: competitive, negative: prosocial).
Utility of an option paying s to self and o to the other: alpha*s + beta*max(s - o, 0),
with payoffs divided by 10. Options are chosen by softmax (temperature 1).

  Phase 1 / 3  `own_choice`:      the participant, uncertain about their own (alpha, beta),
                                  allocates points between self and partner.
  Phase 2      `predict_partner`: the participant *thinks* about a partner with unknown
                                  (alpha, beta), conditions on everything the partner has
                                  done so far, and guesses the partner's next choice.
  Phase 2      `partner_choice`:  the participant's model of how a partner of a given type
                                  chooses; it supplies the evidence from each observed choice.

The four hypotheses (plus the beta-only variant) differ only in where the participant's
prior over the partner is centred and in how phase 3 uses what was learned:

  model | prior over partner centred on   | phase-3 preferences
  ------+---------------------------------+---------------------------------------------
  M1    | own preferences (self-insertion)| shifted toward partner (contagion)
  M2    | own preferences                 | unchanged
  M3    | free parameters                 | shifted toward partner
  M4    | free parameters                 | unchanged
  Beta  | own beta (alpha fixed at 1)     | shifted toward partner

Faithfulness: the grids (alpha 0:0.125:30, beta -30:0.25:30), the +eps on every prior,
parameter transforms and the phase-3 "convolution" are copied from the MATLAB code, and
`validate_memo.py` checks every trial probability against the NumPy translation
(`reference.py`), which matches the MATLAB outputs saved in the repository.

memo's control flow is static, so the 54-trial belief filter is not written as 54 memo
statements. Instead, the memo partner model gives the likelihood of each observed choice
for every partner type, JAX accumulates those likelihoods over trials, and the participant
conditions on the accumulated history with `observes_event`.
"""
from enum import IntEnum

import jax
import jax.numpy as np
from jax.scipy.stats import norm
from memo import memo

jax.config.update("jax_enable_x64", True)

RES, INC = 30.0, 0.25
AG = np.arange(0, 241) * (INC / 2)          # alpha grid: 0, 0.125, ..., 30
BG = -RES + np.arange(0, 241) * INC         # beta grid: -30, -29.75, ..., 30
AI = np.arange(241)                         # domains are grid *indices*
BI = np.arange(241)
Trial36 = np.arange(36)
Trial54 = np.arange(54)
EPS = float(np.finfo(np.float64).eps)


class Option(IntEnum):
    ONE = 0
    TWO = 1


# ---- host functions (scalar in, scalar out) -------------------------------------------
@jax.jit
def at2(i, j, arr):
    return arr[i, j]


@jax.jit
def at3(t, i, j, arr):
    return arr[t, i, j]


@jax.jit
def at1(i, arr):
    return arr[i]


@jax.jit
def at2t(t, j, arr):
    return arr[t, j]


@jax.jit
def utility(c, ia, ib, t, pay):
    """alpha*s + beta*max(s-o, 0) for option c on trial t; pay[t] = (S1, O1, S2, O2) / 10."""
    s = np.where(c == 0, pay[t, 0], pay[t, 2])
    o = np.where(c == 0, pay[t, 1], pay[t, 3])
    return AG[ia] * s + BG[ib] * np.maximum(s - o, 0.0)


@jax.jit
def utility_beta(c, ib, t, pay):
    """Beta-only model: alpha fixed at 1."""
    s = np.where(c == 0, pay[t, 0], pay[t, 2])
    o = np.where(c == 0, pay[t, 1], pay[t, 3])
    return s + BG[ib] * np.maximum(s - o, 0.0)


# ---- the agent models -----------------------------------------------------------------
@memo
def own_choice[t: Trial36, c: Option](belief: ..., pay: ...):
    """Phases 1 and 3: the participant, uncertain about their own preferences, allocates points."""
    participant: knows(t)
    participant: given(ia in AI, ib in BI, wpp=at2(ia, ib, belief))
    participant: chooses(c in Option, wpp=exp(utility(c, ia, ib, t, pay)))
    return Pr[participant.c == c]


@memo
def partner_choice[t: Trial54, ia: AI, ib: BI, c: Option](pay: ...):
    """How a partner with preferences (alpha[ia], beta[ib]) allocates points on trial t."""
    partner: knows(t, ia, ib)
    partner: chooses(c in Option, wpp=exp(utility(c, ia, ib, t, pay)))
    return Pr[partner.c == c]


@memo
def predict_partner[t: Trial54, g: Option](prior: ..., history: ..., pay: ...):
    """Phase 2: having watched the partner's earlier choices, the participant guesses the next one."""
    participant: knows(t)
    participant: thinks[
        partner: knows(t),
        partner: given(ia in AI, ib in BI, wpp=at2(ia, ib, prior)),
        partner: chooses(c in Option, wpp=exp(utility(c, ia, ib, t, pay))),
    ]
    participant: observes_event(wpp=at3(t, partner.ia, partner.ib, history))
    participant: guesses(g in Option, wpp=Pr[partner.c == g])
    return Pr[participant.g == g]


# Beta-only variant (1-D grid over beta, alpha = 1)
@memo
def own_choice_beta[t: Trial36, c: Option](belief: ..., pay: ...):
    """Beta-only model, phases 1 and 3."""
    participant: knows(t)
    participant: given(ib in BI, wpp=at1(ib, belief))
    participant: chooses(c in Option, wpp=exp(utility_beta(c, ib, t, pay)))
    return Pr[participant.c == c]


@memo
def partner_choice_beta[t: Trial54, ib: BI, c: Option](pay: ...):
    """Beta-only model: how a partner with competitiveness beta[ib] allocates points."""
    partner: knows(t, ib)
    partner: chooses(c in Option, wpp=exp(utility_beta(c, ib, t, pay)))
    return Pr[partner.c == c]


@memo
def predict_partner_beta[t: Trial54, g: Option](prior: ..., history: ..., pay: ...):
    """Beta-only model, phase 2."""
    participant: knows(t)
    participant: thinks[
        partner: knows(t),
        partner: given(ib in BI, wpp=at1(ib, prior)),
        partner: chooses(c in Option, wpp=exp(utility_beta(c, ib, t, pay))),
    ]
    participant: observes_event(wpp=at2t(t, partner.ib, history))
    participant: guesses(g in Option, wpp=Pr[partner.c == g])
    return Pr[participant.g == g]


# ---- glue: priors, history, contagion, likelihood ---------------------------------------
def gauss_grid(a_mu, a_sd, b_mu, b_sd):
    """normpdf(alpha) * normpdf(beta) + eps, normalised; layout [ia, ib]."""
    p = norm.pdf(AG, a_mu, a_sd)[:, None] * norm.pdf(BG, b_mu, b_sd)[None, :] + EPS
    return p / p.sum()


def gauss_1d(b_mu, b_sd):
    p = norm.pdf(BG, b_mu, b_sd) + EPS
    return p / p.sum()


def convolve(self_m, self_v, self_v_ref, sigma_par, mu_par):
    """Contagion: precision-weighted combination of own preferences and the learned partner."""
    w_ref = 1.0 / (2 * self_v_ref ** 2 + sigma_par ** 2)
    v2 = 1.0 / (self_v ** -2 + w_ref)
    return np.sqrt(v2), v2 * (self_v ** -2 * self_m + w_ref * mu_par)


def moments(grid, marg):
    mu = np.sum(grid * marg)
    return mu, np.sqrt(np.sum((grid - mu) ** 2 * marg / marg.sum()))


def unpack(model, p):
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    if model in ("H", "C"):    # hierarchical self-other (H) and coupled-intent (C) models: same parameters
        return dict(a=RES * sig(p[0]), b=p[1], v_a=np.exp(p[2]), v_b=np.exp(p[3]),
                    tau_a=np.exp(p[4]), tau_b=np.exp(p[5]))
    if model == "Beta":
        return dict(beta=p[0], beta_v=np.exp(p[1]), beta_par=p[0], beta_v_ref=np.exp(p[2]))
    q = dict(alpha=RES * sig(p[0]), beta=p[1], alpha_v=np.exp(p[2]), beta_v=np.exp(p[3]))
    if model == "I":            # intent model: same transforms as M4; partner prior is over (omega, rho)
        q.update(alpha_par=RES * sig(p[4]), beta_par=p[5], alpha_v_ref=np.exp(p[6]), beta_v_ref=np.exp(p[7]))
        return q
    if model in ("M1", "M2"):   # self-insertion
        q.update(alpha_par=q["alpha"], beta_par=p[1], alpha_v_ref=np.exp(p[4]), beta_v_ref=np.exp(p[5]))
    else:
        q.update(alpha_par=RES * sig(p[4]), beta_par=p[5], alpha_v_ref=np.exp(p[6]), beta_v_ref=np.exp(p[7]))
    return q


def history_weights(lik_obs, log_prior):
    """Exclusive cumulative likelihood of the partner's earlier choices, per trial and type.

    lik_obs[t, ...] = P(partner's observed choice on trial t | type). Returns weights
    proportional to prod_{k<t} lik_obs[k], rescaled per trial for numerical safety
    (the rescaling cancels when the participant normalises), and the final log-posterior.
    """
    logl = np.log(lik_obs)
    cum = np.cumsum(logl, axis=0) - logl              # sum over k < t
    axes = tuple(range(1, cum.ndim))
    shift = np.max(cum + log_prior[None], axis=axes, keepdims=True)
    final = log_prior + cum[-1] + logl[-1]
    return np.exp(cum - shift), final


def split(data):
    pay = np.asarray(data[:, 2:6], dtype=np.float64) / 10.0
    choice = np.asarray(data[:, 6], dtype=np.int32) - 1      # 1/2 -> 0/1
    partner = np.asarray(data[:, 7], dtype=np.int32) - 1
    return pay[:36], pay[36:90], pay[90:], choice, partner[36:90]


def model_netp(model, parms, data):
    """P(option 1) on all 126 trials, plus the phase-2 posterior, for one participant."""
    q = unpack(model, np.asarray(parms, dtype=np.float64))
    pay1, pay2, pay3, _, d2 = split(data)
    t54 = np.arange(54)
    if model == "C":
        # Coupled intent (hier_intent_spec.py): self and partner are agents with the same two goals
        # (own points, the other's points) whose motives (omega, rho) are two draws from one
        # population; beliefs from hier_core (validated in validate_hier_intent.py), choices and
        # predictions from the memo intent models.
        import hier_core
        import intent
        Ja, Jb, Wa, Wb, b1, pi = hier_core.beliefs(
            AG, BG, (q["a"], q["b"], q["v_a"], q["v_b"], q["tau_a"], q["tau_b"]))
        p1 = intent.own_choice_intent(b1, pay1)[:, 0]
        L = intent.intent_choice(pay2)                            # [t, w, r, c]
        lik_obs = L[t54, :, :, d2]
        hist, final = history_weights(lik_obs, np.log(np.maximum(pi, 1e-300)))
        p2 = intent.predict_intent(pi, hist, pay2)[:, 0]
        logL = np.sum(np.log(lik_obs), axis=0)
        b3 = hier_core.contagion_belief(Ja, Jb, Wa, Wb, np.exp(logL - logL.max()))
        p3 = intent.own_choice_intent(b3, pay3)[:, 0]
        post = np.exp(final - final.max()); post = post / post.sum()
        return np.concatenate([p1, p2, p3]), post
    if model == "I":
        # Intent inference (intent.py): M4's self part; the partner is represented by motives
        # (omega: own points; rho: the participant's points). Same 8 parameters as M4, with
        # (alpha_par, beta_par, alpha_v_ref, beta_v_ref) read as the prior over (omega, rho).
        import intent
        self_b = gauss_grid(q["alpha"], q["alpha_v"], q["beta"], q["beta_v"])
        prior = gauss_grid(q["alpha_par"], q["alpha_v_ref"], q["beta_par"], q["beta_v_ref"])
        p1 = own_choice(self_b, pay1)[:, 0]
        L = intent.intent_choice(pay2)                            # [t, w, r, c]
        hist, final = history_weights(L[t54, :, :, d2], np.log(prior))
        p2 = intent.predict_intent(prior, hist, pay2)[:, 0]
        p3 = own_choice(self_b, pay3)[:, 0]
        post = np.exp(final - final.max()); post = post / post.sum()
        return np.concatenate([p1, p2, p3]), post
    if model == "H":
        # Hierarchical self-other model: beliefs from the factorised population computation
        # (hier_core, validated against the memo specification in hier_spec), choices and
        # predictions from the same memo agent models as M1-M4.
        import hier_core
        Ja, Jb, Wa, Wb, b1, pi = hier_core.beliefs(
            AG, BG, (q["a"], q["b"], q["v_a"], q["v_b"], q["tau_a"], q["tau_b"]))
        p1 = own_choice(b1, pay1)[:, 0]
        L = partner_choice(pay2)                                  # [t, ia, ib, c]
        lik_obs = L[t54, :, :, d2]
        log_pi = np.log(np.maximum(pi, 1e-300))
        hist, final = history_weights(lik_obs, log_pi)
        p2 = predict_partner(pi, hist, pay2)[:, 0]
        logL = np.sum(np.log(lik_obs), axis=0)                    # all the partner's choices
        b3 = hier_core.contagion_belief(Ja, Jb, Wa, Wb, np.exp(logL - logL.max()))
        p3 = own_choice(b3, pay3)[:, 0]
        post = np.exp(final - final.max()); post = post / post.sum()
        return np.concatenate([p1, p2, p3]), post
    if model == "Beta":
        self_b = gauss_1d(q["beta"], q["beta_v"])
        prior = gauss_1d(q["beta_par"], q["beta_v_ref"])
        p1 = own_choice_beta(self_b, pay1)[:, 0]
        L = partner_choice_beta(pay2)                         # [t, ib, c]
        hist, final = history_weights(L[t54, :, d2], np.log(prior))
        p2 = predict_partner_beta(prior, hist, pay2)[:, 0]
        post = np.exp(final - final.max()); post = post / post.sum()
        mu_b, sd_b = moments(BG, post)
        cbv, cbm = convolve(q["beta"], q["beta_v"], q["beta_v_ref"], sd_b, mu_b)
        p3 = own_choice_beta(gauss_1d(cbm, cbv), pay3)[:, 0]
        return np.concatenate([p1, p2, p3]), post
    self_b = gauss_grid(q["alpha"], q["alpha_v"], q["beta"], q["beta_v"])
    prior = gauss_grid(q["alpha_par"], q["alpha_v_ref"], q["beta_par"], q["beta_v_ref"])
    p1 = own_choice(self_b, pay1)[:, 0]
    L = partner_choice(pay2)                                  # [t, ia, ib, c]
    hist, final = history_weights(L[t54, :, :, d2], np.log(prior))
    p2 = predict_partner(prior, hist, pay2)[:, 0]
    post = np.exp(final - final.max()); post = post / post.sum()
    if model in ("M1", "M3"):                                 # contagion
        mu_a, sd_a = moments(AG, post.sum(axis=1))
        mu_b, sd_b = moments(BG, post.sum(axis=0))
        cav, cam = convolve(q["alpha"], q["alpha_v"], q["alpha_v_ref"], sd_a, mu_a)
        cbv, cbm = convolve(q["beta"], q["beta_v"], q["beta_v_ref"], sd_b, mu_b)
        b3 = gauss_grid(cam, cav, cbm, cbv)
    else:
        b3 = self_b
    p3 = own_choice(b3, pay3)[:, 0]
    return np.concatenate([p1, p2, p3]), post


def loglik(model, parms, data):
    """Log-likelihood of the participant's 126 responses (MATLAB's F, minus its +eps)."""
    netp, _ = model_netp(model, parms, data)
    choice = np.asarray(data[:, 6], dtype=np.int32) - 1
    return np.sum(np.where(choice == 0, np.log(netp), np.log1p(-netp)))
