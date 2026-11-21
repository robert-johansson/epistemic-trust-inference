"""Intent-inference model I: the participant infers WHY the partner allocates points.

Built as the helping/hindering model of social goal inference (Ullman et al., 2009; BMC ch. 14,
eq. 14.26: B's utility = rho * U_A - cost). Here the partner has two named goals:
  earn  - the partner's own points (self-interest, weight omega >= 0);
  sway  - the participant's points (weight rho: > 0 helping, < 0 harming).
The participant inverts this from the partner's choices. The two post-phase-2 ratings in the
data ask exactly these questions:
  SI  "driven by the desire to earn points"                -> posterior E[omega]
  HI  "driven by the desire to reduce the participant's points" -> posterior E[max(-rho, 0)]
Links fixed before looking at any rating correlation (no fitted rating parameters), so the
ratings are an out-of-sample test. Both questions ask for "the extent to which", so the primary
links are graded: HI <-> expected harm weight E[max(-rho, 0)], SI <-> E[omega]. P(rho < 0) is
secondary (it saturates near 1 because the partner algorithm leans competitive).

Model I = M4's self part (phases 1 and 3: own preferences, no contagion) + this partner model
(phase 2), with the same 8 parameters and transforms as M4. I vs. M4 therefore asks one
question: do participants represent the partner by competitiveness (beta * max(s - o, 0)) or by
benevolence/harm (rho * o)?

Grids: omega on 0..30 (step 0.125), rho on -30..30 (step 0.25), as alpha and beta in the port.
"""
import jax
import jax.numpy as np
from memo import memo

from memo_models import AI, BI, Trial54, Option, AG, BG, at2, at3

OMEGA, RHO = AG, BG
WI, RI = AI, BI


@jax.jit
def their_points(c, t, pay):
    """Points the partner (the decider) gets from option c on trial t; pay[t] = (S1, O1, S2, O2)/10."""
    return np.where(c == 0, pay[t, 0], pay[t, 2])


@jax.jit
def my_points(c, t, pay):
    """Points the participant (the recipient) gets from option c on trial t."""
    return np.where(c == 0, pay[t, 1], pay[t, 3])


@jax.jit
def omega(w):
    return OMEGA[w]


@jax.jit
def rho(r):
    return RHO[r]


@jax.jit
def harm(r):
    """How much the partner wants to reduce the participant's points: max(-rho, 0)."""
    return np.maximum(-RHO[r], 0.0)


@memo
def intent_choice[t: Trial54, w: WI, r: RI, c: Option](pay: ...):
    """How a partner who weighs their own points by omega[w] and the participant's by rho[r] allocates."""
    partner: knows(t, w, r)
    partner: wants(earn=their_points(c, t, pay))
    partner: wants(sway=my_points(c, t, pay))
    partner: chooses(c in Option, wpp=exp(omega(w) * EU[earn] + rho(r) * EU[sway]))
    return Pr[partner.c == c]


@memo
def predict_intent[t: Trial54, guess: Option](prior: ..., history: ..., pay: ...):
    """Phase 2: the participant guesses the partner's next allocation by inferring the partner's
    motives (self-interest and benevolence or harm) from what the partner has done so far."""
    participant: knows(t)
    participant: thinks[
        partner: knows(t),
        partner: given(w in WI, r in RI, wpp=at2(w, r, prior)),
        partner: wants(earn=their_points(c, t, pay)),
        partner: wants(sway=my_points(c, t, pay)),
        partner: chooses(c in Option, wpp=exp(omega(w) * EU[earn] + rho(r) * EU[sway])),
    ]
    participant: observes_event(wpp=at3(t, partner.w, partner.r, history))
    participant: guesses(guess in Option, wpp=Pr[partner.c == guess])
    return Pr[participant.guess == guess]


@memo
def attribute_intent(prior: ..., evidence: ...):
    """After phase 2: how strongly the participant attributes harmful intent (wanting to reduce my
    points) and self-interest (wanting to earn points) to the partner."""
    participant: thinks[
        partner: given(w in WI, r in RI, wpp=at2(w, r, prior)),
    ]
    participant: observes_event(wpp=at2(partner.w, partner.r, evidence))
    return participant[E[harm(partner.r)]]
    return participant[E[omega(partner.w)]]
    return participant[Pr[rho(partner.r) < 0]]


# ---- coupled-intent model C (hier_intent_spec.py): the participant's own choices ----------------
from memo_models import Trial36  # noqa: E402


@jax.jit
def to_decider(c, t, pay):
    """Points the decider (here: the participant) gets from option c on trial t."""
    return np.where(c == 0, pay[t, 0], pay[t, 2])


@jax.jit
def to_recipient(c, t, pay):
    """Points the recipient (here: the partner) gets from option c on trial t."""
    return np.where(c == 0, pay[t, 1], pay[t, 3])


@memo
def own_choice_intent[t: Trial36, c: Option](belief: ..., pay: ...):
    """Phases 1 and 3 of model C: the participant allocates points as their self-model would, an
    agent with the same two goals as the partner, whose motives (omega, rho) are uncertain."""
    participant: knows(t)
    participant: thinks[
        myself: knows(t),
        myself: given(w in WI, r in RI, wpp=at2(w, r, belief)),
        myself: wants(earn=to_decider(c, t, pay)),
        myself: wants(sway=to_recipient(c, t, pay)),
        myself: chooses(c in Option, wpp=exp(omega(w) * EU[earn] + rho(r) * EU[sway])),
    ]
    participant: chooses(c in Option, wpp=Pr[myself.c == c])
    return Pr[participant.c == c]
