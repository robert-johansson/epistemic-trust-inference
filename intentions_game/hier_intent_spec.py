"""The coupled-intent model C, written as an idiomatic memo specification (coarse grid).

It combines the hierarchical self-other model H (hier_spec.py) with the intent model I (intent.py):
  - every person, the participant included, is an agent with two goals: `earn` (the decider's
    own points, weight omega) and `sway` (the recipient's points, weight rho: > 0 helping,
    < 0 harming), choosing by softmax over their weighted expected utility;
  - the participant's motives and the partner's motives are two draws from one population whose
    norm is unknown, with spreads tau_omega and tau_rho;
  - the participant knows their own motives only roughly (introspection), and learns the partner's
    from the partner's choices.
Self-insertion and contagion therefore act on motives: how much people care about their own and
others' outcomes. As in H, one coupling parameter per dimension governs both.

Exact enumeration on a 7 x 7 grid. `validate_hier_intent.py` checks the factorised full-grid
computation (hier_core.py + intent.py, model "C" in memo_models) against this specification.
"""
from enum import IntEnum

import jax
import jax.numpy as np
from jax.scipy.stats import norm
from memo import memo

jax.config.update("jax_enable_x64", True)

GA, GB = 7, 7
OMEGA = np.linspace(0.0, 30.0, GA)       # weight on the decider's own points
RHO = np.linspace(-30.0, 30.0, GB)       # weight on the recipient's points (> 0 helping, < 0 harming)
Motive = np.arange(GA * GB)              # a motive profile (omega, rho), flattened row-major
Trial54 = np.arange(54)
Trial36 = np.arange(36)


class Option(IntEnum):
    ONE = 0
    TWO = 1


@jax.jit
def to_decider(c, t, pay):
    """Points the decider gets from option c on trial t; pay[t] = (S1, O1, S2, O2)/10."""
    return np.where(c == 0, pay[t, 0], pay[t, 2])


@jax.jit
def to_recipient(c, t, pay):
    """Points the recipient gets from option c on trial t."""
    return np.where(c == 0, pay[t, 1], pay[t, 3])


@jax.jit
def omega(k):
    return OMEGA[k // GB]


@jax.jit
def rho(k):
    return RHO[k % GB]


@jax.jit
def resembles(k, norm_k, tau_w, tau_r):
    """How likely a person drawn from the population has motives k, given the population norm."""
    return norm.pdf(OMEGA[k // GB], OMEGA[norm_k // GB], tau_w) * norm.pdf(RHO[k % GB], RHO[norm_k % GB], tau_r)


@jax.jit
def introspection(k, a, b, v_a, v_b):
    """How well motives k match the participant's rough sense of their own motives."""
    return norm.pdf(OMEGA[k // GB], a, v_a) * norm.pdf(RHO[k % GB], b, v_b)


@jax.jit
def at_trial(t, k, arr):
    return arr[t, k]


@jax.jit
def at(k, arr):
    return arr[k]


@memo
def predict_partner[t: Trial54, guess: Option](a, b, v_a, v_b, tau_w, tau_r, history: ..., pay: ...):
    """Phase 2: believing that they and the partner are two draws from one population of motives,
    the participant uses what they know of their own motives, and what the partner has done so far,
    to guess the partner's next allocation."""
    participant: knows(t)
    participant: thinks[
        population: given(norm_k in Motive, uniformly),
        population: given(mine in Motive, wpp=resembles(mine, norm_k, tau_w, tau_r)),
        population: given(theirs in Motive, wpp=resembles(theirs, norm_k, tau_w, tau_r)),
        partner: knows(t, population.theirs),
        partner: wants(earn=to_decider(c, t, pay)),
        partner: wants(sway=to_recipient(c, t, pay)),
        partner: chooses(c in Option, wpp=exp(omega(population.theirs) * EU[earn]
                                              + rho(population.theirs) * EU[sway])),
    ]
    participant: observes_event(wpp=introspection(population.mine, a, b, v_a, v_b))
    participant: observes_event(wpp=at_trial(t, population.theirs, history))
    participant: guesses(guess in Option, wpp=Pr[partner.c == guess])
    return Pr[participant.guess == guess]


@memo
def own_choice[t: Trial36, c: Option](a, b, v_a, v_b, tau_w, tau_r, evidence: ..., pay: ...):
    """Phases 1 and 3: the participant allocates points as their self-model would, an agent with the
    same two goals, whose motives are informed by introspection and, through the shared population,
    by everything the partner did (evidence = 1 everywhere before the partner is met)."""
    participant: knows(t)
    participant: thinks[
        population: given(norm_k in Motive, uniformly),
        population: given(mine in Motive, wpp=resembles(mine, norm_k, tau_w, tau_r)),
        population: given(theirs in Motive, wpp=resembles(theirs, norm_k, tau_w, tau_r)),
        myself: knows(t, population.mine),
        myself: wants(earn=to_decider(c, t, pay)),
        myself: wants(sway=to_recipient(c, t, pay)),
        myself: chooses(c in Option, wpp=exp(omega(population.mine) * EU[earn]
                                             + rho(population.mine) * EU[sway])),
    ]
    participant: observes_event(wpp=introspection(population.mine, a, b, v_a, v_b))
    participant: observes_event(wpp=at(population.theirs, evidence))
    participant: chooses(c in Option, wpp=Pr[myself.c == c])
    return Pr[participant.c == c]


@memo
def partner_likelihood[t: Trial54, k: Motive, c: Option](pay: ...):
    """How a partner with motives k allocates points on trial t."""
    partner: knows(t, k)
    partner: wants(earn=to_decider(c, t, pay))
    partner: wants(sway=to_recipient(c, t, pay))
    partner: chooses(c in Option, wpp=exp(omega(k) * EU[earn] + rho(k) * EU[sway]))
    return Pr[partner.c == c]


def run(q, pay1, pay2, pay3, partner_choices):
    """P(option 1) on all 126 trials under the specification, for natural-scale parameters
    q = (a, b, v_a, v_b, tau_w, tau_r)."""
    L = partner_likelihood(pay2)
    lik = L[np.arange(54), :, partner_choices]
    history = np.exp(np.cumsum(np.log(lik), axis=0) - np.log(lik))
    everything = np.prod(lik, axis=0)
    p1 = own_choice(*q, np.ones(GA * GB), pay1)[:, 0]
    p2 = predict_partner(*q, history, pay2)[:, 0]
    p3 = own_choice(*q, everything, pay3)[:, 0]
    return np.concatenate([p1, p2, p3])
