"""The hierarchical self-other model, written as an idiomatic memo specification.

Theory (the participant's intuitive theory of people):
  - People's social preferences (alpha: weight on own points; beta: weight on being ahead)
    are drawn from one population whose norm is unknown.
  - The participant's own preferences are one draw; the partner's are another.
  - The participant knows their own preferences only roughly (introspection), and learns
    about the partner from the partner's choices.
  - Self-insertion (self -> other) and contagion (other -> self) both follow from this one
    belief, with one coupling parameter per dimension: the population spread tau.

This file runs the model by exact enumeration on a coarse grid (7 x 7 profiles, 7 x 7 norms).
It is the reference semantics. `hier_core.py` computes the same discrete model in factorised
order, so that it can run on the authors' full 241 x 241 grid; `validate_hier.py` checks that
the two agree.
"""
from enum import IntEnum

import jax
import jax.numpy as np
from jax.scipy.stats import norm
from memo import memo

jax.config.update("jax_enable_x64", True)

GA, GB = 7, 7
AG = np.linspace(0.0, 30.0, GA)         # alpha grid (weight on own points)
BG = np.linspace(-30.0, 30.0, GB)       # beta grid (weight on advantageous inequality)
Pref = np.arange(GA * GB)               # a preference profile (alpha, beta), flattened row-major
Trial54 = np.arange(54)
Trial36 = np.arange(36)


class Option(IntEnum):
    ONE = 0
    TWO = 1


@jax.jit
def utility(c, k, t, pay):
    """alpha*s + beta*max(s - o, 0) for option c on trial t, for profile k; pay[t] = (S1, O1, S2, O2)/10."""
    s = np.where(c == 0, pay[t, 0], pay[t, 2])
    o = np.where(c == 0, pay[t, 1], pay[t, 3])
    return AG[k // GB] * s + BG[k % GB] * np.maximum(s - o, 0.0)


@jax.jit
def resembles(k, norm_k, tau_a, tau_b):
    """How likely a person drawn from the population has profile k, given the population norm."""
    return norm.pdf(AG[k // GB], AG[norm_k // GB], tau_a) * norm.pdf(BG[k % GB], BG[norm_k % GB], tau_b)


@jax.jit
def introspection(k, a, b, v_a, v_b):
    """How well profile k matches the participant's rough sense of their own preferences."""
    return norm.pdf(AG[k // GB], a, v_a) * norm.pdf(BG[k % GB], b, v_b)


@jax.jit
def at_trial(t, k, arr):
    return arr[t, k]


@jax.jit
def at(k, arr):
    return arr[k]


@memo
def partner_likelihood[t: Trial54, k: Pref, c: Option](pay: ...):
    """How a partner with profile k allocates points on trial t."""
    partner: knows(t, k)
    partner: chooses(c in Option, wpp=exp(utility(c, k, t, pay)))
    return Pr[partner.c == c]


@memo
def predict_partner[t: Trial54, guess: Option](a, b, v_a, v_b, tau_a, tau_b, history: ..., pay: ...):
    """Phase 2: believing that they and the partner are two draws from one population of social
    preferences, the participant uses what they know about themselves, and what the partner has
    done so far, to guess the partner's next allocation."""
    participant: knows(t)
    participant: thinks[
        population: given(norm_k in Pref, uniformly),
        population: given(mine in Pref, wpp=resembles(mine, norm_k, tau_a, tau_b)),
        population: given(theirs in Pref, wpp=resembles(theirs, norm_k, tau_a, tau_b)),
        partner: knows(t, population.theirs),
        partner: chooses(c in Option, wpp=exp(utility(c, population.theirs, t, pay))),
    ]
    participant: observes_event(wpp=introspection(population.mine, a, b, v_a, v_b))
    participant: observes_event(wpp=at_trial(t, population.theirs, history))
    participant: guesses(guess in Option, wpp=Pr[partner.c == guess])
    return Pr[participant.guess == guess]


@memo
def own_choice[t: Trial36, c: Option](a, b, v_a, v_b, tau_a, tau_b, evidence: ..., pay: ...):
    """Phases 1 and 3: the participant allocates points as their self-model would, where the
    self-model is informed by introspection and, through the shared population, by everything
    the partner did (evidence = 1 everywhere before the partner is met)."""
    participant: knows(t)
    participant: thinks[
        population: given(norm_k in Pref, uniformly),
        population: given(mine in Pref, wpp=resembles(mine, norm_k, tau_a, tau_b)),
        population: given(theirs in Pref, wpp=resembles(theirs, norm_k, tau_a, tau_b)),
        myself: knows(t, population.mine),
        myself: chooses(c in Option, wpp=exp(utility(c, population.mine, t, pay))),
    ]
    participant: observes_event(wpp=introspection(population.mine, a, b, v_a, v_b))
    participant: observes_event(wpp=at(population.theirs, evidence))
    participant: chooses(c in Option, wpp=Pr[myself.c == c])
    return Pr[participant.c == c]


def run(q, pay1, pay2, pay3, partner_choices):
    """P(option 1) on all 126 trials under the specification, for natural-scale parameters
    q = (a, b, v_a, v_b, tau_a, tau_b) and one participant's payoffs and partner choices."""
    L = partner_likelihood(pay2)                                  # [t, k, c]
    lik = L[np.arange(54), :, partner_choices]                    # [t, k]
    history = np.exp(np.cumsum(np.log(lik), axis=0) - np.log(lik))  # prod over trials before t
    everything = np.prod(lik, axis=0)
    p1 = own_choice(*q, np.ones(GA * GB), pay1)[:, 0]
    p2 = predict_partner(*q, history, pay2)[:, 0]
    p3 = own_choice(*q, everything, pay3)[:, 0]
    return np.concatenate([p1, p2, p3])
