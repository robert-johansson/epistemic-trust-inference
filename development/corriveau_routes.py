"""Two routes to the resistant pattern in the Corriveau et al. (2009) trust model (illustration).

Same informant model as corriveau_trust.py. One trial, no learning. A child endorses the mother's
name either because they believe she is reliable (credulous prior) or because disagreeing with her
is costly (agreeableness > 0). Endorsements can match while private beliefs differ.
Parameters are chosen for display, not fitted.

Usage: uv run python -I development/corriveau_routes.py
"""
import sys
from pathlib import Path

import jax.numpy as np
from memo import memo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corriveau_trust import Kind, Name, Informant, at2, looks_like, says, endorse  # noqa: E402,F401


@memo
def believes[n: Name](kinds: ..., p_strangers):
    """After hearing conflicting names and seeing the picture: how likely the child thinks each name is."""
    child: knows(n)
    child: thinks[
        world: given(mother_kind in Kind, stranger_kind in Kind, wpp=at2(mother_kind, stranger_kind, kinds)),
        world: given(name in Name, wpp=looks_like(name, p_strangers)),
        mother: knows(world.name, world.mother_kind),
        mother: chooses(claim in Name, wpp=says(claim, world.name, world.mother_kind)),
        stranger: knows(world.name, world.stranger_kind),
        stranger: chooses(claim in Name, wpp=says(claim, world.name, world.stranger_kind)),
    ]
    child: observes_that[mother.claim == {Name.MOTHERS}]
    child: observes_that[stranger.claim == {Name.STRANGERS}]
    return child[Pr[world.name == n]]


def kinds(tm, ts):
    return np.array([[tm * ts, tm * (1 - ts)], [(1 - tm) * ts, (1 - tm) * (1 - ts)]])


if __name__ == "__main__":
    TS, BETA = 0.6, 6.0
    profiles = [("secure (trust in mother .80)", 0.80, 0.0),
                ("avoidant (.60, mother = stranger)", 0.60, 0.0),
                ("resistant via credulous prior (.97)", 0.97, 0.0),
                ("resistant via cost of disagreeing (.80, agreeableness .4)", 0.80, 0.4)]
    print(f"stranger trust {TS}, beta {BETA}; single trial")
    print(f"{'profile':60s} {'endorse mother: ambiguous / picture favours stranger':>52s}   believes mother: ambiguous / favours stranger")
    for label, tm, agree in profiles:
        k = kinds(tm, TS)
        e = [float(endorse(k, s, BETA, agree)[Name.MOTHERS]) for s in (0.5, 0.75)]
        b = [float(believes(k, s)[Name.MOTHERS]) for s in (0.5, 0.75)]
        print(f"{label:60s} {e[0]:26.2f} / {e[1]:.2f} {'':20s} {b[0]:.2f} / {b[1]:.2f}")
