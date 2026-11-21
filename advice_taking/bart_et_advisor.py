"""The advisor model: a memo model of advice-taking in the BART-ET (Li et al., 2026); Section 5 and Box 1 of the paper.

NOT a fitted model. Every task number below that Li et al. do not report
(pump scale, advice values, explosion points, positions of the 4 late
violations) is an ASSUMPTION, marked as such (see ASSUMPTIONS.md).

Run from the repository root:
    uv run python -I advice_taking/bart_et_advisor.py

The participant (one balloon in Round 2):
  - is uncertain about the balloon's pop point t (prior shaped by Round 1),
  - is uncertain about the advisor's reliability r (carried across balloons),
  - models the advisor as KNOWING t and, on each balloon, either HELPing
    (advises high-but-safe, adv < t) or MISLEADing (advises adv >= t),
  - observes the advice a, infers t and the advisor's intent, picks pumps p
    by softmax expected points, then observes SAFE/POP and updates r.
"""
import jax
import jax.numpy as np
import numpy as onp
from memo import memo
from enum import IntEnum

Level = np.arange(11)                      # pump bins 0..10 (ASSUMED discretisation)
RelIdx = np.arange(5)                      # index into the reliability grid
REL = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # P(advisor helps on a given balloon)


class Intent(IntEnum):
    HELP = 0
    MISLEAD = 1


class Outcome(IntEnum):
    SAFE = 0
    POP = 1


@jax.jit
def pmf(i, arr):
    return arr[i]


@jax.jit
def p_intent(intent, r):
    rel = REL[r]
    return np.where(intent == 0, rel, 1.0 - rel)


@jax.jit
def advice_lik(adv, t, intent, lam):
    help_w = np.where(adv < np.maximum(t, 1), np.exp(lam * adv), 0.0)  # high but safe
    mis_w = np.where(adv >= t, 1.0, 0.0)                               # at/above pop point
    return np.where(intent == 0, help_w, mis_w)


@memo
def choose_pumps[a: Level, p: Level](rel_belief: ..., thr_prior: ..., beta, lam):
    """Participant sees advised pumps a, reasons about why the advisor said it, and picks pumps p."""
    participant: thinks[
        advisor: given(r in RelIdx, wpp=pmf(r, rel_belief)),
        advisor: given(intent in Intent, wpp=p_intent(intent, r)),
        world: given(t in Level, wpp=pmf(t, thr_prior)),
        advisor: knows(world.t),
        advisor: chooses(adv in Level, wpp=advice_lik(adv, world.t, intent, lam)),
    ]
    participant: observes [advisor.adv] is a
    participant: chooses(p in Level, wpp=exp(beta * E[p * (world.t > p)]))
    return Pr[participant.p == p]


@memo
def update_reliability[a: Level, p: Level, o: Outcome, r: RelIdx](rel_belief: ..., thr_prior: ..., lam):
    """Participant's posterior over advisor reliability after advice a, own pumps p, and outcome o."""
    participant: knows(p, o, r)
    participant: thinks[
        advisor: given(r in RelIdx, wpp=pmf(r, rel_belief)),
        advisor: given(intent in Intent, wpp=p_intent(intent, r)),
        world: given(t in Level, wpp=pmf(t, thr_prior)),
        advisor: knows(world.t),
        advisor: chooses(adv in Level, wpp=advice_lik(adv, world.t, intent, lam)),
    ]
    participant: observes [advisor.adv] is a
    participant: observes_that [(world.t <= p) == (o == {Outcome.POP})]
    return participant[Pr[advisor.r == r]]


def beta_grid(m, k):
    """Beta(m*k, (1-m)*k) prior on the reliability grid: m = trust level, k = rigidity."""
    a, b = m * k, (1 - m) * k
    w = REL ** (a - 1) * (1 - REL) ** (b - 1)
    return w / w.sum()


if __name__ == "__main__":
    # ---- Round-2 schedule: accuracy order from the paper, numbers ASSUMED ----
    acc = [1] * 3 + [0] * 3 + [1] * 5 + [1] * 19   # paper: AAA III AAAAA + (15 A, 4 I)
    for i in [14, 19, 23, 27]:                       # ASSUMED positions of the 4 late violations
        acc[i] = 0
    rng = onp.random.default_rng(0)
    advice = rng.integers(4, 8, size=30)             # ASSUMED advice bins
    T_true = onp.where(onp.array(acc) == 1, onp.minimum(advice + 2, 10), advice)  # ASSUMED pop points

    t = onp.arange(11)
    thr = onp.where(t >= 1, onp.exp(-t / 4.0), 0.0)  # pessimistic prior after rigged Round 1 (ASSUMED)
    thr_prior = np.array(thr / thr.sum())
    BETA, LAM = 1.5, 1.0

    profiles = {
        "trusting (calibrated)": beta_grid(0.70, 4),
        "mistrusting": beta_grid(0.30, 4),
        "credulous (rigid high)": beta_grid(0.85, 40),
        "petrified (rigid low)": beta_grid(0.30, 40),
    }

    cp = jax.jit(jax.vmap(lambda rb: choose_pumps(rb, thr_prior, BETA, LAM)))
    ur = jax.jit(jax.vmap(lambda rb: update_reliability(rb, thr_prior, LAM)))

    N = 200
    key = jax.random.PRNGKey(1)
    for name, prior in profiles.items():
        rb = np.tile(prior, (N, 1))
        dev, follow, mean_rel = [], [], [float(REL @ prior)]
        for k in range(30):
            a = int(advice[k])
            probs = cp(rb)[:, a, :]
            key, sk = jax.random.split(key)
            p = jax.random.categorical(sk, np.log(probs + 1e-30), axis=-1)
            o = (p >= T_true[k]).astype(np.int32)
            rb = ur(rb)[np.arange(N), a, p, o, :]
            dev.append(float(np.abs(p - a).mean()))
            follow.append(float((p == a).mean()))
            mean_rel.append(float((rb @ REL).mean()))
        dev = onp.array(dev)
        print(f"{name:24s} score={dev.mean():.2f}  follow={onp.mean(follow):.2f}  "
              f"blocks(1-3/4-6/7-11/12-30)=" +
              "/".join(f"{dev[s].mean():.2f}" for s in [slice(0, 3), slice(3, 6), slice(6, 11), slice(11, 30)]) +
              "  belief r after 0,3,6,11,30: " + ", ".join(f"{mean_rel[i]:.2f}" for i in [0, 3, 6, 11, 30]))

    # ---- Diagnostics ----
    flat = np.ones(5) / 5
    U = update_reliability(flat, thr_prior, LAM)
    print("\nHot stove: belief in reliability after advice 6 and a SAFE balloon (prior 0.50):",
          ", ".join(f"pumped {p}: {float(U[6, p, 0] @ REL):.2f}" for p in [2, 4, 5, 6]))
    for label, rb in [("rigid trust r=.9", np.array([0, 0, 0, 0, 1.0])),
                      ("rigid mistrust r=.1", np.array([1.0, 0, 0, 0, 0]))]:
        C = choose_pumps(rb, thr_prior, BETA, LAM)
        print(f"E|p-a| vs advice 2..8, {label}: " +
              " ".join(f"{float(C[a] @ np.abs(Level - a)):.2f}" for a in range(2, 9)))
