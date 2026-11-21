"""Attachment and selective trust: a memo model of Corriveau et al. (2009), fitted to their Table 2.

Corriveau, K. H., Harris, P. L., Meins, E., Fernyhough, C., Arnott, B., Elliott, L., Liddle, B.,
Hearn, A., Vittorini, L., & de Rosnay, M. (2009). Young children's trust in their mother's claims:
Longitudinal links with attachment security in infancy. Child Development, 80(3), 750-761.


Design (from the paper): attachment (Strange Situation, 15 months): 26 avoidant, 96 secure (Table 2
says 98), 9 resistant, 16 disorganised. Mother and a stranger give conflicting names. On each trial
the child is first asked whom they would like to ask ("ask"), then hears both names and says which is
right ("endorse").
  Novel objects (50 months): 4 label + 4 function trials; no perceptual cue.
  50-50 hybrids (61 months): 4 trials; the picture fits either name equally.
  75-25 hybrids (61 months): 4 trials; the mother always gives the LESS likely name.
Table 2 gives the proportion of trials on which children chose the mother, per group, task and
question. ANOVA: main effects of group and task; no interactions.

The child's theory (the paper's framework): each informant is HELPFUL (knows and tells the true name)
or UNINFORMATIVE (names at random). The child's prior that the mother is helpful is learned from
caregiving history; the stranger prior is shared by all children. Within a task the child updates
beliefs about who is helpful after each trial. Ask: choose the informant believed more helpful
(softmax). Endorse: choose the name more likely true (softmax), optionally weighing agreement with
the mother (the cost route; 0 in the fits).

Assumptions (stated, not from the paper): beliefs reset at the start of each task; the 8 novel-object
trials form one sequence; the 75-25 picture's evidential strength is a free parameter.

Usage: uv run python -I development/corriveau_trust.py
"""
from enum import IntEnum

import jax
import jax.numpy as np
import numpy as onp
from memo import memo
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


class Kind(IntEnum):
    HELPFUL = 0          # knows and tells the true name
    UNINFORMATIVE = 1    # names at random


class Informant(IntEnum):
    MOTHER = 0
    STRANGER = 1


class Name(IntEnum):
    MOTHERS = 0      # the name the mother gives
    STRANGERS = 1    # the conflicting name the stranger gives


@jax.jit
def at2(i, j, arr):
    return arr[i, j]


@jax.jit
def looks_like(name, p_strangers):
    """How well the picture fits each name; p_strangers = 0.5 when the picture does not discriminate."""
    return np.where(name == 1, p_strangers, 1.0 - p_strangers)


@jax.jit
def says(claim, truth, kind):
    """A helpful informant gives the true name; an uninformative one picks either."""
    return np.where(kind == 0, (claim == truth) * 1.0, 0.5)


@memo
def ask[whom: Informant](kinds: ..., beta_ask):
    """Before anyone speaks: whom does the child want to ask, given who has seemed helpful so far?"""
    child: thinks[
        world: given(mother_kind in Kind, stranger_kind in Kind, wpp=at2(mother_kind, stranger_kind, kinds)),
    ]
    child: chooses(whom in Informant, wpp=exp(beta_ask * (
        Pr[world.mother_kind == {Kind.HELPFUL}] if whom == {Informant.MOTHER}
        else Pr[world.stranger_kind == {Kind.HELPFUL}])))
    return Pr[child.whom == whom]


@memo
def endorse[choice: Name](kinds: ..., p_strangers, beta, agreeableness):
    """After hearing conflicting names and seeing the picture: which name does the child accept?"""
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
    child: wants(correct=choice == world.name)
    child: wants(agree=choice == {Name.MOTHERS})
    child: chooses(choice in Name, wpp=exp(beta * (EU[correct] + agreeableness * EU[agree])))
    return Pr[child.choice == choice]


@memo
def update[mk: Kind, sk: Kind](kinds: ..., p_strangers):
    """After the trial: what the child now believes about who is helpful."""
    child: knows(mk, sk)
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
    return child[Pr[world.mother_kind == mk, world.stranger_kind == sk]]


# ---- data: Table 2 (proportion of trials choosing the mother) -------------------------------
GROUPS = ["avoidant", "secure", "resistant", "disorganized"]
N = onp.array([26, 96, 9, 16])                 # Method text (Table 2 header says secure n = 98)
TASKS = ["novel", "50-50", "75-25"]
TRIALS = {"novel": 8, "50-50": 4, "75-25": 4}
OBS = {  # [task][question] -> per group (A, S, R, D)
    ("novel", "ask"): [.59, .71, .74, .63], ("novel", "endorse"): [.57, .63, .75, .62],
    ("50-50", "ask"): [.52, .66, .64, .56], ("50-50", "endorse"): [.43, .63, .81, .48],
    ("75-25", "ask"): [.32, .48, .64, .56], ("75-25", "endorse"): [.48, .40, .53, .48],
}
CELLS = [(t, q) for t in TASKS for q in ("ask", "endorse")]

sig = lambda x: 1.0 / (1.0 + np.exp(-x))


def task_means(trust_mother, trust_stranger, p_strangers, n_trials, beta, beta_ask, learn=True):
    """Mean P(ask mother) and P(endorse mother) over a task's trials, with within-task learning."""
    kinds = np.array([[trust_mother * trust_stranger, trust_mother * (1 - trust_stranger)],
                      [(1 - trust_mother) * trust_stranger, (1 - trust_mother) * (1 - trust_stranger)]])
    a = e = 0.0
    for _ in range(n_trials):
        a = a + ask(kinds, beta_ask)[Informant.MOTHER]
        e = e + endorse(kinds, p_strangers, beta, 0.0)[Name.MOTHERS]
        if learn:
            kinds = update(kinds, p_strangers)
    return a / n_trials, e / n_trials


def predict(theta, shared_trust=False, learn=True):
    """Predicted proportions, array [cell, group]."""
    tm = sig(theta[0]) * np.ones(4) if shared_trust else sig(theta[:4])
    k = 1 if shared_trust else 4
    ts, beta, beta_ask = sig(theta[k]), np.exp(theta[k + 1]), np.exp(theta[k + 2])
    s75 = 0.5 + 0.5 * sig(theta[k + 3])
    cue = {"novel": 0.5, "50-50": 0.5, "75-25": s75}
    out = []
    for t in TASKS:
        a, e = jax.vmap(lambda m: np.stack(task_means(m, ts, cue[t], TRIALS[t], beta, beta_ask, learn)))(tm).T
        out += [a, e]
    return np.stack(out)


def neg_loglik(theta, shared_trust=False, learn=True):
    """Binomial log-likelihood of the cell proportions (trials treated as independent)."""
    p = np.clip(predict(theta, shared_trust, learn), 1e-9, 1 - 1e-9)
    nll = 0.0
    for c, (t, q) in enumerate(CELLS):
        n = np.asarray(N * TRIALS[t], float)
        k = np.asarray(OBS[(t, q)]) * n
        nll = nll - np.sum(k * np.log(p[c]) + (n - k) * np.log(1 - p[c]))
    return nll


def fit(shared_trust=False, learn=True, seed=0):
    f = jax.jit(jax.value_and_grad(lambda th: neg_loglik(th, shared_trust, learn)))
    k = (1 if shared_trust else 4) + 4
    rng = onp.random.default_rng(seed)
    best = None
    for start in [onp.zeros(k)] + [rng.normal(0, 1, k) for _ in range(5)]:
        res = minimize(lambda x: tuple(onp.asarray(v, float) for v in f(np.asarray(x))), start, jac=True,
                       method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    return best


if __name__ == "__main__":
    obs = onp.array([OBS[c] for c in CELLS])
    results = {}
    for label, kw in (("full model", {}), ("lesion: one trust for all groups", {"shared_trust": True}),
                      ("lesion: no within-task learning", {"learn": False})):
        res = fit(**kw)
        pred = onp.asarray(predict(np.asarray(res.x), **kw))
        k = len(res.x)
        aic = 2 * res.fun + 2 * k
        r = onp.corrcoef(pred.ravel(), obs.ravel())[0, 1]
        rmse = onp.sqrt(onp.mean((pred - obs) ** 2))
        results[label] = (res, pred, aic)
        print(f"{label:34s} params {k}  -logL {res.fun:8.2f}  AIC {aic:8.2f}  r = {r:.2f}  RMSE = {rmse:.3f}")
    res, pred, _ = results["full model"]
    th = res.x
    # Conflicting claims rule out "both helpful", so only the RELATIVE odds that the mother vs. the
    # stranger is the uninformative one are identified: mother advantage = log[(1 - ts) / (1 - tm)].
    adv = [float(-onp.log1p(onp.exp(th[4])) + onp.log1p(onp.exp(th[i]))) for i in range(4)]
    print("\nIdentified quantity, mother advantage log[(1-ts)/(1-tm)] by group:",
          ", ".join(f"{g} {a:+.2f}" for g, a in zip(GROUPS, adv)))
    print("Full model: trust in mother by group:",
          ", ".join(f"{g} {float(sig(th[i])):.2f}" for i, g in enumerate(GROUPS)),
          f"| trust in stranger {float(sig(th[4])):.2f}; beta {float(onp.exp(th[5])):.2f}; "
          f"beta_ask {float(onp.exp(th[6])):.2f}; 75-25 picture strength {0.5 + 0.5 * float(sig(th[7])):.2f}")
    print(f"\n{'cell':18s}" + "".join(f"{g:>22s}" for g in GROUPS))
    for c, (t, q) in enumerate(CELLS):
        print(f"{t + ' ' + q:18s}" + "".join(f"{'obs %.2f  pred %.2f' % (obs[c, g], pred[c, g]):>22s}" for g in range(4)))
