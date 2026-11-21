"""Does the Corriveau trust model beat a simple description of the data?

The paper's ANOVA found main effects of attachment group and task and no interactions, so an additive
description could fit the 24 Table-2 cells about as well. Two checks:

1. Additive baselines on the same 24 cells, same binomial likelihood (trials treated as independent):
     A  logit p = group + task                       (6 parameters)
     B  A + question (ask vs. endorse)               (7)
     C  A + task x question                          (9)
   compared with the memo model (8 parameters) by -logL, AIC, r and RMSE.
2. Generalisation to an unseen task (a test a descriptive model cannot pass): fix the 75-25 picture
   strength at .75 (the stimulus design: 75% of features), fit the memo model to the novel-object and
   50-50 cells only (16 cells), and predict the 8 held-out 75-25 cells. The additive baselines have no
   estimate for an unseen task; their best guess is "no task effect" (75-25 = the 50-50 levels).

Usage: uv run python -I development/corriveau_baselines.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corriveau_trust as ct  # noqa: E402

CELLS, GROUPS, TASKS = ct.CELLS, ct.GROUPS, ct.TASKS
OBS = onp.array([ct.OBS[c] for c in CELLS])                       # [cell, group]
NTR = onp.array([[ct.N[g] * ct.TRIALS[t] for g in range(4)] for (t, q) in CELLS], float)


def binom_nll(p, cells=None):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    idx = range(len(CELLS)) if cells is None else cells
    k = OBS * NTR
    return -sum(np.sum(k[c] * np.log(p[c]) + (NTR[c] - k[c]) * np.log(1 - p[c])) for c in idx)


def additive(theta, kind):
    g = theta[:4]                                        # group levels
    t = np.concatenate([np.zeros(1), theta[4:6]])        # task effects (novel = 0)
    rows = []
    for (task, q) in CELLS:
        x = g + t[TASKS.index(task)]
        if kind in ("B", "C"):
            x = x + (theta[6] if q == "endorse" else 0.0)
        if kind == "C" and task != "novel" and q == "endorse":
            x = x + theta[7 + TASKS.index(task) - 1]
        rows.append(1 / (1 + np.exp(-x)))
    return np.stack(rows)


def fit_additive(kind):
    k = {"A": 6, "B": 7, "C": 9}[kind]
    f = jax.jit(jax.value_and_grad(lambda th: binom_nll(additive(th, kind))))
    res = minimize(lambda x: tuple(onp.asarray(v, float) for v in f(np.asarray(x))), onp.zeros(k),
                   jac=True, method="BFGS")
    return res, onp.asarray(additive(np.asarray(res.x), kind))


def report(label, nll, k, pred, cells=None):
    idx = list(range(len(CELLS))) if cells is None else cells
    o, p = OBS[idx].ravel(), pred[idx].ravel()
    r = onp.corrcoef(p, o)[0, 1]
    rmse = onp.sqrt(onp.mean((p - o) ** 2))
    aic = f"{2 * nll + 2 * k:9.2f}" if k is not None else " " * 9
    print(f"  {label:52s} -logL {nll:8.2f}  AIC {aic}  r = {r:5.2f}  RMSE = {rmse:.3f}")


if __name__ == "__main__":
    print("1. All 24 cells")
    full = ct.fit()
    pred_full = onp.asarray(ct.predict(np.asarray(full.x)))
    report("memo model (8 params)", full.fun, 8, pred_full)
    for kind, label in (("A", "additive A: group + task (6)"), ("B", "additive B: + question (7)"),
                        ("C", "additive C: + task x question (9)")):
        res, pred = fit_additive(kind)
        report(label, res.fun, len(res.x), pred)

    print("\n2. Generalisation: fit novel + 50-50 (16 cells), predict 75-25 (8 cells)")
    train = [i for i, (t, q) in enumerate(CELLS) if t != "75-25"]
    test = [i for i, (t, q) in enumerate(CELLS) if t == "75-25"]
    S75 = 0.75
    x_s75 = float(onp.log((S75 - 0.5) / (1 - S75)))       # invert s = 0.5 + 0.5*sigmoid(x)

    def pred_fixed(th):
        return ct.predict(np.concatenate([th, np.array([x_s75])]))

    f = jax.jit(jax.value_and_grad(lambda th: binom_nll(pred_fixed(th), train)))
    rng = onp.random.default_rng(0)
    best = None
    for start in [onp.zeros(7)] + [rng.normal(0, 1, 7) for _ in range(5)]:
        res = minimize(lambda x: tuple(onp.asarray(v, float) for v in f(np.asarray(x))), start, jac=True,
                       method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    pred = onp.asarray(pred_fixed(np.asarray(best.x)))
    report("memo model, picture strength fixed at .75: held-out 75-25", float(binom_nll(np.asarray(pred), test)),
           None, pred, test)
    resA, _ = minimize(lambda x: float(binom_nll(additive(np.asarray(x), "A"), train)), onp.zeros(6),
                       method="BFGS"), None
    predA = onp.asarray(additive(np.asarray(resA.x), "A"))
    predA_test = predA.copy()
    for i in test:                                         # unseen task: no task effect -> use 50-50 level
        t, q = CELLS[i]
        predA_test[i] = predA[CELLS.index(("50-50", q))]
    report("additive A, unseen task = 50-50 level: held-out 75-25", float(binom_nll(np.asarray(predA_test), test)),
           None, predA_test, test)
    print("\n  held-out 75-25 cells (observed / memo prediction):")
    for i in test:
        print(f"    {CELLS[i][0]} {CELLS[i][1]:8s}" + "".join(
            f"  {GROUPS[g][:4]} {OBS[i, g]:.2f}/{pred[i, g]:.2f}" for g in range(4)))
