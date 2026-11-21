"""Per-participant MAP fits and Laplace evidence for the memo models, mirroring the
first stage of the authors' pipeline (cbm_lap: Gaussian prior, mean 0, variance 6.5).

The authors' second stage (cbm_hbi, hierarchical Bayesian inference; Piray et al., 2019)
is a MATLAB toolbox and is not ported. `bms.py` runs random-effects Bayesian model
selection on these Laplace evidences and compares the outcome with the authors' HBI.

Optimiser: SciPy BFGS on the exact JAX gradient (JAX's own BFGS line search failed on
some participants). Starts: the authors' HBI estimate and the prior mean (zeros); a few
random starts are added if neither converges to a positive-definite optimum.

Usage: python -I fit.py g1|g2 [n_max] [--models H,M1,...]
       -> results/laplace_<group>.npz, or results/laplace_<group>_<models>.npz with --models
"""
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, matlab_order, load_hbi, FIT_ORDER, N_PARAMS  # noqa: E402
import memo_models as mm  # noqa: E402

PRIOR_VAR = 6.5
OUT = Path(__file__).resolve().parent / "results"


def neg_log_post(model, parms, data):
    lp = -0.5 * np.sum(parms ** 2) / PRIOR_VAR - 0.5 * parms.size * np.log(2 * np.pi * PRIOR_VAR)
    return -(mm.loglik(model, parms, data) + lp)


ALL_MODELS = FIT_ORDER + ["H", "I", "C"]        # H: hierarchical self-other; I: intent; C: coupled intent
N_PARAMS = {**N_PARAMS, "H": 6, "I": 8, "C": 6}
VG = {m: jax.jit(jax.value_and_grad(partial(neg_log_post, m))) for m in ALL_MODELS}
HESS = {m: jax.jit(jax.hessian(partial(neg_log_post, m))) for m in ALL_MODELS}
LL = {m: jax.jit(partial(mm.loglik, m)) for m in ALL_MODELS}


def warm_start(model, hbi, i):
    """The authors' HBI estimate where one exists. For H, use M1's estimate: same first four
    parameters, and population spread tau = v_ref / sqrt(2) (two draws differ by 2 tau^2)."""
    if model in hbi["parameters"]:
        return hbi["parameters"][model][i]
    if model == "I":
        # M4's estimate, with competitiveness beta read as harm: rho ~ -beta.
        m4 = onp.asarray(hbi["parameters"]["M4"][i], float)
        return onp.concatenate([m4[:5], [-m4[5]], m4[6:]])
    m1 = onp.asarray(hbi["parameters"]["M1"][i], float)
    return onp.concatenate([m1[:4], m1[4:6] - 0.5 * onp.log(2.0)])


def fit_one(model, data, starts, rng):
    def fun(x):
        v, g = VG[model](np.asarray(x), data)
        return float(v), onp.asarray(g, dtype=float)

    best = None
    tried = list(starts)
    k = 0
    while k < len(tried):
        res = minimize(fun, onp.asarray(tried[k], float), jac=True, method="BFGS",
                       options={"gtol": 1e-5, "maxiter": 1000})
        H = onp.asarray(HESS[model](np.asarray(res.x), data))
        posdef = bool(onp.all(onp.linalg.eigvalsh(0.5 * (H + H.T)) > 0))
        ok = posdef and (res.success or onp.linalg.norm(res.jac) < 1e-3)
        if best is None or (ok, -res.fun) > (best[0], -best[1].fun):
            best = (ok, res, H)
        k += 1
        if k == len(tried) and not best[0] and len(tried) < len(starts) + 4:
            tried.append(rng.normal(0, 1.5, size=N_PARAMS[model]))
    ok, res, H = best
    d = res.x.size
    sign, logdet = onp.linalg.slogdet(H)
    log_ev = -res.fun + 0.5 * d * onp.log(2 * onp.pi) - 0.5 * logdet
    return res.x, float(LL[model](np.asarray(res.x), data)), float(log_ev), ok, len(tried)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--models")]
    models = FIT_ORDER
    for a in sys.argv[1:]:
        if a.startswith("--models"):
            models = a.split("=", 1)[1].split(",")
    group = args[0]
    n_max = int(args[1]) if len(args) > 1 else None
    suffix = "" if models == FIT_ORDER else "_" + "-".join(models)
    OUT.mkdir(exist_ok=True)
    P = load_participants()
    order = matlab_order(P, group)[:n_max]
    hbi = load_hbi(group)
    rng = onp.random.default_rng(1)
    k = len(models)
    ev = onp.full((len(order), k), onp.nan)
    ll = onp.full((len(order), k), onp.nan)
    ok = onp.zeros((len(order), k), bool)
    nstart = onp.zeros((len(order), k), int)
    params = {m: onp.full((len(order), N_PARAMS[m]), onp.nan) for m in models}
    t0 = time.time()
    for i, pid in enumerate(order):
        d = np.asarray(P[pid]["data"])
        for j, m in enumerate(models):
            if m == "C":
                # start from this participant's fitted intent-model motives (partner prior), tau = 1
                zi = onp.load(OUT / f"laplace_{group}_I.npz")
                assert zi["ids"][i] == pid
                starts = [onp.concatenate([zi["params_I"][i][4:8], [0.0, 0.0]]), onp.zeros(6)]
            else:
                starts = [warm_start(m, hbi, i), onp.zeros(N_PARAMS[m])]
            x, l, e, good, n = fit_one(m, d, starts, rng)
            params[m][i], ll[i, j], ev[i, j], ok[i, j], nstart[i, j] = x, l, e, good, n
        print(f"{group} {i + 1}/{len(order)} {pid}: best={models[int(onp.argmax(ev[i]))]} "
              f"ok={ok[i].all()} ({time.time() - t0:.0f}s)", flush=True)
        onp.savez(OUT / f"laplace_{group}{suffix}.npz", ids=onp.array(order), models=onp.array(models),
                  log_ev=ev, loglik=ll, ok=ok, nstart=nstart, done=i + 1,
                  **{f"params_{m}": params[m] for m in models})
    print("done", group, f"{time.time() - t0:.0f}s; all ok: {ok.all()} ({ok.mean():.3f})")
