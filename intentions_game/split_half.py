"""Non-circular test of representation vs. accuracy: estimate each participant's partner
representation from the first half of phase 2, test prediction accuracy on the second half.

Fit (MAP + Laplace, prior N(0, 6.5), as fit.py) on phase 1 + phase-2 trials 1-27 only (63 trials;
phase 3 excluded). Predictions on those trials depend only on the partner's earlier choices, so the
masked likelihood is exact. Starts: the prior mean and random draws only (never the full-data fits).
  R1_half = log evidence(I) - log evidence(M4)   (identical except the partner representation)
  R2_half = log evidence(C) - log evidence(H)    (both coupled)
Outcome: accuracy on phase-2 trials 28-54 (never used in fitting).
Tests: Spearman(R_half, accuracy_2nd) within each group and group-controlled; accuracy of
R_half > 0 vs R_half < 0 participants within each group.

Usage: python -I split_half.py run PART NPARTS   -> results/split_half_partPART.npz (checkpointed)
       python -I split_half.py analyze NPARTS
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
from data import load_participants, matlab_order  # noqa: E402
import memo_models as mm  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
MODELS = {"I": 8, "M4": 8, "C": 6, "H": 6}
PRIOR_VAR = 6.5
MASK = onp.zeros(126, bool)
MASK[:36] = True          # phase 1
MASK[36:63] = True        # phase-2 trials 1-27
MASK_J = np.asarray(MASK)


def neg_log_post(model, parms, data):
    netp, _ = mm.model_netp(model, parms, data)
    choice = np.asarray(data[:, 6], dtype=np.int32) - 1
    ll = np.where(choice == 0, np.log(netp), np.log1p(-netp))
    loglik = np.sum(np.where(MASK_J, ll, 0.0))
    lp = -0.5 * np.sum(parms ** 2) / PRIOR_VAR - 0.5 * parms.size * np.log(2 * np.pi * PRIOR_VAR)
    return -(loglik + lp)


VG = {m: jax.jit(jax.value_and_grad(partial(neg_log_post, m))) for m in MODELS}
HESS = {m: jax.jit(jax.hessian(partial(neg_log_post, m))) for m in MODELS}


def laplace(model, data, rng):
    fun = lambda x: tuple(onp.asarray(v, float) for v in VG[model](np.asarray(x), data))
    best = None
    for start in [onp.zeros(MODELS[model]), rng.normal(0, 1, MODELS[model]), rng.normal(0, 1, MODELS[model])]:
        res = minimize(fun, start, jac=True, method="BFGS", options={"gtol": 1e-5, "maxiter": 1000})
        H = onp.asarray(HESS[model](np.asarray(res.x), data))
        ok = bool(onp.all(onp.linalg.eigvalsh(0.5 * (H + H.T)) > 0))
        if best is None or (ok, -res.fun) > (best[0], -best[1].fun):
            best = (ok, res, H)
    ok, res, H = best
    _, logdet = onp.linalg.slogdet(H)
    return -res.fun + 0.5 * res.x.size * onp.log(2 * onp.pi) - 0.5 * logdet, ok


def run(part, nparts):
    P = load_participants()
    rows = [(pid, g) for g in ("g1", "g2") for pid in matlab_order(P, g)][part::nparts]
    rng = onp.random.default_rng(500 + part)
    out = RES / f"split_half_part{part}.npz"
    ev = onp.full((len(rows), len(MODELS)), onp.nan)
    ok = onp.zeros((len(rows), len(MODELS)), bool)
    t0 = time.time()
    for i, (pid, g) in enumerate(rows):
        d = np.asarray(P[pid]["data"])
        for j, m in enumerate(MODELS):
            ev[i, j], ok[i, j] = laplace(m, d, rng)
        print(f"part {part}: {i + 1}/{len(rows)} {g} {pid} ({time.time() - t0:.0f}s)", flush=True)
        onp.savez(out, ids=onp.array([r[0] for r in rows]), groups=onp.array([r[1] for r in rows]),
                  log_ev=ev, ok=ok, models=onp.array(list(MODELS)), done=i + 1)


def analyze(nparts):
    from scipy.stats import spearmanr, mannwhitneyu, rankdata
    P = load_participants()
    zs = [onp.load(RES / f"split_half_part{k}.npz") for k in range(nparts)]
    ids = onp.concatenate([z["ids"][:int(z["done"])] for z in zs])
    grp = onp.concatenate([z["groups"][:int(z["done"])] for z in zs])
    ev = onp.concatenate([z["log_ev"][:int(z["done"])] for z in zs])
    ok = onp.concatenate([z["ok"][:int(z["done"])] for z in zs])
    names = list(zs[0]["models"])
    r1 = ev[:, names.index("I")] - ev[:, names.index("M4")]
    r2 = ev[:, names.index("C")] - ev[:, names.index("H")]
    acc2 = onp.array([(P[str(p)]["data"][63:90, 6] == P[str(p)]["data"][63:90, 7]).mean() for p in ids])
    gnum = (grp == "g1").astype(float)

    def partial_rho(x, y):
        rx, ry = rankdata(x), rankdata(y)
        res = lambda a: a - onp.polyval(onp.polyfit(gnum, a, 1), gnum)
        return spearmanr(res(rx), res(ry))

    print(f"n = {len(ids)}; fits ok {ok.mean():.3f}; outcome = accuracy on phase-2 trials 28-54 (not fitted)")
    for name, r in (("R1_half (I vs M4)", r1), ("R2_half (C vs H)", r2)):
        parts = []
        for g, label in (("g1", "BPD"), ("g2", "Controls")):
            s = grp == g
            rho, p = spearmanr(r[s], acc2[s])
            ai, ac = acc2[s & (r > 0)], acc2[s & (r < 0)]
            pm = mannwhitneyu(ai, ac).pvalue if min(len(ai), len(ac)) > 2 else onp.nan
            parts.append(f"{label}: rho {rho:+.2f} (p={p:.3f}); intent n={len(ai)} {onp.median(ai):.3f} vs "
                         f"competitiveness n={len(ac)} {onp.median(ac):.3f} (p={pm:.3f})")
        rp, pp = partial_rho(r, acc2)
        print(f"\n{name}\n  " + "\n  ".join(parts) + f"\n  group-controlled rho {rp:+.2f} (p={pp:.3f})")
        print(f"  share intent-represented: BPD {(r[grp == 'g1'] > 0).mean():.0%}, "
              f"controls {(r[grp == 'g2'] > 0).mean():.0%}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        analyze(int(sys.argv[2]))
