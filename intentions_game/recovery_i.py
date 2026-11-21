"""Parameter recovery for the intent model I (intent.py), focused on the benevolence prior.

Ground truth: each participant's fitted I parameters (results/laplace_g*_I.npz). For each, simulate
126 responses on that participant's real payoffs and partner choices, then refit I blind (starts: the
prior mean and random draws, never the truth), with the same MAP + Laplace procedure as fit.py.

The key question is whether mu_rho (prior mean of the partner's weight on the participant's points:
the benevolence prior, i.e. the mistrust parameter) is identified at all, given that 54 phase-2
trials may swamp the prior; and whether it is recovered within each group.

Usage: python -I recovery_i.py run PART NPARTS   -> results/recovery_I_partPART.npz (checkpointed)
       python -I recovery_i.py analyze NPARTS    -> recovery statistics (merges parts)
"""
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants  # noqa: E402
import memo_models as mm  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
NAMES = ["alpha (self)", "beta (self)", "log alpha_v", "log beta_v",
         "mu_omega", "mu_rho", "log sd_omega", "log sd_rho"]


def truth_sets():
    P = load_participants()
    rows = []
    for g in ("g1", "g2"):
        z = onp.load(RES / f"laplace_{g}_I.npz")
        for pid, prm in zip(z["ids"], z["params_I"]):
            rows.append((str(pid), g, onp.asarray(prm, float)))
    return P, rows


def natural(x):
    """Raw -> interpretable scale: alpha, mu_omega as 30*sigmoid; beta, mu_rho as is; sds as log."""
    s = lambda v: 30 / (1 + onp.exp(-v))
    return onp.column_stack([s(x[:, 0]), x[:, 1], x[:, 2], x[:, 3], s(x[:, 4]), x[:, 5], x[:, 6], x[:, 7]])


def run(part, nparts):
    from fit import fit_one  # noqa: E402
    P, rows = truth_sets()
    rows = rows[part::nparts]
    rng = onp.random.default_rng(100 + part)
    netp = jax.jit(partial(mm.model_netp, "I"))
    n = len(rows)
    true = onp.array([r[2] for r in rows])
    rec = onp.full_like(true, onp.nan)
    ok = onp.zeros(n, bool)
    out = RES / f"recovery_I_part{part}.npz"
    t0 = time.time()
    for i, (pid, g, theta) in enumerate(rows):
        p, _ = netp(np.asarray(theta), np.asarray(P[pid]["data"]))
        sim = onp.array(P[pid]["data"], dtype=float)
        sim[:, 6] = onp.where(rng.random(126) < onp.asarray(p), 1.0, 2.0)
        x, _, _, good, _ = fit_one("I", np.asarray(sim), [onp.zeros(8), rng.normal(0, 1.0, 8)], rng)
        rec[i], ok[i] = x, good
        print(f"part {part}: {i + 1}/{n} {g} {pid}: ok={good} ({time.time() - t0:.0f}s)", flush=True)
        onp.savez(out, ids=onp.array([r[0] for r in rows]), groups=onp.array([r[1] for r in rows]),
                  true=true, recovered=rec, ok=ok, done=i + 1)


def analyze(nparts):
    from scipy.stats import pearsonr, spearmanr, mannwhitneyu
    zs = [onp.load(RES / f"recovery_I_part{k}.npz") for k in range(nparts)]
    t = onp.concatenate([z["true"][:int(z["done"])] for z in zs])
    r = onp.concatenate([z["recovered"][:int(z["done"])] for z in zs])
    ok = onp.concatenate([z["ok"][:int(z["done"])] for z in zs])
    grp = onp.concatenate([z["groups"][:int(z["done"])] for z in zs])
    tn, rn = natural(t), natural(r)
    print(f"n = {len(t)}; converged with positive-definite Hessian: {ok.mean():.3f}\n")
    print(f"{'parameter':14s} {'Pearson':>8s} {'Spearman':>9s} {'bias':>7s} {'RMSE':>7s} {'true SD':>8s}"
          f" {'Spearman BPD':>13s} {'Spearman CON':>13s}")
    for j, name in enumerate(NAMES):
        b, c = grp == "g1", grp == "g2"
        print(f"{name:14s} {pearsonr(tn[:, j], rn[:, j])[0]:8.2f} {spearmanr(tn[:, j], rn[:, j])[0]:9.2f} "
              f"{onp.mean(rn[:, j] - tn[:, j]):7.2f} {onp.sqrt(onp.mean((rn[:, j] - tn[:, j]) ** 2)):7.2f} "
              f"{onp.std(tn[:, j]):8.2f} {spearmanr(tn[b, j], rn[b, j])[0]:13.2f} {spearmanr(tn[c, j], rn[c, j])[0]:13.2f}")
    j = NAMES.index("mu_rho")
    ut = mannwhitneyu(tn[grp == "g1", j], tn[grp == "g2", j]).pvalue
    ur = mannwhitneyu(rn[grp == "g1", j], rn[grp == "g2", j]).pvalue
    print(f"\nmu_rho group medians: true BPD {onp.median(tn[grp == 'g1', j]):+.2f} vs CON "
          f"{onp.median(tn[grp == 'g2', j]):+.2f} (p = {ut:.3f}); recovered BPD {onp.median(rn[grp == 'g1', j]):+.2f} "
          f"vs CON {onp.median(rn[grp == 'g2', j]):+.2f} (p = {ur:.3f})")
    err = rn - tn
    C = onp.corrcoef(err.T)
    print("\nError correlations of mu_rho with the other parameters:",
          ", ".join(f"{NAMES[k]} {C[j, k]:+.2f}" for k in range(len(NAMES)) if k != j))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        analyze(int(sys.argv[2]))
