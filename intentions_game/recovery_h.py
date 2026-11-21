"""Parameter recovery for the hierarchical self-other model H.

Ground truth: for each of the 103 participants, H parameters derived from the authors' HBI
estimates for M1 (same introspection parameters; population spread tau = v_ref / sqrt(2)).
For each set, simulate all 126 responses on that participant's real payoffs and partner
choices (the partner's choices are exogenous in the model), then refit H blind: starts are
the prior mean (zeros) and random draws, never the true values. Same MAP + Laplace procedure
as fit.py.

Usage: python -I recovery_h.py           -> results/recovery_H.npz (checkpointed)
       python -I recovery_h.py analyze   -> recovery statistics
"""
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, matlab_order, load_hbi  # noqa: E402
import memo_models as mm  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "recovery_H.npz"
NAMES = ["a (alpha mean)", "b (beta mean)", "log v_a", "log v_b", "log tau_a", "log tau_b"]


def truth_sets():
    P = load_participants()
    rows = []
    for g in ("g1", "g2"):
        hbi = load_hbi(g)
        for i, pid in enumerate(matlab_order(P, g)):
            m1 = onp.asarray(hbi["parameters"]["M1"][i], float)
            rows.append((pid, g, onp.concatenate([m1[:4], m1[4:6] - 0.5 * onp.log(2.0)])))
    return P, rows


def simulate(params, data, rng):
    netp, _ = jax.jit(partial(mm.model_netp, "H"))(np.asarray(params), np.asarray(data))
    y = onp.where(rng.random(126) < onp.asarray(netp), 1.0, 2.0)
    sim = onp.array(data, dtype=float)
    sim[:, 6] = y
    return sim


def run():
    from fit import fit_one   # noqa: E402  (compiles the fitting functions)
    P, rows = truth_sets()
    rng = onp.random.default_rng(2026)
    n = len(rows)
    true = onp.array([r[2] for r in rows])
    rec = onp.full_like(true, onp.nan)
    ok = onp.zeros(n, bool)
    t0 = time.time()
    for i, (pid, g, theta) in enumerate(rows):
        sim = simulate(theta, P[pid]["data"], rng)
        starts = [onp.zeros(6), rng.normal(0, 1.0, 6)]
        x, _, _, good, _ = fit_one("H", np.asarray(sim), starts, rng)
        rec[i], ok[i] = x, good
        print(f"{i + 1}/{n} {g} {pid}: ok={good} ({time.time() - t0:.0f}s)", flush=True)
        onp.savez(OUT, ids=onp.array([r[0] for r in rows]), groups=onp.array([r[1] for r in rows]),
                  true=true, recovered=rec, ok=ok, done=i + 1)


def analyze():
    from scipy.stats import pearsonr, spearmanr
    z = onp.load(OUT)
    done = int(z["done"])
    t, r, ok = z["true"][:done], z["recovered"][:done], z["ok"][:done]
    natural = lambda x: onp.column_stack([30 / (1 + onp.exp(-x[:, 0])), x[:, 1], x[:, 2:]])
    tn, rn = natural(t), natural(r)
    print(f"n = {done}; converged with positive-definite Hessian: {ok.mean():.3f}\n")
    print(f"{'parameter':18s} {'Pearson r':>9s} {'Spearman':>9s} {'bias':>8s} {'RMSE':>8s} {'true SD':>8s}")
    for j, name in enumerate(NAMES):
        print(f"{name:18s} {pearsonr(tn[:, j], rn[:, j])[0]:9.2f} {spearmanr(tn[:, j], rn[:, j])[0]:9.2f} "
              f"{onp.mean(rn[:, j] - tn[:, j]):8.2f} {onp.sqrt(onp.mean((rn[:, j] - tn[:, j]) ** 2)):8.2f} "
              f"{onp.std(tn[:, j]):8.2f}")
    print("\nTrade-offs: correlation of recovery errors (recovered - true) between parameters")
    err = rn - tn
    C = onp.corrcoef(err.T)
    print(" " * 12 + "".join(f"{n.split()[0] + ' ' + n.split()[-1][:6]:>12s}" for n in NAMES))
    for j, name in enumerate(NAMES):
        print(f"{name[:11]:12s}" + "".join(f"{C[j, k]:12.2f}" for k in range(len(NAMES))))
    print("\nCross-recovery: Spearman(true parameter, recovered parameter)")
    print(" " * 12 + "".join(f"{('rec ' + n.split()[-1][:7]):>12s}" for n in NAMES))
    for j, name in enumerate(NAMES):
        print(f"{('true ' + name.split()[-1][:6]):12s}" + "".join(
            f"{spearmanr(tn[:, j], rn[:, k])[0]:12.2f}" for k in range(len(NAMES))))
    # the theoretically central quantity: coupling, collapsed over dimensions
    ct, cr = tn[:, 4:6].mean(axis=1), rn[:, 4:6].mean(axis=1)
    print(f"\nMean log tau (coupling) recovery: Spearman = {spearmanr(ct, cr)[0]:.2f}; "
          f"Pearson = {pearsonr(ct, cr)[0]:.2f}")
    for g, label in (("g1", "BPD-derived"), ("g2", "control-derived")):
        s = z["groups"][:done] == g
        if s.sum() > 3:
            print(f"  {label:16s} (n = {s.sum()}): Spearman = {spearmanr(ct[s], cr[s])[0]:.2f}")


if __name__ == "__main__":
    analyze() if len(sys.argv) > 1 and sys.argv[1] == "analyze" else run()
