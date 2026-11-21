"""Data for Fig. 6: simulated epistemic stances in the BART-ET advisor model (Section 5.3).

Re-runs the simulation of advice_taking/bart_et_advisor.py exactly (same schedule, seeds, profile order
and key splitting), but saves every balloon instead of checkpoints. Also exports the hot-stove
update and the expected deviation from advice at fixed trust. Every task feature the paper does not
report is an assumption, as marked in the sketch.

Usage (from the project root):
  uv run python -I figures/scripts/export_fig6_bart.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "advice_taking"))

import jax  # noqa: E402
import jax.numpy as np  # noqa: E402
import numpy as onp  # noqa: E402
import bart_et_advisor as sk  # noqa: E402

OUT = ROOT / "figures" / "data"


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {name} ({len(rows)} rows)")


# ---- schedule and priors: identical to the sketch's __main__ ----
acc = [1] * 3 + [0] * 3 + [1] * 5 + [1] * 19
for i in [14, 19, 23, 27]:                       # ASSUMED positions of the 4 late violations
    acc[i] = 0
rng = onp.random.default_rng(0)
advice = rng.integers(4, 8, size=30)
T_true = onp.where(onp.array(acc) == 1, onp.minimum(advice + 2, 10), advice)
t = onp.arange(11)
thr = onp.where(t >= 1, onp.exp(-t / 4.0), 0.0)
thr_prior = np.array(thr / thr.sum())
BETA, LAM = 1.5, 1.0
profiles = {                                      # same order as the sketch (key splitting depends on it)
    "trusting": sk.beta_grid(0.70, 4),
    "mistrusting": sk.beta_grid(0.30, 4),
    "credulous": sk.beta_grid(0.85, 40),
    "petrified": sk.beta_grid(0.30, 40),
}
cp = jax.jit(jax.vmap(lambda rb: sk.choose_pumps(rb, thr_prior, BETA, LAM)))
ur = jax.jit(jax.vmap(lambda rb: sk.update_reliability(rb, thr_prior, LAM)))

N = 200
key = jax.random.PRNGKey(1)
belief = {}
deviation = {}
for name, prior in profiles.items():
    rb = np.tile(prior, (N, 1))
    mean_rel, dev = [float(sk.REL @ prior)], []
    for k in range(30):
        a = int(advice[k])
        probs = cp(rb)[:, a, :]
        key, sub = jax.random.split(key)
        p = jax.random.categorical(sub, np.log(probs + 1e-30), axis=-1)
        o = (p >= T_true[k]).astype(np.int32)
        rb = ur(rb)[np.arange(N), a, p, o, :]
        dev.append(float(np.abs(p - a).mean()))
        mean_rel.append(float((rb @ sk.REL).mean()))
    belief[name], deviation[name] = mean_rel, dev
    print(f"  {name:12s} score {onp.mean(dev):.2f}  belief after 0,3,6,11,30: "
          + ", ".join(f"{mean_rel[i]:.2f}" for i in (0, 3, 6, 11, 30)))

names = list(profiles)
write("fig6_beliefs.csv", ["balloon"] + names,
      [[k] + [f"{belief[n][k]:.5f}" for n in names] for k in range(31)])
write("fig6_schedule.csv", ["balloon", "accurate", "advice_bin", "assumed_position"],
      [[k + 1, acc[k], int(advice[k]), int(k in (14, 19, 23, 27))] for k in range(30)])

# ---- hot stove: belief after advice 6 and a balloon that did not burst, by pumps entered ----
flat = np.ones(5) / 5
U = sk.update_reliability(flat, thr_prior, LAM)
rows = []
for p in range(0, 11):
    post = U[6, p, 0]
    if float(post.sum()) < 0.5:          # a safe balloon is impossible here (p = 10): no posterior
        continue
    rows.append([p, f"{float(post @ sk.REL):.5f}"])
write("fig6_hotstove.csv", ["pumps", "belief"], rows)
print("  hot stove:", ", ".join(f"p={r[0]}: {float(r[1]):.2f}" for r in rows))

# ---- expected deviation from advice at fixed trust ----
rows = []
for a in range(2, 9):
    row = [a]
    for rb in (np.array([0, 0, 0, 0, 1.0]), np.array([1.0, 0, 0, 0, 0])):
        C = sk.choose_pumps(rb, thr_prior, BETA, LAM)
        row.append(f"{float(C[a] @ np.abs(sk.Level - a)):.5f}")
    rows.append(row)
write("fig6_deviation.csv", ["advice", "certain_reliable", "certain_unreliable"], rows)
print("  deviation reliable:", " ".join(f"{float(r[1]):.2f}" for r in rows),
      "| unreliable:", " ".join(f"{float(r[2]):.2f}" for r in rows))
