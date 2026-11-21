"""Data for Fig. 3: one coupling parameter (tau) controls both self-insertion and contagion in model H.

Same computation as intentions_game/coupling_demo.py (one participant's phase-2 payoffs and partner
choices, introspection fixed at a = 15, b = -8, v_a = 3, v_b = 4; tau_a = tau_b = tau), on a dense
log-spaced tau grid. Outputs, on the relative-reward (beta) dimension:
  insertion = precision of the partner prior relative to a flat prior (1 = flat);
  contagion = fraction of the way one's own mean moves toward the partner's posterior mean.

Usage (from the project root):
  uv run python -I figures/scripts/export_fig3_coupling.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "intentions_game"))

import numpy as onp  # noqa: E402
import jax.numpy as np  # noqa: E402
import memo_models as mm  # noqa: E402
import hier_core as core  # noqa: E402
from data import load_participants, matlab_order  # noqa: E402

P = load_participants()
d = P[matlab_order(P, "g2")[0]]["data"]
pay2 = np.asarray(d[36:90, 2:6] / 10.0)
partner = np.asarray(d[36:90, 7].astype(int) - 1)
L = mm.partner_choice(pay2)[np.arange(54), :, :, partner]
logL = np.log(L).sum(axis=0)
Lall = np.exp(logL - logL.max())

a, b, v_a, v_b = 15.0, -8.0, 3.0, 4.0
flat_sd = float(np.std(mm.BG))
mean_b = lambda g: float((g.sum(axis=0) * mm.BG).sum())
sd_b = lambda g: float(np.sqrt((g.sum(axis=0) * (mm.BG - mean_b(g)) ** 2).sum()))

taus = 2.0 ** (onp.arange(-4, 25) / 4.0)          # 0.5 ... 64 in quarter-octave steps
rows = []
for tau in taus:
    Ja, Jb, Wa, Wb, b1, pi = core.beliefs(mm.AG, mm.BG, (a, b, v_a, v_b, float(tau), float(tau)))
    b3 = core.contagion_belief(Ja, Jb, Wa, Wb, Lall)
    post = pi * Lall
    post = post / post.sum()
    m1, m3, mp = mean_b(b1), mean_b(b3), mean_b(post)
    rows.append(dict(tau=float(tau), insertion=(flat_sd / sd_b(pi)) ** 2,
                     contagion=(m3 - m1) / (mp - m1), prior_mean=mean_b(pi)))

out = ROOT / "figures" / "data" / "fig3_coupling.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: f"{v:.6g}" for k, v in r.items()})
print(f"wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
for r in rows[::4]:
    print(f"  tau {r['tau']:7.3f}  insertion {r['insertion']:6.2f}  contagion {r['contagion']:5.3f}")
