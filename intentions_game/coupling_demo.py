"""Demonstrate that the population spread tau controls both self-insertion and contagion.

For one participant's payoffs and partner choices, with introspection fixed (a, b, v_a, v_b),
vary tau and report, on the beta (competitiveness) dimension:
  insertion: how far the partner prior is pulled toward the participant's own mean, as the
             precision of that prior relative to a flat one (1 = flat; larger = stronger pull),
             and its mean;
  contagion: the fraction of the way the participant's own mean moves toward the partner's
             posterior mean between phase 1 and phase 3.
Usage: python -I coupling_demo.py
"""
import sys
from pathlib import Path

import jax.numpy as np
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, matlab_order  # noqa: E402
import hier_core as core  # noqa: E402
import memo_models as mm  # noqa: E402

P = load_participants()
d = P[matlab_order(P, "g2")[0]]["data"]
pay2 = np.asarray(d[36:90, 2:6] / 10.0)
partner = np.asarray(d[36:90, 7].astype(int) - 1)
L = mm.partner_choice(pay2)[np.arange(54), :, :, partner]          # [t, ia, ib]
logL = np.log(L).sum(axis=0)
Lall = np.exp(logL - logL.max())

a, b, v_a, v_b = 15.0, -8.0, 3.0, 4.0
flat_sd = float(np.std(mm.BG))
mean_b = lambda g: float((g.sum(axis=0) * mm.BG).sum())
sd_b = lambda g: float(np.sqrt((g.sum(axis=0) * (mm.BG - mean_b(g)) ** 2).sum()))
print(f"own beta (introspection) = {b}; partner's posterior beta under a flat prior = "
      f"{mean_b(Lall / Lall.sum()):.2f}")
print(f"{'tau':>6} | {'partner prior mean':>18} {'prior precision vs flat':>24} | {'own mean ph1 -> ph3':>20} {'contagion fraction':>19}")
for tau in [0.5, 1, 2, 4, 8, 16, 64]:
    Ja, Jb, Wa, Wb, b1, pi = core.beliefs(mm.AG, mm.BG, (a, b, v_a, v_b, tau, tau))
    b3 = core.contagion_belief(Ja, Jb, Wa, Wb, Lall)
    post = pi * Lall; post = post / post.sum()
    m1, m3, mp = mean_b(b1), mean_b(b3), mean_b(post)
    frac = (m3 - m1) / (mp - m1) if abs(mp - m1) > 1e-9 else float("nan")
    print(f"{tau:6.1f} | {mean_b(pi):18.2f} {(flat_sd / sd_b(pi)) ** 2:24.1f} | "
          f"{m1:8.2f} -> {m3:8.2f} {frac:19.2f}")
