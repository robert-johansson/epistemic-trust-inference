"""Check the memo port against the NumPy reference (itself checked against MATLAB).

For every participant (50 BPD, 53 controls) and every model (Beta, M1-M4), at that
participant's HBI-fitted parameters for that model: max |P(option 1) difference| over
126 trials, and the log-likelihood difference. Also times the jitted memo likelihood.

Usage: python -I validate_memo.py
"""
import sys
import time
from functools import partial
from pathlib import Path

import jax
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, matlab_order, load_hbi, FIT_ORDER  # noqa: E402
import reference as ref  # noqa: E402
import memo_models as mm  # noqa: E402

if __name__ == "__main__":
    P = load_participants()
    netp_jit = {m: jax.jit(partial(mm.model_netp, m)) for m in FIT_ORDER}
    worst = {}
    for g in ("g1", "g2"):
        order = matlab_order(P, g)
        hbi = load_hbi(g)
        for m in FIT_ORDER:
            t0 = time.time()
            e_p, e_ll = 0.0, 0.0
            for i, pid in enumerate(order):
                d = P[pid]["data"]
                prm = hbi["parameters"][m][i]
                ours, _ = netp_jit[m](prm, d)
                ours = onp.asarray(ours)
                r = ref.run(m, prm, d)
                e_p = max(e_p, onp.abs(ours - r["netp"]).max())
                ch = d[:, 6]
                ll = onp.sum(onp.where(ch == 1, onp.log(ours), onp.log1p(-ours)))
                e_ll = max(e_ll, abs(ll - (r["F"] - ref.EPS)))
            worst[(g, m)] = (e_p, e_ll)
            print(f"{g} {m:4s}: max |netp diff| {e_p:.2e}   max |loglik diff| {e_ll:.2e}   "
                  f"({time.time() - t0:.1f}s incl. reference)")
    # timing of the memo likelihood alone
    d = P[matlab_order(P, "g2")[0]]["data"]
    prm = load_hbi("g2")["parameters"]["M1"][0]
    ll = jax.jit(partial(mm.loglik, "M1"))
    ll(prm, d).block_until_ready()
    t0 = time.time()
    for _ in range(10):
        ll(prm, d).block_until_ready()
    print(f"\nmemo M1 log-likelihood, one participant: {(time.time() - t0) / 10 * 1000:.0f} ms per call (jitted, CPU)")
    print("ALL WITHIN 1e-9:", all(v[0] < 1e-9 and v[1] < 1e-7 for v in worst.values()))
