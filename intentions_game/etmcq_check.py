"""ETMCQ x belief updating: the preprint's result, and a fair re-test with ONE model for everyone.

Barnby et al.'s preprint (Barnby_etal_SocGen_Preprint_Combined.pdf in the repository):
"The credulity (ETMCQ) of participants was negatively associated with belief updating
(rho[d alpha_par] = -0.25, p = 0.013; rho[d beta_par] = -0.35, p < 0.001)". The final eLife
version does not report it.

Belief updating, as in BPD_CON_Analysis.R: |E_posterior - E_prior| of the partner's alpha and beta
over phase 2. The preprint computed it from each group's winning model (BPD: M4, controls: M1),
so the pooled correlation mixes group, model and individual differences.

Modes:
  python -I etmcq_check.py                      group winners at HBI parameters (preprint replication)
  python -I etmcq_check.py MODEL SOURCE         one model for everyone; SOURCE = hbi | laplace
                                                (MODEL in M1, M2, M3, M4, H; H only with laplace)
Reports pooled Spearman correlations, within-group correlations, and partial correlations
controlling for group (rank-residualised).
"""
import sys
from functools import partial
from pathlib import Path

import jax
import numpy as onp
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, matlab_order, load_hbi  # noqa: E402
import memo_models as mm  # noqa: E402

WINNER = {"g1": "M4", "g2": "M1"}
RES = Path(__file__).resolve().parent / "results"


def partner_prior(model, parms):
    q = mm.unpack(model, jax.numpy.asarray(parms))
    if model == "H":
        import hier_core
        return hier_core.beliefs(mm.AG, mm.BG, (q["a"], q["b"], q["v_a"], q["v_b"], q["tau_a"], q["tau_b"]))[5]
    return mm.gauss_grid(q["alpha_par"], q["alpha_v_ref"], q["beta_par"], q["beta_v_ref"])


def shifts(model, parms, data):
    _, post = jax.jit(partial(mm.model_netp, model))(parms, data)
    pri = partner_prior(model, parms)
    ea = lambda g: float((onp.asarray(g).sum(axis=1) * onp.asarray(mm.AG)).sum())
    eb = lambda g: float((onp.asarray(g).sum(axis=0) * onp.asarray(mm.BG)).sum())
    return abs(ea(post) - ea(pri)), abs(eb(post) - eb(pri))


def params_for(group, model, source):
    if source == "hbi":
        return load_hbi(group)["parameters"][model]
    f = RES / (f"laplace_{group}_H.npz" if model == "H" else f"laplace_{group}.npz")
    return onp.load(f)[f"params_{model}"]


def partial_rho(x, y, g):
    rx, ry = rankdata(x), rankdata(y)
    res = lambda a: a - onp.polyval(onp.polyfit(g, a, 1), g)
    return spearmanr(res(rx), res(ry))


if __name__ == "__main__":
    common = sys.argv[1] if len(sys.argv) > 1 else None
    source = sys.argv[2] if len(sys.argv) > 2 else "hbi"
    P = load_participants()
    rows = []
    for g in ("g1", "g2"):
        model = common or WINNER[g]
        prm = params_for(g, model, source)
        for i, pid in enumerate(matlab_order(P, g)):
            da, db = shifts(model, prm[i], P[pid]["data"])
            meta = P[pid]["meta"]
            rows.append((g, da, db, *[float(meta[k]) if meta[k] not in ("", "NA") else onp.nan
                                      for k in ("A_ETS_Mistrust", "A_ETS_Trust", "A_ETS_Credulity")]))
    grp = onp.array([r[0] == "g1" for r in rows], float)
    arr = onp.array([r[1:] for r in rows], dtype=float)
    keep = ~onp.isnan(arr).any(axis=1)
    arr, grp = arr[keep], grp[keep]
    label = f"one model for everyone: {common} ({source} parameters)" if common else \
            "group winners (BPD M4, controls M1) at HBI parameters (the preprint's approach)"
    print(f"{label}; n = {keep.sum()}")
    for j, name in enumerate(("Mistrust", "Trust", "Credulity")):
        for k, sh in enumerate(("|d alpha_par|", "|d beta_par|")):
            x, y = arr[:, 2 + j], arr[:, k]
            r, p = spearmanr(x, y)
            rb, pb = spearmanr(x[grp == 1], y[grp == 1])
            rc, pc = spearmanr(x[grp == 0], y[grp == 0])
            rp, pp = partial_rho(x, y, grp)
            print(f"  {name:9s} vs {sh:13s} pooled {r:+.2f} (p={p:.3f}) | BPD {rb:+.2f} (p={pb:.3f}) | "
                  f"controls {rc:+.2f} (p={pc:.3f}) | group-controlled {rp:+.2f} (p={pp:.3f})")
