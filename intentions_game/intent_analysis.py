"""Pre-specified analysis of the intent model I (written before any intent results were seen).

1. Model comparison. I vs. M4 (same 8 parameters and self part; partner represented by
   benevolence/harm instead of competitiveness): per-participant log Bayes factors and
   two-model random-effects BMS, per group. Also I within the full model set.
2. Out-of-sample attributions. From each participant's fitted I parameters, the posterior after
   phase 2 gives (fixed links, no fitted rating parameters):
     PRIMARY    HI rating  <->  E[max(-rho, 0)]   (expected harm weight)
                SI rating  <->  E[omega]          (expected self-interest weight)
     SECONDARY  HI rating  <->  P(rho < 0)
   Spearman correlations across all 103 participants. Rival, cue-based predictors computed from
   the partner's phase-2 choices alone:
     harm cue  = share of choices giving the participant fewer points;
     self cue  = share of choices giving the partner more points.
   Incremental value: partial Spearman of the model measure with the rating, controlling for the cue.
3. Benevolence prior. The prior mean of rho (natural scale) per participant:
     PRIMARY    BPD vs. controls (Mann-Whitney; predicted lower in BPD);
                correlation with ETMCQ mistrust (predicted negative) and RGPTS-B persecution
                (predicted negative).
     SECONDARY  ETMCQ trust, credulity; CTQ.
   Benjamini-Hochberg FDR within each family (primary, secondary).

Usage: python -I intent_analysis.py
"""
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp
from scipy.stats import spearmanr, mannwhitneyu, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, FIT_ORDER  # noqa: E402
from bms import rfx_bms, RES  # noqa: E402
import memo_models as mm  # noqa: E402
import intent  # noqa: E402


def bh(p):
    p = onp.asarray(p, float)
    order = onp.argsort(p)
    adj = onp.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * m / rank)
        adj[i] = prev
    return adj


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - onp.polyval(onp.polyfit(rz, a, 1), rz)
    return spearmanr(res(rx), res(ry))


def attributions(params, data):
    q = mm.unpack("I", np.asarray(params))
    prior = mm.gauss_grid(q["alpha_par"], q["alpha_v_ref"], q["beta_par"], q["beta_v_ref"])
    pay2 = np.asarray(data[36:90, 2:6] / 10.0)
    L = intent.intent_choice(pay2)
    lik = L[np.arange(54), :, :, np.asarray(data[36:90, 7].astype(int) - 1)]
    logL = np.log(lik).sum(axis=0)
    harm, e_omega, p_neg = intent.attribute_intent(prior, np.exp(logL - logL.max()))
    return float(harm), float(e_omega), float(p_neg), float(q["beta_par"])


def cues(data):
    d = data[36:90]
    chosen_o = onp.where(d[:, 7] == 1, d[:, 3], d[:, 5])
    other_o = onp.where(d[:, 7] == 1, d[:, 5], d[:, 3])
    chosen_s = onp.where(d[:, 7] == 1, d[:, 2], d[:, 4])
    other_s = onp.where(d[:, 7] == 1, d[:, 4], d[:, 2])
    return onp.mean(chosen_o < other_o), onp.mean(chosen_s > other_s)


if __name__ == "__main__":
    P = load_participants()
    rows = []
    print("=== 1. Model comparison: I vs. M4")
    for g, label in (("g1", "BPD"), ("g2", "Controls")):
        z5 = onp.load(RES / f"laplace_{g}.npz")
        zi = onp.load(RES / f"laplace_{g}_I.npz")
        assert int(zi["done"]) == zi["log_ev"].shape[0], f"{g}: I fit incomplete"
        assert (z5["ids"] == zi["ids"]).all()
        ev_i, ev_m4 = zi["log_ev"][:, 0], z5["log_ev"][:, FIT_ORDER.index("M4")]
        d = ev_i - ev_m4
        freq, xp, _ = rfx_bms(onp.column_stack([ev_i, ev_m4]))
        ev_all = onp.column_stack([z5["log_ev"], ev_i])
        f_all, xp_all, _ = rfx_bms(ev_all)
        print(f"  {label:8s} median log BF(I - M4) = {onp.median(d):+.2f}; I better for {(d > 0).mean():.0%} "
              f"(|log BF| > 1: I {(d > 1).mean():.0%}, M4 {(d < -1).mean():.0%}); RFX I vs M4: freq {freq[0]:.2f}, "
              f"xp {xp[0]:.3f}; fits ok {zi['ok'].mean():.2f}")
        print(f"           6-model RFX ({', '.join(FIT_ORDER + ['I'])}): freq "
              + " ".join(f"{x:.2f}" for x in f_all) + " | xp " + " ".join(f"{x:.2f}" for x in xp_all))
        for i, pid in enumerate(zi["ids"]):
            harm, e_om, p_neg, mu_rho = attributions(zi["params_I"][i], P[pid]["data"])
            # P(rho < 0) equals 1 to within 1e-6 for more than half the sample; below that, the order of
            # those values is floating-point noise that differs across platforms, so they are treated as ties.
            p_neg = round(p_neg, 6)
            hc, sc = cues(P[pid]["data"])
            meta = P[pid]["meta"]
            num = lambda k: float(meta[k]) if meta[k] not in ("", "NA") else onp.nan
            rows.append(dict(group=label, harm=harm, e_omega=e_om, p_neg=p_neg, mu_rho=mu_rho, harm_cue=hc,
                             self_cue=sc, HI=num("HI"), SI=num("SI"), mistrust=num("A_ETS_Mistrust"),
                             trust=num("A_ETS_Trust"), credulity=num("A_ETS_Credulity"),
                             persecution=num("RGPTSB"), ctq=num("CTQtot")))
    col = lambda k: onp.array([r[k] for r in rows], float)

    print("\n=== 2. Out-of-sample attributions (n = %d)" % len(rows))
    for meas, rating, cue, tag in (("harm", "HI", "harm_cue", "PRIMARY"), ("e_omega", "SI", "self_cue", "PRIMARY"),
                                   ("p_neg", "HI", "harm_cue", "SECONDARY")):
        x, y, c = col(meas), col(rating), col(cue)
        ok = ~onp.isnan(x + y + c)
        rho_m, p_m = spearmanr(x[ok], y[ok])
        rho_c, p_c = spearmanr(c[ok], y[ok])
        pr, pp = partial_spearman(x[ok], y[ok], c[ok])
        print(f"  {tag:9s} {rating} ~ {meas:8s}: rho = {rho_m:+.2f} (p = {p_m:.3f}) | cue {cue}: rho = {rho_c:+.2f} "
              f"(p = {p_c:.3f}) | model beyond cue: partial rho = {pr:+.2f} (p = {pp:.3f})")

    print("\n=== 3. Benevolence prior (prior mean of rho)")
    mu, grp = col("mu_rho"), onp.array([r["group"] for r in rows])
    u = mannwhitneyu(mu[grp == "BPD"], mu[grp == "Controls"])
    tests_primary = [("BPD vs controls (Mann-Whitney)", u.pvalue,
                      f"median BPD {onp.median(mu[grp == 'BPD']):+.2f} vs controls {onp.median(mu[grp == 'Controls']):+.2f}")]
    for k in ("mistrust", "persecution"):
        y = col(k); ok = ~onp.isnan(y)
        r, p = spearmanr(mu[ok], y[ok]); tests_primary.append((f"mu_rho ~ {k}", p, f"rho = {r:+.2f} (n = {ok.sum()})"))
    tests_secondary = []
    for k in ("trust", "credulity", "ctq"):
        y = col(k); ok = ~onp.isnan(y)
        r, p = spearmanr(mu[ok], y[ok]); tests_secondary.append((f"mu_rho ~ {k}", p, f"rho = {r:+.2f} (n = {ok.sum()})"))
    for fam, tests in (("PRIMARY", tests_primary), ("SECONDARY", tests_secondary)):
        adj = bh([t[1] for t in tests])
        for (name, p, desc), q in zip(tests, adj):
            print(f"  {fam:9s} {name:32s} {desc:40s} p = {p:.3f}, FDR q = {q:.3f}")
