"""Does how a participant represents the partner relate to how accurately they predict the partner?

Context: the partner algorithm generates choices from (alpha, beta) utilities, i.e. it IS a
competitiveness agent. Controls are better fit by an intent representation (the partner helps or
harms me); BPD participants by a competitiveness representation. If imposing intent on a
non-intentional agent is miscalibrated, intent-represented participants should predict less
accurately. Barnby et al. report equal accuracy between groups.

Specified before looking:
  representation index R1 = log BF(I vs M4)  (identical models except the partner representation;
                            > 0 intent, < 0 competitiveness); R2 = log BF(C vs H) (both coupled).
  accuracy A   = share of the 54 phase-2 predictions matching the partner's actual choice;
  accuracy A2  = the same over trials 28-54 (after learning).
  Tests: Spearman(R, A) within each group and group-controlled; A for R1 > 0 vs R1 < 0
  (Mann-Whitney), within group; A by group. Exploratory: Spearman(R1, HI rating).

Usage: python -I representation_accuracy.py
"""
import sys
from pathlib import Path

import numpy as onp
from scipy.stats import spearmanr, mannwhitneyu, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants  # noqa: E402
from pxp_report import load  # noqa: E402


def partial(x, y, g):
    rx, ry = rankdata(x), rankdata(y)
    res = lambda a: a - onp.polyval(onp.polyfit(g, a, 1), g)
    return spearmanr(res(rx), res(ry))


if __name__ == "__main__":
    P = load_participants()
    rows = []
    for g, label in (("g1", "BPD"), ("g2", "Controls")):
        ev = load(g)
        ids = onp.load(Path(__file__).resolve().parent / "results" / f"laplace_{g}.npz")["ids"]
        for i, pid in enumerate(ids):
            d = P[str(pid)]["data"][36:90]
            hit = d[:, 6] == d[:, 7]
            rows.append(dict(group=label, R1=ev["I"][i] - ev["M4"][i], R2=ev["C"][i] - ev["H"][i],
                             A=hit.mean(), A2=hit[27:].mean(), HI=float(P[str(pid)]["meta"]["HI"])))
    col = lambda k: onp.array([r[k] for r in rows], float)
    grp = onp.array([r["group"] for r in rows])
    gnum = (grp == "BPD").astype(float)
    print(f"n = {len(rows)}")
    for acc in ("A", "A2"):
        a = col(acc)
        u = mannwhitneyu(a[grp == "BPD"], a[grp == "Controls"])
        print(f"\nAccuracy {acc}: BPD median {onp.median(a[grp == 'BPD']):.3f} vs controls "
              f"{onp.median(a[grp == 'Controls']):.3f} (Mann-Whitney p = {u.pvalue:.3f})")
        for rep in ("R1", "R2"):
            r = col(rep)
            out = []
            for gl in ("BPD", "Controls"):
                s = grp == gl
                rho, p = spearmanr(r[s], a[s]); out.append(f"{gl} rho {rho:+.2f} (p={p:.3f})")
            rp, pp = partial(r, a, gnum)
            print(f"  {rep} vs {acc}: " + "; ".join(out) + f"; group-controlled {rp:+.2f} (p={pp:.3f})")
        r1 = col("R1")
        for gl in ("BPD", "Controls"):
            s = grp == gl
            ai, ac = a[s & (r1 > 0)], a[s & (r1 < 0)]
            p = mannwhitneyu(ai, ac).pvalue if min(len(ai), len(ac)) > 2 else onp.nan
            print(f"  {gl:8s}: intent-represented (R1>0, n={len(ai)}) median {onp.median(ai):.3f} vs "
                  f"competitiveness (n={len(ac)}) {onp.median(ac):.3f} (p = {p:.3f})")
    print("\nShare intent-represented (R1 > 0): BPD "
          f"{(col('R1')[grp == 'BPD'] > 0).mean():.0%}, controls {(col('R1')[grp == 'Controls'] > 0).mean():.0%}")
    hi, r1 = col("HI"), col("R1")
    rho, p = spearmanr(r1, hi)
    rp, pp = partial(r1, hi, gnum)
    print(f"Exploratory: R1 vs HI rating: rho {rho:+.2f} (p={p:.3f}); group-controlled {rp:+.2f} (p={pp:.3f})")
