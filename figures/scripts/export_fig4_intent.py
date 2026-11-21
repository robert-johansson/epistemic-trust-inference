"""Data for Fig. 4: inferred intent predicts explicit attributions (Section 4.5).

Per participant (n = 103), using intentions_game/intent_analysis.py's own functions:
  harm      posterior expected harm weight E[max(-rho, 0)] under model I (fitted parameters)
  harm_cue  share of the partner's phase-2 choices that gave the participant fewer points
  HI        harmful-intent rating (0-100), not used in fitting
plus the rank residuals used for the partial correlation (harm and HI, each with the cue's rank
regressed out), so panel C plots exactly what the partial r_s summarizes.

Usage (from the project root):
  uv run python -I figures/scripts/export_fig4_intent.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "intentions_game"))

import numpy as onp  # noqa: E402
from scipy.stats import spearmanr, rankdata  # noqa: E402
from data import load_participants  # noqa: E402
from bms import RES  # noqa: E402
import intent_analysis as ia  # noqa: E402

P = load_participants()
rows = []
for g, label in (("g1", "BPD"), ("g2", "Controls")):
    zi = onp.load(RES / f"laplace_{g}_I.npz")
    for i, pid in enumerate(zi["ids"]):
        harm, e_om, p_neg, mu_rho = ia.attributions(zi["params_I"][i], P[pid]["data"])
        hc, sc = ia.cues(P[pid]["data"])
        meta = P[pid]["meta"]
        num = lambda k: float(meta[k]) if meta[k] not in ("", "NA") else onp.nan
        rows.append(dict(id=str(pid), group=label, harm=harm, e_omega=e_om, p_neg=p_neg,
                         harm_cue=float(hc), self_cue=float(sc), HI=num("HI"), SI=num("SI")))

x = onp.array([r["harm"] for r in rows]); y = onp.array([r["HI"] for r in rows])
c = onp.array([r["harm_cue"] for r in rows])
assert not onp.isnan(x + y + c).any()
rx, ry, rc = rankdata(x), rankdata(y), rankdata(c)
res = lambda a: a - onp.polyval(onp.polyfit(rc, a, 1), rc)
for r, ex, ey in zip(rows, res(rx), res(ry)):
    r["harm_resid"], r["HI_resid"] = float(ex), float(ey)

out = ROOT / "figures" / "data" / "fig4_intent.csv"
cols = ["id", "group", "harm", "e_omega", "p_neg", "harm_cue", "self_cue", "HI", "SI", "harm_resid", "HI_resid"]
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in rows:
        w.writerow([r[k] if isinstance(r[k], str) else f"{r[k]:.6g}" for k in cols])
print(f"wrote {out.relative_to(ROOT)} ({len(rows)} rows)")

# checks against the reported statistics (Section 4.5: .54, .52, partial .25, p = .012)
rm, pm = spearmanr(x, y); rcue, _ = spearmanr(c, y); pr, pp = ia.partial_spearman(x, y, c)
print(f"  HI ~ harm r_s = {rm:.2f} (p = {pm:.2g}); HI ~ cue r_s = {rcue:.2f}; partial r_s = {pr:.2f} (p = {pp:.3f})")
print(f"  residual-plot r_s (should equal the partial) = {spearmanr(res(rx), res(ry))[0]:.2f}")
print(f"  ranges: harm {x.min():.2f}-{x.max():.2f}; cue {c.min():.2f}-{c.max():.2f}")
