"""Check reference.py (a line-by-line NumPy translation of the authors' MATLAB models) against the MATLAB
outputs saved in Barnby et al.'s repository (Data_Simulated/full_sim_*.mat).

Two kinds of file (established by inspecting them; see the repository README):
  full_sim_g1.mat, full_sim_g2.mat   produced with sim = 0: prob1/2/3 are the model's probabilities of the
                                     participant's REAL responses, for each group's winning model at its HBI
                                     parameters (BPD: M4; controls: M1). So
                                         P_matlab(option 1)[t] = prob[t] if choice[t] == 1 else 1 - prob[t]
                                     and the total log-likelihood is sum(log prob).
  full_sim_g1_M3.mat, _g2_M3.mat     produced with sim = 1 for M3: prob is the probability of the SIMULATED action
                                     simA, so P_matlab(option 1)[t] = prob[t] if simA[t] == 1 else 1 - prob[t].
In these models a trial's choice probability does not depend on the participant's own responses, so both
reduce to a comparison of P(option 1) on every trial with reference.run(...)["netp"]. M2 and Beta have no
saved outputs.

Usage: python -I validate_reference.py
"""
import sys
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_participants, matlab_order, load_hbi, load_full_sim  # noqa: E402
import reference as ref  # noqa: E402

CHOICE = 6          # column of the participant's response (1 or 2) in the 126 x 9 data layout


def matlab_prob(r):
    return onp.concatenate([onp.atleast_1d(r.prob1)[:36], onp.atleast_1d(r.prob2)[36:90],
                            onp.atleast_1d(r.prob3)[90:126]])


def check(group, suffix, model, real_choices):
    P = load_participants()
    order = matlab_order(P, group)
    hbi = load_hbi(group)
    _, results = load_full_sim(group, suffix)
    p_err, ll_err, m_err = [], [], []
    for i, pid in enumerate(order):
        data = P[pid]["data"]
        out = ref.run(model, hbi["parameters"][model][i], data)
        prob = matlab_prob(results[i])
        resp = data[:, CHOICE] if real_choices else onp.atleast_1d(results[i].simA)
        netp_matlab = onp.where(resp == 1, prob, 1 - prob)
        p_err.append(onp.max(onp.abs(out["netp"] - netp_matlab)))
        if real_choices:
            ll_ref = onp.sum(onp.log(onp.where(data[:, CHOICE] == 1, out["netp"], 1 - out["netp"])))
            ll_err.append(abs(ll_ref - onp.sum(onp.log(prob))))
        if out["beta_marg2b"] is not None:
            m_err.append(onp.max(onp.abs(out["beta_marg2b"] - onp.atleast_1d(results[i].beta_marg2b))))
    kind = "real responses (sim = 0)" if real_choices else "simulated actions (sim = 1)"
    ll = f"; max |total log-likelihood diff| = {max(ll_err):.2e}" if ll_err else ""
    print(f"full_sim_{group}{suffix}.mat, {model}, {len(order)} participants, {kind}:\n"
          f"   max |P(option 1) diff| per trial = {max(p_err):.2e}{ll}; "
          f"max |partner posterior marginal diff| = {max(m_err):.2e}")
    return max(p_err), (max(ll_err) if ll_err else 0.0)


if __name__ == "__main__":
    worst_p, worst_ll = 0.0, 0.0
    for group, suffix, model, real in (("g1", "", "M4", True), ("g2", "", "M1", True),
                                       ("g1", "_M3", "M3", False), ("g2", "_M3", "M3", False)):
        p, ll = check(group, suffix, model, real)
        worst_p, worst_ll = max(worst_p, p), max(worst_ll, ll)
    print(f"\nWORST: per trial {worst_p:.2e}, total log-likelihood {worst_ll:.2e} -> "
          f"{'AGREE' if worst_p < 1e-9 and worst_ll < 1e-6 else 'DISAGREE'}")
