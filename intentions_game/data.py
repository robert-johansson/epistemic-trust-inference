"""Load the Intentions Game data (Barnby et al., 2025, eLife 14:RP104008) in the
layout the MATLAB models expect, and recover the participant order that the
HBI fits and simulations use.

MATLAB data layout (per participant, 126 rows x 9 columns):
  0 ID | 1 trial | 2 S1 | 3 O1 | 4 S2 | 5 O2 | 6 choice/prediction (1/2) |
  7 partner action (1/2) | 8 Phase (1, 2, 3)
Phases: 36 own choices, 54 predictions of the partner, 36 own choices.

Group g1 = BPD (ID shorter than 10 characters), g2 = controls (as in the
authors' BPD_CON_Analysis.R). The fitted parameters in FittedFiles/hbi_g*.mat
are ordered as in Data/intent_dat_g*.csv, which is not in the repository; we
recover that order by matching each participant's payoff and partner-choice
sequences against Data_Simulated/data_sim_g*.mat.
"""
import csv
from pathlib import Path

import numpy as onp
import scipy.io as sio

UPSTREAM = Path(__file__).resolve().parent.parent / "third_party" / "SocialTransfer_Barnby_etal_2024"
FIT_ORDER = ["Beta", "M2", "M1", "M3", "M4"]   # order of cbm.output.parameters (FitData.m)
N_PARAMS = {"Beta": 3, "M1": 6, "M2": 6, "M3": 8, "M4": 8}


def load_participants(path=UPSTREAM / "Data" / "intent_dat.csv"):
    """Return {ID: dict(data=(126, 9) float array in file row order, meta=...)}, in file order."""
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        pid = r["ID"]
        if pid not in out:
            out[pid] = {"rows": [], "meta": {k: r[k] for k in (
                "SI", "HI", "MZQ_TotalScore", "CAMSQ_Self", "CAMSQ_Other", "RGPTSA", "RGPTSB",
                "CTQtot", "A_ETS_Mistrust", "A_ETS_Trust", "A_ETS_Credulity", "Age", "Gender")}}
        out[pid]["rows"].append([0.0] + [float(r[k]) for k in (
            "trial", "S1", "O1", "S2", "O2", "choice", "real_answer", "Phase")])
    for pid, d in out.items():
        a = onp.array(d.pop("rows"))
        assert a.shape == (126, 9), (pid, a.shape)
        # contiguous phases in file order: 36 x phase 1, 54 x phase 2, 36 x phase 3
        assert (a[:, 8] == onp.repeat([1, 2, 3], [36, 54, 36])).all(), pid
        d["data"] = a
        d["group"] = "BPD" if len(pid) < 10 else "CON"
    return out


def matlab_order(participants, group):
    """IDs in the order used by the HBI fits for group 'g1' (BPD) or 'g2' (CON)."""
    sim = sio.loadmat(UPSTREAM / "Data_Simulated" / f"data_sim_{group}.mat", squeeze_me=True)["data_sim"]
    want = "BPD" if group == "g1" else "CON"
    pool = {pid: d["data"] for pid, d in participants.items() if d["group"] == want}
    order = []
    for s in sim:
        s = onp.asarray(s, dtype=float)
        hits = [pid for pid, a in pool.items()
                if onp.array_equal(a[:, 1:6], s[:, 1:6]) and onp.array_equal(a[:, 7:9], s[:, 7:9])]
        assert len(hits) == 1, f"ambiguous or missing match: {hits}"
        order.append(hits[0])
    assert len(set(order)) == len(pool)
    return order


def load_hbi(group, recovery=False):
    """cbm.output of the HBI fit: parameters per model (FIT_ORDER), responsibilities, frequencies."""
    name = f"hbi_{group}_BPD{'_recovery' if recovery else ''}.mat"
    cbm = sio.loadmat(UPSTREAM / "FittedFiles" / name, squeeze_me=True, struct_as_record=False)["cbm"]
    out = cbm.output
    return {
        "parameters": {m: onp.atleast_2d(out.parameters[k]) for k, m in enumerate(FIT_ORDER)},
        "responsibility": out.responsibility,
        "model_frequency": out.model_frequency,
        "exceedance_prob": out.exceedance_prob,
        "protected_exceedance_prob": out.protected_exceedance_prob,
    }


def load_full_sim(group, suffix=""):
    """Per-participant MATLAB outputs saved by SimulateData.m. Reads the original .mat file when present;
    otherwise the extract of the fields validate_reference.py uses (tools/extract_matlab_outputs.py)."""
    path = UPSTREAM / "Data_Simulated" / f"full_sim_{group}{suffix}.mat"
    if path.exists():
        f = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return f["F"], list(f["results"])
    from types import SimpleNamespace
    z = onp.load(UPSTREAM / "Data_Simulated" / "full_sim_extract.npz")
    key = f"full_sim_{group}{suffix}"
    fields = ("prob1", "prob2", "prob3", "simA", "beta_marg2b")
    n = z[f"{key}__prob1"].shape[0]
    return None, [SimpleNamespace(**{f: z[f"{key}__{f}"][i] for f in fields}) for i in range(n)]
