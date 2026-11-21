"""Extract, from Barnby et al.'s four large MATLAB output files (Data_Simulated/full_sim_*.mat, 76 MB), exactly the
fields that intentions_game/validate_reference.py compares against: prob1, prob2, prob3, simA and beta_marg2b,
per participant. The result (about 0.5 MB) lets the repository and its archived release check the port against the
authors' outputs without redistributing the full files. If the original files are present in
third_party/SocialTransfer_Barnby_etal_2024/Data_Simulated/, data.load_full_sim reads them instead.

Usage: uv run python -I tools/extract_matlab_outputs.py /path/to/SocialTransfer_Barnby_etal_2024
       (a checkout at commit 9f6d175 of https://github.com/josephmbarnby/SocialTransfer_Barnby_etal_2024)
"""
import hashlib
import sys
from pathlib import Path

import numpy as onp
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "third_party" / "SocialTransfer_Barnby_etal_2024" / "Data_Simulated" / "full_sim_extract.npz"
FILES = ["full_sim_g1", "full_sim_g2", "full_sim_g1_M3", "full_sim_g2_M3"]
FIELDS = ["prob1", "prob2", "prob3", "simA", "beta_marg2b"]


def main(src):
    arrays = {}
    for name in FILES:
        path = Path(src) / "Data_Simulated" / f"{name}.mat"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        results = list(sio.loadmat(path, squeeze_me=True, struct_as_record=False)["results"])
        for field in FIELDS:
            rows = [onp.atleast_1d(getattr(r, field)).astype(float) for r in results]
            assert len({row.shape for row in rows}) == 1, (name, field)
            arrays[f"{name}__{field}"] = onp.stack(rows)
        arrays[f"{name}__sha256"] = onp.array(digest)
        print(f"{name}.mat  {len(results)} participants  sha256 {digest}")
    onp.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1])
