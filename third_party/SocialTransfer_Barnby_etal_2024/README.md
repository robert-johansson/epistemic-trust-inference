# Barnby et al. (2025): the files this repository uses

Source: https://github.com/josephmbarnby/SocialTransfer_Barnby_etal_2024, commit `9f6d175` (MIT licence, `LICENSE` in this folder).
Paper: Barnby, J. M., et al. (2025). Self-other generalisation shapes social interaction and is disrupted in borderline personality disorder. *eLife*, 14, RP104008.

Copied unchanged (SHA-256):

| File | SHA-256 |
|---|---|
| `Data/intent_dat.csv` | `9ecadf438741609ab6d97ca00b2633ae141a055d5b602da2d54cce7b24cb2711` |
| `FittedFiles/hbi_g1_BPD.mat` | `0147829697473a92baadef2ff493f078445cb3c5952b7b1195d1b5fd520219f7` |
| `FittedFiles/hbi_g2_BPD.mat` | `3836b98b1b069cbae0b31ed1f63310569ee71500ccbc73f776b154641049102e` |
| `Data_Simulated/data_sim_g1.mat` | `52446d063bc81b381e102b0139a2d51e355d0f2e6ad9c992026a7e555bbac0a3` |
| `Data_Simulated/data_sim_g2.mat` | `cdea76c1cab9be723c665c144fcdeb6516ecbd49cdee9f0af9842801e158771a` |

Not copied (76 MB; used only by `intentions_game/validate_reference.py`): the four MATLAB output files below. `Data_Simulated/full_sim_extract.npz` holds exactly the fields that check reads (prob1, prob2, prob3, simA, beta_marg2b), made by `tools/extract_matlab_outputs.py`. To check against the full originals, place them in `Data_Simulated/`; the code then reads them instead.

| Original file (not included) | SHA-256 |
|---|---|
| `Data_Simulated/full_sim_g1.mat` | `89710c30d5afaa5c491b98fdc20971417f0729649c065e2d26012d50689e798c` |
| `Data_Simulated/full_sim_g2.mat` | `d471778c7a4967e050ba4d25c7933047c5cad1df621bade7d34aa93938273e64` |
| `Data_Simulated/full_sim_g1_M3.mat` | `0c908bb8169ac3f70880d9cf0260f23b2f639f7582c9d19f7f119e6876610f27` |
| `Data_Simulated/full_sim_g2_M3.mat` | `f01236f2d5da91cf767604e3c0365692d7a23b8916626723b7c4d2590f86a064` |
