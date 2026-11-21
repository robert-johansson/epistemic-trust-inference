# Intentions Game: memo port, new models and re-analysis (Section 4)

Data: Barnby et al. (2025), *eLife* 14:RP104008, through the submodule `../third_party/SocialTransfer_Barnby_etal_2024` (commit `9f6d175`, MIT licence). Run every script from this folder with `uv run python -I <script>`.

## Files

| Role | Files |
|---|---|
| Data loading | `data.py` (participant data in the authors' MATLAB layout; recovers the participant order of the HBI fits by matching sequences) |
| Published models | `reference.py` (line-by-line NumPy translation of the MATLAB models), `memo_models.py` (the memo port; also the glue for H, I and C) |
| Model H (hierarchical self–other) | `hier_spec.py` (memo specification with an explicit population agent), `hier_core.py` (factorized version used for fitting), `coupling_demo.py` |
| Model I (inferred intent) | `intent.py` |
| Model C (coupled intent) | `hier_intent_spec.py` (memo specification); fitted through `memo_models.py` and `intent.py` |
| Fitting | `fit.py` (MAP + Laplace evidence, prior N(0, 6.5) as the authors' first stage; SciPy BFGS on exact JAX gradients), `split_half.py` |
| Model comparison | `bms.py` (random-effects BMS; HBI comparison; MAP vs HBI agreement), `pxp_report.py` (protected exceedance probabilities), `family_report.py` (family tests, contagion contrasts, exact between-group tests), `bms_h.py` |
| Analyses | `intent_analysis.py` (intent vs ratings; benevolence prior), `representation_accuracy.py`, `etmcq_check.py` (credulity re-tests) |
| Recovery | `recovery_h.py`, `recovery_i.py`, `model_recovery.py` |
| Validation | `validate_reference.py` (vs the authors' MATLAB outputs), `validate_memo.py` (memo vs reference), `validate_hier.py`, `validate_intent.py`, `validate_hier_intent.py` |
| Results | `results/*.npz` (all fits) and `results/*.log` (every result; see `../CLAIMS.md`). Logs named `fit_*`, `recovery_H.log`, `recovery_I_part*`, `split_half_part*` and `model_recovery.log` are progress logs of the long runs. |

## Validation (from `results/`)
- Reference vs the authors' MATLAB outputs: ≤ 5.2e-13 per trial, ≤ 7.5e-12 total log-likelihood (M1, M3, M4; M2 and Beta have no saved outputs). The two main output files were produced with `sim = 0` (probabilities of the participants' real responses); the `_M3` files with simulated actions (`validate_reference.py` handles both).
- memo vs reference: ≤ 2.4e-15 per trial, all five models and 103 participants.
- H, I and C: memo specifications vs factorized or independent implementations agree to ≤ 1e-14.

## Model H in one paragraph
The participant believes that their own and the partner's social preferences are two draws from one population whose norm is unknown. They know their own preferences only roughly and learn the partner's from its choices. Self-insertion (the prior over the partner is pulled toward oneself) and contagion (one's own preferences move toward the partner after observation) are then one inference run in two directions, governed by the population spread τ. M1 already couples the two through shared prior widths; its phase-3 update is a moment-matched shortcut of H's inference, and the two still differ in details such as the width of the self-centered prior.
