# Epistemic trust as inference about other minds: code and results

Code, saved fits and every result log for the paper

> Johansson, R. *Epistemic Trust as Inference About Other Minds: A Computational Account of How It Is Disrupted and Restored.* Manuscript in preparation for *Frontiers in Psychiatry*, Research Topic "Epistemic Trust – From Neurobiology to Psychotherapy and Social Connectedness".

The paper treats epistemic trust (the capacity to treat communication as trustworthy, personally relevant and generalizable) as inference about other minds, and writes the models as probabilistic programs in [memo](https://github.com/kach/memo). This repository lets anyone

1. **check every number, table and figure in the paper** against the code that produced it, in minutes, from the saved fits; and
2. **refit everything from scratch** if they want to (hours).

`CLAIMS.md` maps each number in the paper to the script and log line it comes from.

## Quick start

```sh
git clone https://github.com/robert-johansson/epistemic-trust-inference.git
cd epistemic-trust-inference
uv sync --frozen          # Python 3.14 and the exact package versions used for the paper (uv.lock)
make check                # ~4 min: rerun every analysis and compare with the committed logs
```

`make check` ends with `ALL RESULTS REPRODUCED` when every regenerated result agrees with the committed one (text exactly; numbers to the last printed digit; numerical-validation residuals below 1e-8 and timings ignored, since they vary by machine).

## What is here

| Folder | Contents | Paper |
|---|---|---|
| `intentions_game/` | Exact memo port of the five published models of Barnby et al. (2025); new models **H** (hierarchical self–other), **I** (inferred intent) and **C** (coupled intent); fitting, protected model comparison, between-group tests, intent–rating tests, split-half test, parameter and model recovery; validation against independent implementations and the authors' MATLAB outputs; `results/` with every fit (`*.npz`) and log | §4, Table 3, Box 2, Figs 2–5 |
| `advice_taking/` | The advisor model of the BART-ET (Li et al., 2026) and its simulations; `ASSUMPTIONS.md` lists every task feature the original report does not give | §5, Box 1, Table 4, Fig. 6 |
| `development/` | Informant model of Corriveau et al. (2009), fitted to their Table 2; descriptive baselines and held-out transfer; the two routes to the resistant pattern | §6 |
| `figures/` | TikZ/pgfplots sources, exported data, export scripts and final PDFs of Figs 1–6 | Figs 1–6 |
| `tools/` | `reproduce.py` (the check above) and `check_memo_idioms.py` (memo-mcp's idiom check of every memo program, with its log) | §2.3 |
| `third_party/` | The files of Barnby et al.'s repository (commit `9f6d175`) that the code reads, copied unchanged under their MIT licence, plus an extract of their large MATLAB output files; checksums in its README | §4 |

## Reproducibility levels

| Command | What it does | Time |
|---|---|---|
| `make validate` | memo port vs. NumPy translation of the authors' MATLAB code vs. their saved MATLAB outputs; memo specifications of H, I and C vs. factorized implementations | ~1 min |
| `make check` | everything in `make validate`, plus every result the paper reports, from the saved fits; compares with the committed logs and changes nothing | ~4 min |
| `make numbers` | same as `make check` but overwrites the committed logs | ~4 min |
| `make figures` | compiles the six figures with xelatex (TeX Live; fonts TeX Gyre Heros and Fira Math) | ~1 min |
| `make idioms` | memo-mcp's idiom check of every memo program; runs in memo-mcp's own environment (install [memo-mcp](https://github.com/robert-johansson/memo-mcp), then `make idioms MEMO_MCP_DIR=/path/to/memo-mcp`, and `MEMO_MCP_PYTHON=...` if its Python is not `venv/bin/python`) | ~1 min |
| refit (below) | all fits, split-half fits and recovery studies from scratch | many hours |

Model comparison draws its Monte Carlo samples from a fixed seed, so results are deterministic on a given platform.

## The memo programs

| Program(s) | File | What it models | Paper |
|---|---|---|---|
| `own_choice`, `partner_choice`, `predict_partner` (+ β-only versions) | `intentions_game/memo_models.py` | The published models M1–M4 and Beta | §4.2 |
| `partner_likelihood`, `predict_partner`, `own_choice` | `intentions_game/hier_spec.py` | Model H: self and partner as draws from one population (specification; `hier_core.py` is the factorized version used for fitting) | §4.4, Fig. 3 |
| `intent_choice`, `predict_intent`, `attribute_intent`, `own_choice_intent` | `intentions_game/intent.py` | Model I: inferring the partner's intent (helping/hindering) and the attributions it implies | §4.5, Fig. 4 |
| `predict_partner`, `own_choice`, `partner_likelihood` | `intentions_game/hier_intent_spec.py` | Model C: self and partner as the same kind of agent, with motives from one population | §4.6, Box 2 |
| `choose_pumps`, `update_reliability` | `advice_taking/bart_et_advisor.py` | The advisor model | §5, Box 1 |
| `ask`, `endorse`, `update`, `believes` | `development/corriveau_trust.py`, `corriveau_routes.py` | Children's trust in mother vs. stranger | §6 |

All programs compile under memo's static checks and pass memo-mcp's idiom check (`tools/check_memo_idioms.log`).

## Refitting from scratch

Run from `intentions_game/` with `uv run python -I ...`. Long runs are checkpointed and can be split into parts and run in parallel.

```sh
uv run python -I fit.py g1 && uv run python -I fit.py g2                    # published models, ~40 min (BPD), ~10 min (controls)
uv run python -I fit.py g1 --models H,I,C && uv run python -I fit.py g2 --models H,I,C
for k in 0 1 2; do uv run python -I split_half.py run $k 3; done            # ~1.3 h per part
uv run python -I recovery_h.py                                               # ~45 min
for k in 0 1; do uv run python -I recovery_i.py run $k 2; done              # ~15 min per part
uv run python -I model_recovery.py                                           # ~55 min
```

Then `make check` compares the downstream results with the committed logs. Timings are from an Apple-silicon laptop (CPU, JAX).

## Data and licences

- **Intentions Game data** (Barnby et al., 2025, *eLife* 14:RP104008): openly shared by the authors under the MIT licence. The files the code reads are copied unchanged into `third_party/` (with their licence and SHA-256 checksums), so this repository and its archived release are self-contained. Ethics approval for the original study: Research Ethics Committee for Wales, 12/WA/0283. `intentions_game/reference.py` is a translation of the authors' MATLAB models; see `NOTICE`.
- **BART-ET** (Li et al., 2026): no data are used; the task's reported design is simulated (`advice_taking/ASSUMPTIONS.md`).
- **Corriveau et al. (2009)**: the published group-level proportions of their Table 2, cited in the code.
- **Licences**: code under the MIT licence (`LICENSE`); figures under CC BY 4.0 (`LICENSE-figures`).

## How this code was made

The models and analyses were developed with the assistance of a large language model, Claude (Anthropic; Claude Opus 5.5), which wrote and checked much of the code under the author's direction. Every result was validated as described above (independent implementations, the original authors' outputs, parameter and model recovery), and the author takes responsibility for the code and results. The paper's Generative AI Statement describes this in full.

## Citation

See `CITATION.cff`. Please cite the paper and, for the data, Barnby et al. (2025).
