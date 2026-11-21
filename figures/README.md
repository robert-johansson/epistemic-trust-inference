# Figures

Each figure is a standalone TikZ/pgfplots document compiled with **xelatex** at exactly the Frontiers template's `\textwidth` (504 pt = 177.1 mm), so it is included at 100% and never rescaled. Licence: CC BY 4.0 (`../LICENSE-figures`).

| Figure | Source | Data (export script) |
|---|---|---|
| 1, framework (§2) | `fig1_framework.tex` | none |
| 2, Intentions Game and model space (§4.1–4.2) | `fig2_game.tex` | none |
| 3, coupling (§4.4) | `fig3_coupling.tex` | `data/fig3_coupling.csv` (`scripts/export_fig3_coupling.py`) |
| 4, intent vs ratings (§4.5) | `fig4_intent.tex` | `data/fig4_intent.csv` (`scripts/export_fig4_intent.py`) |
| 5, form of the theory (§4.6) | `fig5_form.tex` | `data/fig5_families.csv`, `data/fig5_rows.tex`, `data/fig5_splithalf.csv` (`scripts/export_fig5_form.py`) |
| 6, BART-ET stances (§5.3) | `fig6_bart.tex` | `data/fig6_*.csv` (`scripts/export_fig6_bart.py`) |

Shared style: `figstyle.tex` (TeX Gyre Heros and Fira Math, loaded by file name from TeX Live; colour roles checked for colour-vision deficiency: groups blue/orange, stances green/violet with dashes for rigid stances, theory curves in ink; hairline axes; labels are nodes placed against anchors; no hyphenation).

## Build
From the repository root:
```sh
make numbers   # (re)exports figures/data along with all results
make figures   # compiles all six figures into figures/pdf/
```
Fig. 5A's rows are generated into `data/fig5_rows.tex` by the export script, so no value is typed by hand.
