# Attachment and selective trust (Section 6)

Corriveau et al. (2009), *Child Development* 80(3), 750–761: children (N = 147; attachment assessed at 15 months) chose between their mother's and a stranger's conflicting claims at 50 and 61 months. The models are fitted to the group-level proportions of the paper's Table 2, which are reproduced in `corriveau_trust.py` as cited published values.

| File | What it does |
|---|---|
| `corriveau_trust.py` | memo programs `ask`, `endorse`, `update` (informants are helpful or uninformative; within-task learning); fit to the 24 Table-2 cells; lesions |
| `corriveau_baselines.py` | additive descriptive baselines; transfer to the held-out 75–25 task |
| `corriveau_routes.py` | adds `believes`; two routes to the resistant pattern (credulous prior vs cost of disagreeing): similar endorsements, different private beliefs |

Run from the repository root: `uv run python -I development/<file>.py`. The support is qualitative (see the paper).
