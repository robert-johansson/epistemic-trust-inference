# The BART-ET advisor model: what is assumed (Section 5.2)

Li et al. (2026) report the design of the BART-ET but not every detail a model needs, and no trial-level data are
available. `bart_et_advisor.py` therefore simulates the task; its numbers illustrate the logic of the model and are
not predictions of the published scores. Every assumption is marked `ASSUMED` in the code:

| Feature | Reported by Li et al. | Assumed here |
|---|---|---|
| Rounds | Round 1: 30 balloons, 21 burst regardless; Round 2: 30 balloons with advice | Round 1 enters only through a pessimistic prior over burst points |
| Advice accuracy schedule | Accurate on balloons 1–3, inaccurate on 4–6, accurate on 7–11, then 15 accurate and 4 inaccurate | Positions of the 4 late inaccurate balloons: 15, 20, 24, 28 |
| Pump scale | not reported | 11 levels (0–10) |
| Advised values | not reported | drawn once from levels 4–7 (fixed seed) |
| Burst rule | not reported | a burst point compared with the number of pumps entered |
| Advisor | experimenter "plays along" | knows the burst point; helps (high but safe advice) with probability r, otherwise misleads (advice at or above the burst point) |
| Participant | — | softmax over expected points (inverse temperature `beta`), with uncertainty over the burst point and over r |

Stances are priors over r on a five-point grid: trusting (mean .70, weakly held), mistrusting (.30, weakly held), credulous (.89, rigidly held), petrified (.31, rigidly held).
