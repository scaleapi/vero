# Archived benchmarks

Three optimization tasks that were wired, run, and left out of the paper. They
still compile with `vero harbor build`, but they are not maintained to the
conventions in `../CONFIGURATION.md` and nothing in CI exercises them.

| benchmark | why it is not in the paper |
|---|---|
| `swe-atlas-qna/` | The binary held-out reward sits at the floor (seed 0.067, sd 0.025): 20 optimizer cells spanned 0.054 to 0.127 and no adjacent pair separated. The continuous `agg_score` the rubric also emits (seed 0.632) is more informative but is not surfaced as a selectable reward. |
| `tau3/` | Its user simulator and grader are both LLMs, so a single unchanged harness scored 0.800 and 0.547 on the same development set. Deltas under about 0.1 are unresolved, and only 4 of 16 grid cells recorded a usable baseline. |
| `swe-bench-pro/` | Never normalized to the shared conventions: 1x case budgets, single-shot held-out scoring, no pinned baseline, an agent clock at 0.6x the declared case timeout, and a seed that edits nothing. Its `build.yaml` is the full 731-instance split; `build.sample.yaml` is a nested 33/66/66 subsample. |

Runs that were made against these configs are under `runs/swe-atlas-qna/`,
`runs/tau3/` and `runs/conformance/`, and the write-ups that judged them are
`runs/AUDIT-swe-atlas-qna.md` and the tau3 rows of `runs/RESULTS-INDEX.md`.
