---
id: figure_01_prevalence
archetype: heatmap
script: vero.interpret.analysis.paper_figures::fig_prevalence
outputs: [figure_01_prevalence.pdf, figure_01_prevalence.png]
status: review
---

## Takeaway

Every optimizer edits the instruction prompt and the control loop, but which other
parts of the harness they touch is dictated by the benchmark, not by the optimizer.

## Caption

Share of optimization runs that made at least one edit of each kind, per benchmark
(20 runs each, 100 total). Cell shading is the share; the annotation is the count.
Prompt and control-loop edits are near-universal, while the remaining categories vary
sharply by benchmark — submission-path edits appear in 15 of 20 swe-atlas-qna runs but
2 of 20 terminal-bench runs, and retrieval edits appear almost only in
browsecomp-plus, the sole benchmark with a retrieval corpus. Counted per run rather
than per edit, since runs produced between 1 and 18 candidates and an edit-weighted
count would measure verbosity instead of coverage. The left colour bar groups
categories. \textsuperscript{*}gaia-shell's seed is an empty skeleton, so every
category is present there by construction and its column is not comparable with the
others. Reward is not shown; see \Cref{tab:signal-validity} for why score-versus-category
comparisons are unsupportable in this corpus.

## What the reader should see

- Read down the first two rows first: `prompt` and `control_loop` are dark across every
  column. That is the universal behaviour.
- Then read across `submission`, `retrieval`, `tool_impl` — the variance between columns
  is the finding. Benchmark, not optimizer, selects the target.
- Shading encodes the same quantity as the annotation, deliberately: shading carries the
  pattern at a glance, the fraction carries the exact read. Nothing else is encoded.
- `gaia-shell` is marked and excluded from any cross-benchmark claim.
- Absent by design: reward, and any ordering of benchmarks by score.

## Data

Cells that made ≥1 edit of each kind, out of 20 per benchmark. Rows ordered by mean
share across benchmarks, which is the order the figure uses.

| role | browsecomp-plus | officeqa | swe-atlas-qna | terminal-bench | gaia-shell* |
|---|---|---|---|---|---|
| prompt | 20/20 | 19/20 | 20/20 | 20/20 | 18/20 |
| control_loop | 17/20 | 19/20 | 18/20 | 17/20 | 20/20 |
| budget_turns | 15/20 | 17/20 | 16/20 | 18/20 | 11/20 |
| tool_surface | 15/20 | 7/20 | 15/20 | 10/20 | 18/20 |
| tool_impl | 16/20 | 7/20 | 11/20 | 10/20 | 18/20 |
| other | 14/20 | 8/20 | 9/20 | 9/20 | 20/20 |
| tests | 11/20 | 7/20 | 14/20 | 10/20 | 16/20 |
| model_client | 11/20 | 14/20 | 10/20 | 8/20 | 11/20 |
| metadata | 9/20 | 7/20 | 6/20 | 11/20 | 19/20 |
| submission | 7/20 | 8/20 | 15/20 | 2/20 | 18/20 |
| budget_output | 5/20 | 13/20 | 11/20 | 7/20 | 7/20 |
| budget_wallclock | 5/20 | 7/20 | 7/20 | 6/20 | 12/20 |
| context_mgmt | 7/20 | 7/20 | 6/20 | 10/20 | 7/20 |
| initialization | 6/20 | 4/20 | 4/20 | 5/20 | 17/20 |
| env_setup | 7/20 | 5/20 | 3/20 | 7/20 | 13/20 |
| retrieval | 9/20 | 0/20 | 1/20 | 0/20 | 2/20 |

## Style notes

- Column tick labels sit on top, not bottom: with 16 rows the eye enters at the top and
  the header should be adjacent to the first row it applies to.
- Benchmark names are abbreviated to keep all five columns inside the text width without
  rotation; rotated column headers cost more legibility than the abbreviation does.

## Provenance

```
vero interpret extract --runs runs/{officeqa,browsecomp-plus,terminal-bench,swe-atlas-qna,gaia-shell} \
                       --cells-file scope100.json
vero interpret edits
vero interpret label --model gpt-5.4-mini
python -c "from vero.interpret.analysis import stats; stats.prevalence(rows)"
```
3,986 symbol-scoped edits over 100 runs. Roles set by deterministic rule where the
path, symbol kind or name settles it (2,114 edits) and by model otherwise (1,872).
