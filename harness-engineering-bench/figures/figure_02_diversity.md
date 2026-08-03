---
id: figure_02_diversity
archetype: dumbbell
script: vero.interpret.analysis.paper_figures::fig_diversity
outputs: [diversity.pdf, diversity.png]
status: review
---

## Takeaway

Independent optimizers converge on more similar repertoires of edits than chance
allows, in every benchmark.

## Caption

Mean pairwise Jaccard distance between the sets of edit categories that different
optimization runs touched (filled marker), against a permutation null that holds each
run's repertoire size and the corpus-wide category frequencies fixed and reshuffles the
assignment (open marker, null mean; band, null 95\% interval). All five benchmarks fall
below their null, so runs are more alike than independent draws from the same marginal
would be. The raw distance alone carries no information without the null: a value near
0.5 is equally consistent with genuine diversity and with every run sampling a few
categories from one skewed distribution. 20 runs per benchmark, 190 pairs each.

## What the reader should see

- The filled marker sits left of the grey band in every row. That gap is the whole
  finding: left of the null means more similar than chance.
- Position on the x-axis is the only quantitative channel. The connector shows the size
  of the departure from the null mean; it encodes nothing extra.
- Open versus filled distinguishes null from observed. Colour does not carry benchmark
  identity here, because the y-axis label already does.
- Absent by design: a significance star or p-value. The null interval is shown directly
  so the reader judges the margin rather than a threshold.

## Data

| benchmark | runs | observed | null 2.5% | null 97.5% | null mean | verdict |
|---|---|---|---|---|---|---|
| browsecomp-plus | 20 | 0.561 | 0.630 | 0.667 | 0.651 | converged |
| officeqa | 20 | 0.564 | 0.663 | 0.709 | 0.687 | converged |
| swe-atlas-qna | 20 | 0.538 | 0.601 | 0.653 | 0.630 | converged |
| terminal-bench | 20 | 0.572 | 0.643 | 0.691 | 0.670 | converged |
| gaia-shell | 20 | 0.341 | 0.399 | 0.446 | 0.425 | converged |

500 permutations per benchmark, seed 0.

## Style notes

- Value labels sit above the marker, not beside it: gaia-shell's observed value is at
  the far left of the axis and a left-placed label collided with its tick label.
- Legend is upper-left. Lower-right — the usual choice — overlapped the terminal-bench
  row, whose null band extends furthest right.
- gaia-shell is included here, unlike Figure 1, because this statistic is computed
  within a benchmark and never compared across them, so its constructed seed does not
  distort it.

## Provenance

```
python -c "from vero.interpret.analysis import stats; stats.jaccard(rows, trials=500, seed=0)"
```
Same 3,986-edit label set as Figure 1.
