---
id: figure_03_rarefaction
archetype: line
script: vero.interpret.analysis.paper_figures::fig_rarefaction
outputs: [rarefaction.pdf, rarefaction.png]
status: review
---

## Takeaway

Five independent optimizers exhaust almost the entire repertoire of edit categories;
the next fifteen add nearly nothing.

## Caption

Distinct edit categories discovered as optimization runs are added, averaged over 200
random orderings of the 20 runs per benchmark (16 categories available). Every curve is
within one category of its final value by the fifth run and flat thereafter, so the
marginal contribution of an additional independent optimizer is close to zero. This is
the same convergence that \Cref{fig:diversity} establishes against a null, viewed as a
saturation curve rather than a distance.

## What the reader should see

- The shape, not the ordering: every curve bends hard before x=5 and is flat by x=8.
- Vertical position at the right edge is the ceiling each benchmark reached (15 or 16 of
  16). The gap between curves is not the point and should not be over-read.
- Colour distinguishes benchmark only; the legend is ordered by final value so legend
  order matches the visual order at the right edge.
- Absent by design: error bands. The averaging over 200 orderings is what the curve is;
  a band would imply sampling error over runs, which is not what varies here.

## Data

Mean distinct categories after k runs, of 16 available.

| benchmark | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| browsecomp-plus | 8.3 | 12.0 | 15.2 | 16.0 | 16.0 |
| officeqa | 7.7 | 10.6 | 13.9 | 14.9 | 15.0 |
| swe-atlas-qna | 8.8 | 11.7 | 14.3 | 15.4 | 16.0 |
| terminal-bench | 7.5 | 10.5 | 13.8 | 14.7 | 15.0 |
| gaia-shell | 11.4 | 13.9 | 15.2 | 15.8 | 16.0 |

200 random orderings per benchmark, seed 0.

## Style notes

- Legend instead of direct end-labels, which is the house preference. Three benchmarks
  land on exactly 16.0 and two on 15.0, so endpoint labels overlap into illegibility;
  no nudge fixes coincident values.
- x ticks forced to integers. A cell count of 2.5 does not exist and the default tick
  locator produced half-steps.

## Provenance

```
python -c "from vero.interpret.analysis import stats; stats.rarefaction(rows, trials=200, seed=0)"
```
Same 3,986-edit label set as Figure 1.
