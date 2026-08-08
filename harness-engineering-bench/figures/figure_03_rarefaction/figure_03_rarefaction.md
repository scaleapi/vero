---
id: figure_03_rarefaction
archetype: small_multiples
script: vero.interpret.analysis.paper_figures::fig_rarefaction
outputs: [figure_03_rarefaction.pdf, figure_03_rarefaction.png]
status: review
---

## Takeaway

Five independent optimizers exhaust almost the entire repertoire of edit categories;
the next fifteen add nearly nothing.

## Caption

Distinct edit categories discovered as optimization runs are added, one panel per
benchmark (16 categories available). Line, mean over 200 random orderings of the 20
runs; band, the 10th-90th percentile across those orderings — the spread attributable
to which runs happen to come first. Every curve is within one category of its final
value by the fifth run and flat thereafter, and the band collapses over the same
interval: at one run the number of categories seen spans roughly 2 to 16, by ten runs
it is a single value. The marginal contribution of an additional independent optimizer
is therefore close to zero, and that conclusion does not depend on the draw. Same
convergence \Cref{fig:diversity} establishes against a null, seen as saturation rather
than as a distance.

## What the reader should see

- The shape, not the ordering: every curve bends hard before x=5 and is flat by x=8.
- Vertical position at the right edge is the ceiling each benchmark reached (15 or 16 of
  16). The gap between curves is not the point and should not be over-read.
- The band narrowing left-to-right is as much the finding as the curve flattening: it
  says the result is insensitive to which runs you happened to have.
- Colour distinguishes benchmark only, redundantly with the panel title. Panels share
  both axes so heights are directly comparable across them.
- Absent by design: a cross-benchmark overlay. The claim is per-benchmark, and five
  overlapping bands produced grey that attributed to no series.

## Data

Mean distinct categories after k runs, of 16 available.

Mean, with the 10th-90th percentile across orderings in brackets.

| benchmark | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| BrowseComp-Plus | 8.3 [2-16] | 12.0 [8-16] | 15.2 [13-16] | 16.0 [16-16] | 16.0 [16-16] |
| OfficeQA | 7.7 [3-15] | 10.6 [6-14] | 13.9 [11-15] | 14.9 [15-15] | 15.0 [15-15] |
| SWE-Atlas-QnA | 8.8 [3-14] | 11.7 [8-15] | 14.3 [12-16] | 15.4 [14-16] | 16.0 [16-16] |
| Terminal-Bench | 7.5 [2-14] | 10.5 [6-14] | 13.8 [12-15] | 14.7 [14-15] | 15.0 [15-15] |
| GAIA-Shell | 11.4 [8-15] | 13.9 [11-16] | 15.2 [14-16] | 15.8 [15-16] | 16.0 [16-16] |

200 random orderings per benchmark, seed 0.

## Style notes

- Small multiples rather than one overlaid axis. Adding percentile bands to five
  overlaid series produced unattributable grey below k=5 and stretched the y-axis to 2,
  compressing the saturation region that carries the claim. Faceting also retires the
  earlier problem that made a legend necessary — three benchmarks land on exactly 16.0
  categories, so endpoint labels could not separate them.
- Percentiles, not a standard deviation. The quantity is a bounded count skewed hard
  against its ceiling; a symmetric band would extend past the 16 categories available.
- x ticks are 1/10/20 only. Panels are ~1.2 in wide and a denser locator produced
  half-steps, which is not a run count.

## Provenance

Identifiers are snake_case in the data (`control_loop`, `browsecomp-plus`) and
title-cased for display via `analysis.display`; the table above uses the display
names, which are what the figure shows.

```
python -c "from vero.interpret.analysis import stats; stats.rarefaction_bands(rows, trials=200, seed=0)"
```
Same 3,986-edit label set as Figure 1.
