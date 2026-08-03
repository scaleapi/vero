---
id: figure_04_knob_direction
archetype: dumbbell
script: vero.interpret.analysis.paper_figures::fig_knob_direction
outputs: [knob_direction.pdf, knob_direction.png]
status: review
---

## Takeaway

Optimizers raise turn and step budgets almost without exception, but move output caps
and timeouts in both directions.

## Caption

Numeric constants most often changed, by direction of change (filled marker, raised;
open marker, lowered). Turn and step budgets move overwhelmingly upward — `MAX_TURNS`
raised in 63 edits against 14 lowered — while output-truncation caps and per-command
timeouts are as often reduced as increased. Direction is derived by comparing the
literal before and after values, not inferred from the commit message. Constants
touched by reformatting without a value change are excluded. Counted per edit rather
than per run, since one run may retune the same constant several times and each
retuning is a separate decision.

## What the reader should see

- The top two rows against the rest: budgets go up, everything else is mixed.
- `MAX_TOOL_OUTPUT_CHARS` is the notable inversion — lowered slightly more often than
  raised, the only frequently-touched constant where that holds.
- Horizontal position is a count, and the connector's length is the imbalance between
  the two directions. Open versus filled is the only other channel.
- Absent by design: the magnitudes of the changes. A constant moved 24→100 and 24→32
  count the same here; direction is the claim, size is not.

## Data

| constant | raised | lowered |
|---|---|---|
| MAX_TURNS | 63 | 14 |
| MAX_STEPS | 26 | 5 |
| MAX_TOOL_OUTPUT_CHARS | 13 | 16 |
| MAX_OUTPUT_CHARS | 3 | 3 |
| SOFT_DEADLINE_SEC | 2 | 3 |
| COMMAND_TIMEOUT_SEC | 1 | 4 |
| N_ATTEMPTS | 1 | 2 |
| MAX_HISTORY_CHARS | 0 | 2 |
| MAX_CONCURRENT_SHELLS | 2 | 0 |
| RESEARCH_DEADLINE_SEC | 0 | 2 |

Corpus totals across all scalar constants: 137 raised, 78 lowered.

## Style notes

- Top 10 constants only, by total edits. The tail is single-digit and would add rows
  without adding signal; the cut is stated here so it is not read as the full set.

## Provenance

```
python -c "from vero.interpret.analysis import stats; stats.tuning_direction(rows, edits, top=10)"
```
Direction from `taxonomy.direction_of` over captured before/after literals; same
3,986-edit set as Figure 1.
