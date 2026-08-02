# Output template

Write to `harness-engineering-bench/<benchmark>/OPTIMIZER-BEHAVIOR.md`.

The reader is whoever writes the paper's discussion section. They want material
they can lift, and they want to be able to check any bullet in under a minute
without asking you. That shapes everything below.

## Structure

```markdown
# What the optimizer did — <benchmark>

<Two or three sentences: how many cells, which target model and harness, how many
candidates, and the one thing a reader should take away. No preamble about
methodology — that's the last section.>

## Scope

- N cells analyzed, M excluded (why, per cell)
- K candidates, seed excluded
- Reward range across cells: low – high

## How it worked

<Bullets. The optimizer's process: iteration, measurement, convergence.>

## What it changed

<Bullets. The substance of the edits.>

## What it did not do

<Bullets. Absences, with the denominator that makes them absences.>

## What this cannot separate

<Two to four bullets. The confounds, stated once, plainly.>

## Method

<Short. Which script, which artifacts, how findings were verified. Enough that
someone can rerun it, not a narrative.>
```

## Bullet form

One claim, citation inline, no wind-up:

> - 11 of 73 candidates (15%) were explicit reverts of the optimizer's own earlier
>   work, and 4 of 12 cells shipped a revert as their final answer — including the
>   top scorer at 0.6067: *"Revert v7 guidance: it regressed 8 of 30 development
>   cases"* (`claude-opus-5-opencode-r2`, `a3f2c1d8`).

That bullet works because the count has a denominator, the quote is the
optimizer's own, and the citation lets a doubter go look. Compare:

> - The optimizers showed a notable tendency toward self-correction, often
>   reverting changes that appeared to have regressed performance.

Same information, unusable. No count, no citation, "often" and "appeared to"
doing the work that evidence should.

More bullets at the right length:

> - Diff size carried no signal: the best cell shipped +1/−8 lines
>   (`claude-opus-5-opencode-r2`, `8487bf75508e`), the worst +19/−10
>   (`claude-sonnet-5-opencode-r2`, `b125bda2614f`). Candidate *count* did track
>   reward (r = +0.405, n = 16 — suggestive, not significant): cells that iterated
>   more scored better regardless of how much they wrote.

> - `reasoning_effort` was found by 5 of 16 cells and explicitly reverted by one,
>   *"agent: revert reasoning effort to medium (no gain at high, higher latency)"*
>   (`kimi-k3-opencode-r2`, `7394ea436c04`) — the optimizer declining a change the
>   objective would have paid for, since reward is single-term on score and never
>   charges for latency.

> - No cell ever set `[agent] timeout_sec`, though it is exposed in the same config
>   block as knobs that 23% of shipped candidates did change. Whatever draws the
>   optimizer to a knob, availability alone isn't it.

> - 2 of 16 cells shipped a harness functionally identical to the seed: one whose
>   shipped tree hash equals the seed's exactly (`claude-sonnet-5-claude-code-r2`,
>   `0031cb45be1f`, tree `65e2c147b655`, test 0.4267), one whose only cumulative
>   diff is a 3-line `.gitignore` the target never reads
>   (`claude-sonnet-5-opencode-r1`, `7179bd048021`, test 0.4511). Both had made
>   scored attempts and measured every one below the seed, so reverting was the
>   best move available to them.

Note the shape of that last bullet: it cites a *tree hash*, not a commit message.
"Shipped nothing" is the single most consequential claim this analysis can make —
it converts a cell into a free measurement of the unmodified seed — so it has to
rest on `git rev-parse <sha>^{tree}` agreeing, never on a commit that says
"revert". On tau3 those two cells scored 0.4267 and 0.4511 through the
finalization path while the seed measured 0.5618 through `rescore_candidate.py`,
which is how a 0.12 measurement-path gap became visible at all. If your benchmark
has such a cell, it is the most valuable one in the corpus — find it first.

## Citations

`(cell-name, 12-char-sha)`. Cell name as it appears in `runs/<benchmark>/`, since
that's what someone will `cd` into. Include reward when the bullet is about
outcome. For a claim spanning cells, cite two or three examples rather than all of
them — the count carries the generality, the citations prove the kind.

## What to leave out

- **Reward analysis.** Score tables belong in `RESULTS.md`. Here, reward appears
  only where it's evidence about behaviour.
- **Hedging on every bullet.** State confounds once, in their own section, and
  then write plainly. A bullet qualified three ways reads as if you don't believe
  it.
- **Methodology narrative.** Nobody needs the fan-out described. One short Method
  section, at the end.
- **Anything unverified.** If it didn't survive step 3, it isn't in the file. A
  finding you liked but couldn't support is worth a line in the Method section as
  a dropped claim, if it's the kind of thing a reader would otherwise assume you
  checked.
