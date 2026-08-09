# Pattern lenses

Six angles to check for when drafting observations in step 2. Originally these
were six separate blind agents; cut after the first run because a 20-line script
(`scripts/cross_cell_stats.py`) got the same numbers faster and exactly, and
reading the per-cell reports through this checklist yourself covers the rest.
Use it as a checklist while reading, not a dispatch list.

Each entry gives what the lens looks for, why it earns a slot, and what it turned
up when the procedure was first run on tau3 (16 cells, 83 candidates, gpt-5.4-mini
target) and by hand on GAIA (46 cells). Treat those as calibration for the *kind* and
*specificity* of observation wanted — not as expected answers. If a lens on a new
benchmark reproduces the tau3 finding verbatim, suspect the agent read this file as
a hint sheet rather than reading the cells.

## 1. Measurement discipline

Does the optimizer behave like an experimentalist or like a code generator?

Look for: commit messages citing measured numbers; candidates evaluated on `dev`
before `val`; the same idea tried twice with a variation; explicit reverts after a
measurement; ideas abandoned without ever being scored.

Why it matters: this is the closest thing to evidence that these agents can do
empirical work rather than plausible-looking edits. It's the finding most likely
to survive into the paper's argument.

*tau3:* 11 of 83 candidates flagged as revert-ish by keyword, across 8 of 16 cells;
treat that as a list to verify, not a count to quote. 4 of 16 cells shipped a
revert as their final answer, including the top scorer at 0.6067 — its message read
*"Revert v7 guidance: it regressed 8 of 30 development cases."* Reverting on a
measurement, not a hunch.

## 2. Unpriced knobs

Edits to knobs that change behaviour but aren't in the objective: `MAX_TURNS`,
`reasoning_effort`, tool-output caps, client retry counts, timeouts, context limits.

Why it matters: the objective is single-term (`metric: score`). Anything the
optimizer spends on latency or tokens it does for reasons the reward never asked
for, and finding a knob that buys accuracy for unpriced cost is a distinct
capability from improving a harness. This is also where the paper is most exposed:
if gains come mostly from knob-turning, "harness engineering" is doing less work
than the framing implies. Count these carefully.

Look for: which knobs get touched at all, in which direction, and whether the
optimizer's message reasons about the cost it isn't charged for. Note knobs that
*no* cell ever touched — the untouched ones bound the claim.

*tau3:* `reasoning_effort` touched by 6 candidates across 5 of 16 cells, reverted
once: *"agent: revert reasoning effort to medium (no gain at high, higher
latency)"* (`kimi-k3-opencode-r2`, `7394ea436c04`) — the optimizer declining a win
the objective would have paid for, since latency is never charged. Note it cited no
number for the "no gain" half, and the two candidates' dev scores were in fact
identical (0.453), so the claim holds but the reasoning was cheaper than it sounds. *GAIA:* 23% of
shipped candidates changed at least one parameter; no cell ever set
`[agent] timeout_sec`, which was available.

## 3. Inert and cosmetic changes

Candidates that cannot have changed behaviour: comments, formatting, dead code,
`.gitignore`, docstrings, renames, config that isn't read.

Why it matters: it separates real optimization from motion. A cell whose shipped
candidate is inert scored whatever it scored *without the harness changing*, which
makes it an accidental noise measurement — genuinely useful for bounding
cell-to-cell variance.

Look for: diffs touching only non-executed lines; a shipped candidate identical in
behaviour to the seed; edits to files the target never imports.

*tau3:* 2 of 16 cells shipped a tree functionally identical to the seed, confirmed
by tree hash, not by trusting the commit message — `claude-sonnet-5-claude-code-r2`
(`0031cb45be1f`, tree identical to seed) and `claude-sonnet-5-opencode-r1`
(`7179bd048021`, cumulative diff = a 3-line `.gitignore`). They scored 0.4267 and
0.4511 while the seed measured 0.5618 on the same partition through a different
script, which is how a 0.12 measurement-path gap surfaced. Run this lens first.

## 4. Structural shape

The physical shape of the edits: size, spread, and what kind of thing was changed.

Why it matters: mostly to kill the intuition that bigger edits do more. If diff
size doesn't predict reward, that's worth one sentence in the paper and it stops a
reviewer asking.

Look for: insertions/deletions per shipped candidate against reward; how many files
a candidate spans; prompt/instruction text vs control flow vs configuration vs new
helper code; whether cells that edit prompts differ systematically from cells that
edit code.

*tau3:* diff size carried no signal — the best cell shipped +1/−8 and the worst
+19/−10. Candidate *count* did correlate with reward (+0.405, n=16, so suggestive at
best): cells that iterated more scored better, independent of how much they wrote.

## 5. Failure-mode targeting

Did the optimizer find the *actual* dominant failure mode, and did it aim at it?

Look for: commit messages naming a specific failure (a crash, a tool misuse, a
truncation, a format violation); edits that plainly target one; the gap between
what the optimizer believed was failing and what the eval records show. And the
inverse: known failure modes nothing ever touched.

Why it matters: a benchmark where every cell fixes the same thing is a benchmark
with one dominant failure mode — exactly the concern raised about DABstep and
OfficeQA jumping most of their headroom in one step. Convergent targeting across
independent cells is evidence for that; divergent targeting is evidence against.

Include the seed's known defects in the agent's brief where they're documented
(e.g. tau3's empty-turn crash), so it can check whether the optimizer independently
found what we already knew was wrong.

## 6. Cost and latency awareness

Does the optimizer reason about time and money it isn't scored on?

Look for: messages mentioning latency, tokens, cost, timeouts, or turn budgets;
edits that trade accuracy for speed or the reverse; awareness of the per-case
timeout; any sign it modelled the eval loop's economics.

Why it matters: the reward is accuracy-only, but the *paper* wants to talk about a
reward vector including latency and dollars. Whether the optimizer volunteers this
reasoning unprompted bears directly on whether pricing it would change behaviour.

Overlaps lens 2 by design — that lens counts knob edits, this one reads the
reasoning. Two agents reaching the same finding by different routes is
corroboration, and it's cheaper than trying to draw a clean boundary.

## Adding a lens

Add one when a benchmark has an affordance these don't cover — a browsing
benchmark's retry-and-backoff behaviour, a coding benchmark's test-writing, a
multimodal benchmark's image handling. Write it in the same shape (what to look
for, why it earns a slot) and leave the calibration section empty until it's been
run. A lens with fabricated example findings is worse than no lens.
