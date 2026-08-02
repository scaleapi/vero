---
name: analyze-optimizer-commits
description: >-
  Analyze what the optimizer agent actually did inside a benchmark's grid cells —
  read every candidate commit, classify the edits, find cross-cell trends, and
  produce a bulleted, citation-bearing observations .md for the paper's discussion
  section. Runs a fan-out of subagents (per-cell extraction → adversarial
  verification → synthesis), with cross-cell stats computed by script rather than
  by agent, so findings are comprehensive and every claim is checkable. Use
  whenever someone
  asks what the optimizer commits are doing, what the optimizer LLM changed or
  learned, whether it reverted or iterated, what trends appear across cells or
  models or harnesses, or asks for optimizer-behaviour observations for the
  write-up — including vaguer forms like "any interesting patterns in these runs?"
  or "why did this cell score higher than that one?"
---

# Analyzing what the optimizer did

An optimizer agent is handed a target harness and a scoring CLI, and left to
improve the harness however it likes. The reward tells us *whether* it improved
things. This analysis answers *what it did* — the part the paper's discussion
section is made of, and the part no metric captures.

The output is a bulleted `.md` of observations, each carrying a citation to a
specific cell and commit. Apaar's requirement, verbatim: *"bulleted, so not very
cloudy — an exhaustive compilation of interesting observations of what the LLM
did... keep it light because I will take that and put it in the discussion
section."* So: exhaustive in coverage, terse in prose. Every claim traceable.

Different people run this across different benchmarks and the results get pooled,
so consistency matters as much as depth. Same procedure, same output shape, same
evidence standard, whether it's tau3 or GAIA or swe-atlas.

## Why this is a fan-out and not one pass

One agent reading twelve candidate repositories does three things badly, and all
three have already happened on this project:

- **It runs out of attention before it runs out of cells.** Cells read late get
  a paragraph; cells read early get a page. The write-up then describes the first
  three cells and calls it a trend.
- **It sees the pattern it went looking for.** A single reader who notices reverts
  early starts reading everything as evidence about reverts, and never notices
  that two cells shipped nothing at all.
- **It reports plausible numbers it never counted.** "Roughly a third of commits
  were reverts" is the kind of claim that reads fine, gets into a paper, and is
  wrong. Nobody recomputes it.

So: extraction is separated from interpretation, every cross-cell number comes
from a script rather than an agent's count, and every candidate observation is
attacked by a verifier that re-reads the artifact before it ships.

An earlier version of this skill also ran the cross-cell interpretation step as
six independent "blind" agents (one per lens, unable to see each other's
findings) plus a completeness critic reading everything at the end. Cut after the
first real run: the blind-lens fan-out reproduced almost nothing that a 20-line
python script computing correlations and keyword hits didn't already surface
faster and for free, and with 16/16 cells covered by step 1 the completeness
critic had nothing left to find. If a future benchmark's corpus is large enough
that step 2 below misses real patterns, bring a lens back — but earn it, don't
default to it.

## Step 0 — Extract the facts deterministically

Run the bundled extractor first. It unpacks each cell's `session.tar.gz`, walks
the candidate git repo, and emits one record per candidate with its message,
diffstat, files touched, and per-partition scores.

```bash
cd harness-engineering-bench
python3 skills/analyze-optimizer-commits/scripts/extract_candidates.py \
    --benchmark <benchmark> --json /tmp/<benchmark>-candidates.json
```

Read the stdout table yourself before dispatching anything. It tells you how many
cells there are, which are marked `[NOT REPORTABLE]`, and where the shipped
candidate sits in each chain. That shapes the fan-out — and it is also your first
sanity check: if the extractor finds two cells where you expected twelve, the
problem is the path or the archive, not the optimizer.

**Only analyze reportable cells, and say how many you dropped.** A cell that
crashed still writes a `finalization.json` with `shipped: false` or reward 0.0,
and its candidate chain is real but truncated — the optimizer was interrupted, not
finished. Mixing those in makes the optimizer look like it abandons work. The
extractor's `reportable` flag encodes the check (shipped, zero error rate,
non-zero token spend); trust it, and report the excluded cells as their own
observation, because *why* cells died is itself a finding.

## Step 0.5 — Find the cells that shipped the seed, before anything else

For every cell, compare the seed's tree hash with the shipped candidate's:

```bash
git --git-dir=<repo> rev-parse <seed-sha>^{tree} <shipped-sha>^{tree}
git --git-dir=<repo> diff --name-only <seed-sha> <shipped-sha>   # ignore __pycache__, .gitignore
```

A cell where these agree shipped the **unmodified seed**, which makes its reward a
measurement of the baseline through the *finalization* path. That is the single most
valuable artifact in the corpus, and it is why this check comes before
interpretation rather than during it.

On tau3's first run, two cells had this property and scored 0.4267 and 0.4511 —
while the seed measured **0.5618** on the same test partition through
`rescore_candidate.py`. Same code, same cases, two scripts, 0.12 apart, against a
floor whose own round-to-round sd was 0.0063. Every per-cell gain computed against
that floor was wrong by roughly the same amount, and the error inverted the
headline: 4 of 14 cells looked like improvements against the bare-path floor, 12 of
14 against the in-path seed.

So: **if the benchmark's pinned baseline was measured by a different script than the
one that scored the candidates, no gain number means anything until that gap is
quantified.** Verify the tree hashes rather than believing a commit message — three
tau3 cells have messages saying "revert" while still carrying behavioural changes,
and one says only "Add .gitignore" while being a total revert in effect.

If no cell shipped the seed, say so explicitly in the write-up: it means the
comparator was never cross-checked in-path, and every gain inherits that risk.

## Step 1 — Per-cell extraction agents (parallel, one per cell)

One agent per reportable cell. Their job is to read and report, not to conclude.
Ask each for:

- A one-line summary of every candidate: what changed, in the optimizer's own
  words plus what the diff actually shows. These differ more often than you'd
  expect, and the gap is a finding.
- The **arc** of the cell: did it explore then converge, thrash, or make one edit
  and stop? Where does the shipped candidate sit in that arc?
- Any candidate whose message claims a measurement (*"regressed 8 of 30
  development cases"*) — quote it. These are the optimizer showing its evidence,
  and they are the most quotable material in the whole corpus.
- Anything that surprised the agent, flagged as such.

Require every item to carry `cell` + 12-char sha. An observation without a
citation cannot be verified later, so it will be dropped — tell the agents this,
so they don't waste effort on unciteable impressions.

Tell them what the artifacts are so they don't go hunting: the extractor JSON has
messages and diffstats, and the full diffs are in the session archive under
`candidates/repository.git` (`git --git-dir=... show <sha>`). Reading actual
diffs matters for at least the shipped candidate and anything the message
describes vaguely — commit messages oversell, and a "comprehensive rewrite of the
retry logic" is sometimes a two-line change.

## Step 2 — Compute cross-cell stats yourself, then draft observations

Don't dispatch agents for this — write a short script (or reuse the snippet in
`references/cross_cell_stats.py`) over the extractor JSON and get, deterministically:
diff size vs. reward, candidate count vs. reward, which knobs got touched and by
how many cells, a keyword sweep for revert-ish and measurement-citing commit
messages. This is the step that used to be six blind lens agents; a script gets
the same numbers in seconds and they're exact rather than eyeballed.

Read those stats alongside the 16 per-cell reports from step 1 and draft the
candidate observation list yourself: one claim per item, a citation (cell + sha)
on every one, and a count wherever the claim implies one. `references/lenses.md`
still lists the angles worth checking for — measurement discipline, unpriced
knobs, inert/cosmetic changes, structural shape, failure-mode targeting, cost
awareness — read it as a checklist for what to look for in the stats and reports,
not as a set of agents to spawn. Note explicit absences too: *"no cell in this
benchmark ever touched a tool-output cap"* is as much a finding as a hit.

This draft list is what step 3 verifies. Nothing in it ships without surviving
that pass.

## Step 3 — Adversarial verification (parallel, one per draft observation)

Every candidate observation gets a verifier whose brief is to **refute it**, not
to confirm it. This is the step that keeps hallucinations out, so frame it that
way explicitly: the verifier re-opens the cited artifact and checks that it says
what the observation claims.

An observation survives only if the verifier cannot refute it. Instruct verifiers
to default to *refuted* when the citation is unclear, when the diff does not show
what the message claims, or when a stated count is off. Failing an unclear claim
is cheap; a wrong claim in a paper is not.

Refute on any of:

- **Citation doesn't support it.** The sha is real, the commit says something
  else. The single most common failure.
- **The count is wrong.** Recompute from the extractor JSON. Never accept a
  tally an agent produced by reading.
- **n=1 dressed as a trend.** "Optimizers prefer X" from one cell is an anecdote.
  Either it gets a count or it gets rewritten as a single-cell example.
- **Intent attributed beyond the evidence.** "The optimizer realized the timeout
  was the bottleneck" requires the optimizer to have *said* so. Otherwise it
  changed a timeout and we don't know why.
- **Confounded comparison.** Cells differ in model *and* harness *and* seed. A
  claim that a model behaves a certain way needs cells that vary only in model.
- **Reward attribution without support.** A candidate that changed X and scored
  higher does not show X caused it, unless the per-candidate partition scores in
  the extractor JSON bracket that specific commit.

Before moving to synthesis, glance at coverage yourself: any reportable cell with
zero surviving observations is worth a second look — silence about a cell usually
means it was skipped, not that it was boring. With every cell run through step 1
this is rarely more than a one-line check, not a reason to spawn another agent.

## Step 4 — Synthesize the observations file

Write `harness-engineering-bench/<benchmark>/OPTIMIZER-BEHAVIOR.md`, using
`references/output-template.md` as the shape. Only verified observations go in.

What makes this file useful to whoever writes the discussion section:

- **Bullets, one claim each, citation inline.** `(opus-5×opencode-r1, a3f2c1d8)`.
  A reader who doubts a bullet can check it in under a minute.
- **Counts stated as counts.** "4 of 12 cells" beats "several cells". Where the
  denominator matters, give it.
- **Quote the optimizer.** Its own commit messages are better evidence than any
  paraphrase, and they're what makes the section readable.
- **Order by how much the reader learns**, not by pipeline stage.
- **A section for what wasn't found.** Absences constrain the claims the paper can
  make, and they're invisible unless written down.
- **Say what's confounded.** One honest line about what the design can't separate
  is worth more than a hedge on every bullet.

Keep it light. If a bullet needs a paragraph of setup, it belongs in the run's
`RESULTS.md`, not here.

## Guardrails worth holding onto

**Reward is not the subject.** It is easy to slide into explaining the scores,
because scores are quantitative and satisfying. The subject is the optimizer's
behaviour. A cell where it did something fascinating and gained nothing is more
interesting than one where it changed a constant and gained 0.02.

**The seed commit is not a candidate.** It's position 0 and it's what everyone
started from. Counting it inflates every denominator.

**"Shipped nothing" is a real outcome.** Cells that ship a revert-to-baseline, or
whose only surviving diff is a `.gitignore`, are among the strongest findings
available — the optimizer had budget, made attempts, measured them, and concluded
the baseline was better. Don't let those cells get filtered out as uninteresting.

**Don't harmonize across benchmarks.** Each benchmark's file describes its own
cells. Pooling is a later step done deliberately, and premature blending hides
the benchmark-specific behaviour that motivated running several.

## References

- `references/lenses.md` — angles to check for in step 2, with what each found on
  tau3 and GAIA. A checklist for drafting observations, not a set of agents.
- `references/cross_cell_stats.py` — the script for step 2: diff-size/reward
  correlation, candidate-count/reward correlation, knob-touch counts, keyword
  sweep. Run it, don't re-derive it by eye.
- `references/output-template.md` — the structure of the observations file, with
  worked example bullets at the right length and citation density.
- `scripts/extract_candidates.py` — deterministic extraction. All counts come
  from here; see its docstring for what it reads and what it deliberately leaves
  to the analysis.
