# DABstep results

Yash Maurya, 2026-07-30.

## Baseline

Pinned at **0.596 ±0.026** on the 66-case held-out partition. K=3 rounds
(0.621 / 0.561 / 0.606), 198 trials, zero exceptions,
`scripts/rescore_candidate.py --seed`.

Seed run-to-run spread is ~0.05, so treat any single-round delta below that as
unresolved.

## First optimizer cell

opus-5 + claude-code, 1h 58m, `shipped: true`.

| | |
|---|---|
| Held-out reward | **0.879** (58/66) |
| Baseline | 0.596 (39/66) |
| Absolute gain | **+0.283** |
| Normalized gain `(score-base)/(max-base)` | **0.700** |

The gain is ~11x the baseline's own sd.

### Against the other benchmarks

| benchmark | baseline | best | normalized gain |
|---|---|---|---|
| dabstep | 0.596 | 0.879 | 0.70 |
| officeqa | 0.341 | ~0.65 ◇ | ~0.47 |
| browsecomp-plus | 0.462 | ~0.52 ◇ | ~0.11 |

◇ Varun's mid-run figures from the 07-29 status post. dabstep's is a completed
cell against a K=3 pin.

### Trajectory, 6 candidates

| candidate | dev | validation |
|---|---|---|
| seed | 0.545 | |
| 57433be | 0.970 | |
| 872a3f6 | 0.970 | 0.970 |
| f6abf08 | 1.000 | 0.909 |
| 0bb8ee4 | | 0.909 |
| 60aeb15 shipped | | 0.909 |

First edit captured most of the gain. Validation flat at 0.909 across the last
three, so selection was not chasing noise. Full case budget consumed on both
partitions.

### Per-case cost

| | seed | shipped |
|---|---|---|
| inference requests | 24.2 | 4.3 |
| input tokens | 636k | 16k |
| wall | 234s | 28s |

**40x fewer input tokens, 8x faster.** Cost is not scored, so nothing forced this.

### Spend and reliability

| | |
|---|---|
| Optimizer (opus-5) | 158 requests, 14.5M in / 118k out, 0 errors |
| Evaluation (deepseek-v4-flash) | 2 584 requests, 35.6M in / 1.68M out |
| Upstream 429s | 101/2 584 = 3.9%, all absorbed by `max_retries: 4` |
| Total metered | 51.7M tokens |

## What the optimizer changed

+1 022 lines across 5 commits: `dabstep_lib.py` (535), `agent.py` (+332),
`tests/test_agent.py` (+174).

**57433be, fee engine + playbook.** Diagnosed all 15 seed failures as fee-rule
questions with two undocumented semantics: a null rule field is a wildcard, and a
transaction is charged the sum of every rule it matches. Seed returned 78.55 for
148.61, 385.79 for 642.39, 329.88 for 1120.77, all the same missing summation.
Wrote a 26-function rule engine, uploaded at `setup()`. Validated offline against
all 33 development answers, 28 exact checks. Upload failure degrades to the plain
playbook instead of burning turns on ImportError. Memoised rule matching on
(merchant, month): one pass over a merchant's year went 22s to 0.6s.

**872a3f6, eager load and mean-not-max.** Its own library loaded lazily, so a trial
reading `L.merchant_category_codes` before any call got an empty dict and spent 15
of 24 turns introspecting. Also fixed an MCC ranking graded on mean, not max.

**f6abf08, one-directional narrowing.** A trial "talked itself out of it at turn
37" on rule-narrowing questions. Added `narrowed_out_merchants()` so there is a
helper to call rather than a judgement to make, and recorded why `patch=` cannot
express it.

**0bb8ee4, corrected its own docs.** Had listed one country set for all three
country columns; `acquirer_country` differs from `ip_country`, which is why
intracountry holds for 17.84% of rows.

**60aeb15, template gaps.** Cases the 33 dev tasks do not exercise, including
that 25 of 30 merchants have no transactions.

Notable: it never touched `MAX_TURNS`. Every change was structural, each driven by
a named failing case (adyen/1440, adyen/2571).

## Setup decisions worth knowing

**Split.** 165 of 450 as 33/66/66, keeping the dataset's own 16/84 difficulty
ratio. Matches gaia and browsecomp-plus. A reweighted 44/56 variant was built and
dropped: it existed because a proportional draw looked like it would seed near the
floor (published frontier is ~14.5% on hard), but the seed reads 0.596, so the
premise was wrong.

**Turn cap 40, not 24.** At 24 the seed truncated 18 of 33 dev cases and
force-answered 17. Measured 9.1s per turn against the declared 1800s clock. 40
matches swe-atlas-qna and cuts truncation to 3 of 33. Sized for tokens, not the
clock: history is resent every turn, and filling 1800s would be ~197 turns.

**Seed differs from the other five deliberately.** No `read_image` (the pinned
target is text-only; officeqa ships one and every call 400s). Empty model turns
nudge instead of raising, matching gaia; four other seeds still raise. Token
accounting in `finally`, so a crashed trial still reports usage.

## Open

- Re-pin if `agent.py` changes at all. 0.596 belongs to the turn-cap-40 seed.
- One cell only. No claim about model or harness ranking until a second lands.
- Remaining headroom is 0.12, so further gains will be hard to resolve.
- officeqa carries the same 24-turn cap on the same 1800s clock. Cost is not
  scored, so raising it is an unpriced win. Worth grepping its traces for
  `forced_final` before its 0.341 → 0.6-0.7 numbers are reported.

## Artifacts

`runs/dabstep/opus-claude-code/jobs/2026-07-30__09-49-22/task__EQ6W3ZE/verifier/`
holds `finalization.json`, `reward.json`, `experiment.html` and `session.tar.gz`.
The candidate repo is inside the archive at `session/candidates/repository.git`;
`git log --all` lists every candidate.
