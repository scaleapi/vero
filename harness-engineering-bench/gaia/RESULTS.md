# GAIA results

Yash Maurya, 2026-07-30. First cell, opus-5 + claude-code, 1h 44m, `shipped: true`.

## Headline

| | |
|---|---|
| Held-out reward | 0.6768 (67/99 trial-equivalents, 66 cases x 3) |
| Pinned baseline | 0.6205 |
| Absolute gain | **+0.056** |
| Baseline sd (own 3 rounds) | 0.052 |
| Normalized gain | 0.148 |

**Not resolvable.** `CONFIGURATION.md` already says to treat any GAIA delta under
~0.1 as noise. +0.056 is 1.1x the seed's own run-to-run spread, so this cell
supports no claim that the optimizer improved anything.

For contrast, dabstep's first cell was +0.283 against a 0.026 sd, which is 11x.

## Trajectory

| candidate | development | validation |
|---|---|---|
| seed | 0.8182 | 0.6818 |
| d228b48 research tooling, review gate, clock discipline | 0.9091 | |
| e666c10 harden retrieval, evidence-aware review gate | 0.8788 | 0.6818 |
| 0032e67 recover answers from noisy finals, spend idle clock | 0.9091 | 0.6970 |
| 04ca727 raise reasoning effort to high (shipped) | | 0.7576 |

Two things to read off this.

**The shipped candidate is a one-word diff.** `REASONING_EFFORT = "medium"` ->
`"high"`, 1 line in `agent.py`. It moved validation 0.6970 -> 0.7576, or +0.061.
The three structural candidates before it, together 1,200+ lines across `agent.py`
and three new helper modules, moved validation +0.015 total.

**Validation overstates held-out by 0.081** (0.7576 against 0.6768). Selection
had three validation readings inside 0.076 of each other to choose between, which
is under the noise floor, so the pick was close to arbitrary.

## The unpriced knobs

Both of GAIA's monotone parameters got turned up, and neither is charged for.

- `MAX_TURNS` 24 -> 40 in the first candidate, the same knob and the same two
  values as dabstep. Measured here: mean case wall 169s, median 101s, max 537s
  against a 600s clock, so turns were not the binding constraint after the change.
- `REASONING_EFFORT` medium -> high in the shipped candidate, which is the whole
  reported gain.

Cost is not in the reward, so raising reasoning effort is free score. Its own
commit message says so plainly: "latency is not the binding constraint." Held-out
spend was 83.3M input tokens for 198 trials, 1.26M mean per case, 34 requests per
case. A cell that buys its gain this way is measuring the seed's parameter choices,
not the optimizer's harness engineering.

## Reliability

| | |
|---|---|
| Upstream errors | 4 / 5,850 requests = 0.07% |
| Evaluations terminated | 0 |
| `error_rate` per evaluation | 0.0 on all 9 |
| Case budgets | exactly exhausted, 132/132 dev and 264/264 validation |
| Held-out spend | 2,245 requests, 83.3M in (59.1M cached), 2.0M out |

GAIA runs on `gpt-5.4-mini`, not `fireworks_ai/deepseek-v4-flash`, so it sits
outside the shared Fireworks bucket that root-caused this morning's blocker. It is
the one benchmark in the suite the rate limiting does not touch. dabstep ran 3.9%
429s on the same day.

## A false start worth recording

The first launch died in 4m 41s: `UnknownApiError`, the optimizer's first request
returning `502 upstream inference request failed`, W&B showing `producer=11/11`
errors and nothing else attempted. Cause was `--environment modal`. The gateway
runs wherever the optimizer runs, and on Modal it cannot resolve an internal-only
upstream host. `launch_cell.sh` said "normally `modal`"; that header is now fixed.

## Open

- **One cell, and its outcome is a null.** A second cell decides whether GAIA has
  reachable headroom at all or whether 0.62 is close to what this seed can do.
- **Resolving anything on GAIA needs more than a single pass.** At sd 0.052 on 66
  cases, a real +0.05 effect needs roughly 3 rounds per side to separate from
  noise. Budget that before promising a number.
- **Decide whether the parameter knobs count.** If `REASONING_EFFORT` and
  `MAX_TURNS` are fair game, GAIA's reported gain is mostly a spend increase, and
  the same is true of the other seeds that hardcode both. If they are not, they
  belong outside the editable surface. This is a suite-wide question, not a GAIA
  one, and it is the second time it has come up after dabstep's turn cap.

## Artifacts

`runs/gaia/opus-claude-code/jobs/2026-07-30__16-52-10/task__4FfKikj/verifier/`
holds `finalization.json`, `reward.json`, `experiment.html` and `session.tar.gz`.
Candidates are in the archive at `session/candidates/repository.git`; per-candidate
scores are in `session/evaluations/*/evaluation.json`.
