# What the optimizer did — tau3

16 cells, target model `gpt-5.4-mini`, 4 optimizer models x 2 harnesses each x 2
repeats. 83 candidate commits (seed excluded). Every row below is generated from
`extract_candidates.py`'s JSON, not typed by hand; every behavioral claim in the
findings table cites a `(cell, sha)` you can re-check in under a minute with
`git --git-dir=<cell>/session/candidates/repository.git show <sha>`.

**Read the measurement-path gap section before the gain column of any table
below** — two cells here shipped the unmodified seed, and it scores ~0.12 below
the benchmark's own pinned floor for that same code. Every "gain vs floor" number
in this file inherits that until it's resolved.

## Per-cell

| cell | model | harness | reward | gain vs floor | candidates | shipped pos | shipped sha | knobs touched |
|---|---|---|---:|---:|---:|---|---|---|
| claude-opus-5-opencode-r2 | claude-opus-5 | opencode | 0.6067 | +0.0449 | 8 | 8/8 | `8487bf75508e` | reasoning_effort |
| claude-opus-5-claude-code-r1 | claude-opus-5 | claude-code | 0.5911 | +0.0293 | 5 | 3/5 | `1a0ad193ce59` | retry, timeout |
| claude-opus-5-claude-code-r2 | claude-opus-5 | claude-code | 0.5800 | +0.0182 | 8 | 6/8 | `d8e991280d32` | MAX_TURNS, reasoning_effort, retry |
| claude-opus-5-opencode-r1 | claude-opus-5 | opencode | 0.5733 | +0.0115 | 6 | 6/6 | `2fc25a3fa0d0` | reasoning_effort, retry |
| kimi-k3-kimi-cli-r1 | kimi-k3 | kimi-cli | 0.5378 | −0.0240 | 13 | 13/13 | `ffb61506326c` | tool-output cap |
| kimi-k3-kimi-cli-r2 | kimi-k3 | kimi-cli | 0.5244 | −0.0374 | 3 | 2/3 | `f64128c5e08d` | retry, timeout |
| gpt-5.6-sol-opencode-r1 | gpt-5.6 | opencode | 0.5178 | −0.0440 | 2 | 2/2 | `193644484ba3` | reasoning_effort |
| gpt-5.6-terra-opencode-r1 | gpt-5.6 | opencode | 0.4889 | −0.0729 | 2 | 2/2 | `36211081992f` | reasoning_effort (implicit, see note) |
| gpt-5.6-sol-codex-r1 | gpt-5.6 | codex | 0.4867 | −0.0751 | 2 | 2/2 | `f128802715532` | reasoning_effort (unstated in message) |
| kimi-k3-opencode-r1 | kimi-k3 | opencode | 0.4844 | −0.0774 | 6 | 2/6 | `3d89801a6438` | — |
| kimi-k3-opencode-r2 | kimi-k3 | opencode | 0.4778 | −0.0840 | 7 | 7/7 | `ba2955e328f2` | reasoning_effort (reverted upstream) |
| claude-sonnet-5-claude-code-r1 | claude-sonnet-5 | claude-code | 0.4622 | −0.0996 | 3 | 3/3 | `ad38074ed316` | MAX_TURNS, retry |
| claude-sonnet-5-opencode-r1 | claude-sonnet-5 | opencode | 0.4511 | −0.1107 | 5 | 5/5 | `7179bd048021` | — (shipped = seed + `.gitignore`) |
| gpt-5.6-terra-codex-r1 | gpt-5.6 | codex | 0.4356 | −0.1262 | 4 | 4/4 | `90325a78cefa` | — |
| claude-sonnet-5-claude-code-r2 | claude-sonnet-5 | claude-code | 0.4267 | −0.1351 | 6 | 6/6 | `0031cb45be1f` | — (shipped = seed, byte-identical) |
| claude-sonnet-5-opencode-r2 | claude-sonnet-5 | opencode | 0.3978 | −0.1640 | 3 | 3/3 | `b125bda2614f` | — |

"knobs touched" = any candidate in that cell's chain, not necessarily the shipped
one; a cell can touch a knob and still ship something that doesn't carry it. Two
rows are marked with the sha of the `reasoning_effort` commit even though the
shipped subject doesn't name it — see the findings table for exactly which
message it was buried in.

## Per optimizer model, aggregated

| model | cells | mean reward | cells above 0.5618 floor | range |
|---|---:|---:|---:|---|
| claude-opus-5 | 4 | 0.5878 | 4/4 | 0.5733 – 0.6067 |
| kimi-k3 | 4 | 0.5061 | 0/4 | 0.4778 – 0.5378 |
| gpt-5.6 | 4 | 0.4822 | 0/4 | 0.4356 – 0.5178 |
| claude-sonnet-5 | 4 | 0.4344 | 0/4 | 0.3978 – 0.4622 |

**This ranking should NOT be read as "only opus-5 improved the harness."** Two of
claude-sonnet-5's four cells shipped a harness with zero behavioral difference
from the seed (see next section) — their score is a floor measurement mislabeled
as sonnet-5's output, not evidence sonnet-5 made things worse. With those two
cells removed, claude-sonnet-5's remaining pair is 0.4622 and 0.3978 — still last,
but on n=2 rather than n=4, and the gap to the (mismeasured) floor shrinks once
you compare against the in-path seed value (~0.44) instead of 0.5618.

## Per optimizer harness (opencode vs. each model's native harness)

| harness | cells | mean reward | mean candidates | notes |
|---|---:|---:|---:|---|
| opencode | 8 | 0.5054 | 5.9 | used by every model; widest score spread (0.3978–0.6067) |
| claude-code | 4 | 0.5100 | 5.5 | opus-5 and sonnet-5 only |
| kimi-cli | 2 | 0.5311 | 8.0 | kimi-k3 only; both cells cite measured numbers in commit messages, the only native harness besides claude-code that does |
| codex | 2 | 0.4612 | 3.0 | gpt-5.6 only; both cells have bare one-line subjects with empty bodies on every candidate |

`codex` and `kimi-cli` each have only 2 cells, so treat those rows as descriptive,
not statistically load-bearing.

## Verified behavioral findings

Each row was independently attacked by a second agent instructed to refute it
before it's listed here. Verdict is what survived, corrected where the correction
itself matters.

| cell | sha | finding |
|---|---|---|
| `claude-sonnet-5-claude-code-r2` | `0031cb45be1f` | Shipped tree is byte-identical to the seed (tree hash `65e2c147b655` matches exactly). Reward 0.4267 is a measurement of the unmodified seed through the finalization path. |
| `claude-sonnet-5-opencode-r1` | `7179bd048021` | Shipped tree's only cumulative diff from seed is a 3-line `.gitignore`, never read by the target at runtime. Reward 0.4511 is likewise effectively the seed. |
| `gpt-5.6-sol-codex-r1` | `f128802715532` | Message says only "bound knowledge retrieval and enforce strict tool schemas." Diff also silently raises `reasoning_effort` medium→high and adds new behavioral prompt rules never mentioned. |
| `gpt-5.6-terra-codex-r1` | `90325a78cefa` | Message says "Add concise safeguards for knowledge-base offers." Diff is dominantly (43 of 57 changed lines) a revert of the prior candidate's entire prompt rewrite; the named addition is 5 lines. |
| `claude-opus-5-opencode-r1` | `25afede52a4f` | Adds an `end_conversation` interception: on the model's first attempt to end, injects a fabricated tool-role reply — *"NOT ENDED — the conversation is still open..."* — without calling the real tool. |
| `claude-opus-5-claude-code-r2` | `968131deb0a7` | Adds a `KB_search` result cache keyed on exact arguments. On a repeat query the cached text is returned with a *"you already ran this exact search... do not run it a third time"* prefix; the real tool is never invoked on a hit. |
| `kimi-k3-opencode-r1` | `3d89801a6438` | Shipped because validation beat the seed (0.4267 vs 0.3667) despite its own development score (0.28) being worse than both the seed (0.3333) and the candidate immediately before it (0.36). |
| `claude-opus-5-claude-code-r1` | `1a0ad193ce59` | Largest measured dev gain in the corpus (0.4267→0.5733) is 100% a retry/backoff mechanism on provider rate-limit errors — zero prompt change in the same commit. Message cites "9 cases lost outright to Azure 429s." |
| `kimi-k3-opencode-r2` | `7394ea436c04` | Only `reasoning_effort` reversion in the corpus: *"revert reasoning effort to medium (no gain at high, higher latency)"*. The two candidates immediately upstream had identical dev scores (0.453/0.453) — supports "no gain"; the revert commit itself carries no recorded score, so it restates rather than re-measures. |
| `kimi-k3-kimi-cli-r1` | `ffb61506326c` | Shipped a byte-exact revert to its own earlier candidate `e5e69600fdf0` (empty `git diff` on the target file). Same code scored development 0.80 there and 0.70 here — a 0.10 swing with zero behavioral difference, i.e. same-cell run-to-run noise. |
| `claude-sonnet-5-opencode-r2` | `b125bda2614f` | Bundles a measured revert with a second, unmeasured behavioral change (rewritten stop-marker regex). The bundle's own dev (0.3571) and val (0.38) are both worse than the candidate it partially reverted from (dev 0.4533) and worse than the seed (val 0.4067). Shipped anyway. |
| `kimi-k3-opencode-r2` | `ba2955e328f2` | Commit message calls itself a test — *"baseline prompt + conversation-ended loop fix (isolate loop-fix effect)"* — reverts every accumulated prompt change to the byte-identical seed prompt while keeping two earlier code-level changes. An explicit ablation shipped as the final answer. |

Three drafted claims did **not** survive and are recorded so the correction isn't
lost: (1) "all four opus-5 cells cite measured numbers" — false, only the two
`claude-code`-harness ones do; the two `opencode` ones have empty bodies like
gpt-5.6's cells. (2) A knob-touch count of "6 candidates / 5 cells" for
`reasoning_effort` — the real count is 7 candidates / 6 cells (see the per-cell
table above, which reflects the corrected count). (3) A claim that
`claude-opus-5-opencode-r2`'s one-tool-call-per-turn enforcement was undisclosed —
it's real, but it's named in the commit title itself ("one action per turn") and
explained in a code comment, so "silent" was wrong.

## The measurement-path gap

Two cells above (`claude-sonnet-5-claude-code-r2`, `claude-sonnet-5-opencode-r1`)
shipped a harness with zero behavioral difference from the seed, confirmed by
exact git tree-hash comparison. Their rewards (0.4267, 0.4511) are therefore
direct measurements of the unmodified seed through the finalization path — the
same path every cell in the tables above was scored through.

The benchmark's pinned `baseline_reward` is **0.5618**, measured on the identical
seed through a different script (`rescore_candidate.py`, bare `harbor run`, no
gateway/sidecar). Same code, same 150 held-out cases, ~0.12 apart.

A same-code repeat elsewhere in the corpus bounds how much of that could be
ordinary noise rather than a systematic path effect: `kimi-k3-kimi-cli-r1`'s
byte-exact revert (`e5e69600fdf0` → `ffb61506326c`) scored 0.80 and 0.70
development on identical code — a 0.10 swing with no path change at all. That
doesn't resolve the gap either way; a rescore probe aimed at settling it directly
hit budget-exhausted API errors mid-run and was inconclusive.

**Every "gain vs floor" column above is provisional until this is settled.**
Against the in-path seed instead of the 0.5618 floor, most non-opus cells look
like modest improvements rather than regressions — the ranking direction for
opus-5 doesn't change, but the sign for everyone else might.

## What this cannot separate

- **Model, harness, and seed aren't independently varied.** A "model X vs model
  Y" claim is really about that model+harness+seed combination. The
  per-harness table above is the closest this gets to isolating harness effect,
  and even there codex/kimi-cli have only 2 cells each.
- **The measurement-path gap**, described above, affects every gain number but
  not the behavioral findings (those describe what happened, independent of
  which script scored it).

## Method

`extract_candidates.py` unpacked each cell's `session.tar.gz` and walked the
candidate git history. 16 per-cell agents read the actual diffs and commit
history, not just messages. All tables above are regenerated from that JSON by
script, not hand-typed. 15 candidate findings were drafted from those reports;
each was independently attacked by a verifier told to default to "refuted" on any
unclear citation or count. 12 survived as stated, 3 required the corrections
noted above.
