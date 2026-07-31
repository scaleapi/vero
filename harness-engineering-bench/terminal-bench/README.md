# Terminal-Bench 2.1

Candidate Family A benchmark: the editable target is a terminal agent, the domain
is long-horizon work in a command line, and each task's own tests decide pass or
fail, so the reward is a pass rate.

**Status: measured and pinned.** Target model `xai/grok-build-0.1`, held-out
baseline **0.2600 ±0.0176** (K=3, n=100). Recorded in `runs/BASELINES.md` and
`CONFIGURATION.md`, reproducible with `python3 runs/recompute.py`.

One number to know before comparing a candidate against it: 18 of 108 baseline
trials ended in `AgentTimeoutError`, a 20% exception rate against 0.3-3.5%
elsewhere in the suite. The seed allows 40 steps at a 300s per-command cap while
48 of 89 tasks declare a 900s budget, so three slow commands exhaust the clock.
That is real headroom, but a sixth of the baseline's failures are wall-clock rather
than capability.

## The dataset

89 tasks, Apache-2.0, on the Harbor Hub, pinned by content digest:

```
terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a
```

Registry-sourced like GAIA, tau3 and SWE-Atlas-QnA, so **nothing needs vendoring**
and a fresh checkout can run it — unlike OfficeQA and BrowseComp-Plus, whose task
data lives outside the repository.

Re-export the tasks with `harbor dataset download terminal-bench/terminal-bench-2-1
--export -o <dir>`; the exported layout is what `scripts/partition_dataset.py`
reads.

### Use 2.1, not 2.0

Same 89 tasks. 2.1 repaired 28 of them: external dependencies that had changed
since the benchmark was built, resource budgets too tight for a valid solution to
finish, and tasks whose instructions did not match their tests. Those are exactly
the failures that would show up in our runs as zeros no harness change can fix,
and would be misread as "this benchmark has no headroom".

### Not FrontierBench, and not a mix of the two

FrontierBench was considered. Its own `dataset.toml` declares it
`terminal-bench/terminal-bench-3` — it is Terminal-Bench's successor, not an
independent benchmark, and its tasks are namespaced `terminal-bench/*`. Task names
overlap on exactly one (`gpt2-codegolf`), so pooling would be legitimate, but two
things argued against it:

- **Cost.** FrontierBench's median declared agent budget is 2.0 hours against
  Terminal-Bench's 0.2, and its expert time estimates sum to 515 hours across 74
  tasks. It is roughly an order of magnitude more expensive per task.
- **Churn.** It was published 2026-07-22 and is still moving; its README warns
  that the reference solutions may flake. Terminal-Bench 2.1 exists precisely
  because 2.0 shipped 28 broken tasks, which is the same risk one release earlier.

It remains a reasonable second benchmark later, or a source of extra held-out
cases if 36 proves too few.

### Unfiltered, deliberately

An earlier plan filtered to software-engineering and ML categories. Rejected:
it cuts the set to about 50 tasks (a 10/20/20 split), and deciding which
categories count as in-domain is a selection knob that would then have to be
defended. Taking all 89 removes the argument entirely.

The set is broad but code-shaped: `software-engineering` 26, `system-administration`
9, `scientific-computing` 8, `security` 8, `data-science` 8, `debugging` 5,
`file-operations` 5, `model-training` 4, `mathematics` 4, `data-processing` 4,
`machine-learning` 3, plus six single-task categories. Difficulty is `medium` 55,
`hard` 30, `easy` 4.

## The split

17 / 36 / 36, generated deterministically:

```bash
uv run --project vero --with 'harbor[modal]==0.20.0' \
  python harness-engineering-bench/scripts/partition_dataset.py terminal-bench \
  --tasks-dir <exported-tasks> --output-dir harness-engineering-bench/terminal-bench/partitions \
  --fetch-registry
# verify later without regenerating:
#   ... terminal-bench --tasks-dir <exported-tasks> --output-dir <partitions> --check
```

**89 does not divide 1:2:2.** Validation and test hold the exact two-fifths each
and development absorbs the rounding loss, because development is the optimizer's
own full-disclosure search set while test is the measurement.

**Stratified by category *and* difficulty.** Category alone looks sufficient and
is not. Measured:

| stratified by | dev hard | val hard | test hard |
| --- | --- | --- | --- |
| category only | 29% | 28% | **42%** |
| category + difficulty | 35% | 31% | 36% |

The dataset is 34% hard. Category-only stratification hands the optimizer a
noticeably easier search set than the set it is scored on, which biases every
measured improvement downward and confounds it with difficulty. The many singleton
strata that category×difficulty creates are harmless — allocation floors per
stratum then fills to the exact target by largest remainder.

## Sizing, and the one structural difference from other benchmarks

**Terminal-Bench declares a different agent timeout per task** — 600s to 12,000s,
48 of 89 at 900s, 13 at 3600s. Every other benchmark here declares one value for
the whole set. This needs no special handling: vero passes Harbor a single ratio
(`case_timeout_seconds / task_agent_timeout_seconds`) which Harbor applies to each
task's *own* declared budget, so keeping that pair equal gives every task exactly
the clock its author intended. Do not "fix" it by making them differ.

Derived from the split and the declared budgets, at `max_concurrency: 24`:

| quantity | value | derivation |
| --- | --- | --- |
| worst-case finalize wall | 36,000s | 108 case-runs ÷ 24 = 5 waves × slowest test task (7,200s) |
| widest single search eval | 24,000s | 36 validation ÷ 24 = 2 waves × slowest validation task (12,000s) |
| `timeout_seconds` | 43,200 | above worst-case finalize |
| `verifier_timeout_seconds` | 64,800 | finalize + a `rescore_top_k: 3` validation pass |
| optimizer `BASH_MAX_TIMEOUT_MS` | 28,800,000 | above the widest single eval, so one evaluation fits in a single blocking foreground call |

## The seed agent

`baseline/target/`, `terminal_bench_agent.agent:TerminalBenchAgent`. A shell loop:
one command per turn via function calling, linear history, a step budget. Grading
runs each task's own tests against the container's final state, so there is nothing
to submit.

It is written into the target repo rather than pulled in as a dependency because
`agent_repo: target` is the tree the optimizer edits — an installed package would
leave it able to change a config and nothing else.

**It uses function calling, and an earlier version did not.** The first draft
followed mini-SWE-agent in parsing a fenced ```bash``` block out of a plain reply.
That silently filtered the *target model* population: measured against a real task,
`mistral/devstral-small-latest` and `anthropic/claude-haiku-4-5` both replied
conversationally and emitted no block, because they are trained to call tools.
Either would have scored near zero for protocol reasons rather than capability.

Worse, it handed the optimizer a large gain for repairing a handicap the seed's
author introduced, which measures nothing about harness engineering. After the
switch all eight candidate models drive the loop, and the two that had failed open
by inspecting the environment rather than guessing — better first moves than some
that passed before.

**Keep design rationale out of `target/`.** The optimizer reads that tree, so
notes about what is weak or improvable are hints, and turn the benchmark into an
instruction-following test. Reasoning about the seed's limitations belongs here.

It fails closed on credentials, following the BrowseComp-Plus fix: `OPENAI_*` in
the eval container can point at the unmetered upstream, so a fallback would bypass
metering and the allow-list while still appearing to work. 15 tests cover the tool
schema, argument parsing, truncation, the credential guard, and the run loop.

### Verified

`vero harbor build` produces the expected artifacts: only `harbor-test` carries
`n_attempts: 3` / `aggregate: mean` (development and validation stay at 1/best),
W&B routes to the shared project under group `terminal-bench`, and the gateway
allow-lists hold the target model for evaluation and finalization.

One gotcha: the compiler snapshots the target with `git archive HEAD:<path>`, so
**the target must be committed before it will compile**. An uncommitted target
fails with `not a valid object name`, which says nothing about the real cause.

## Target model

`xai/grok-build-0.1`, chosen from a measured probe (seed harness, all 17
development tasks, one round each):

| model | solved | reward | $/Mtok* |
| --- | --- | --- | --- |
| `xai/grok-build-0.1` | 6/17 | 0.3529 | 0.380 |
| `azure_ai/gpt-5-nano` | 2/17 | 0.1176 | 0.044 |
| `xai/grok-4-1-fast-reasoning` | 2/17 | 0.1176 | 0.095 |
| `gemini/gemini-2.5-flash-lite` | 0/12 | 0.0000 | 0.049 |

\* 90% cache-read + 10% output, matching our measured token mix. Gemini also
crashed 5 of 17 trials returning bodies the SDK could not parse.

It costs ~8.6x the cheapest candidate, which is the right trade: what a benchmark
needs is a baseline with headroom, not reward per dollar. 0.26 sits near officeqa's
0.3412, where the best cell moved +0.41; 0.12 is swe-atlas-qna territory (0.0676),
the least informative cell in the suite.

Deliberately **not** a Fireworks model. The shared per-minute generated-token quota
is what limits how many cells run at once, so a target off that provider does not
contend with the officeqa or browsecomp-plus grids.

## Spread, and why n_attempts stays at 3

36 held-out cases is the smallest test partition in the suite and was expected to
be noisy. Measured sd is **0.0176** -- second tightest of the six, behind only
tau3's 0.0099, and tighter than officeqa at 99 cases (0.0330) or gaia at 66
(0.0524). Each task ships its own deterministic tests, so no LLM judge or user
simulator rides on the score. Raising `n_attempts` to 5 was on the table before
this was measured; it is not needed.

## Known seed bug

`UnicodeDecodeError` killed 4 baseline trials: a command emitting undecodable bytes
ends the trial instead of degrading to truncated output. Unlike the timeouts, this
is a bug rather than headroom. Fixing it changes the seed and therefore requires
re-pinning the baseline, so it is recorded rather than fixed.

## Sources

- Terminal-Bench 2.1 announcement — https://www.tbench.ai/news/terminal-bench-2-1
- Dataset — https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1
- Repository — https://github.com/harbor-framework/terminal-bench
- FrontierBench (considered) — https://github.com/harbor-framework/frontier-bench
