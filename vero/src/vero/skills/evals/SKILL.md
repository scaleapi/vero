---
name: evals
description: >-
  Run evaluations and navigate their results with the `evals` CLI over the
  read-only `.evals/` directory. Use whenever you need to score a candidate
  program, compare two candidates case by case, inspect why a case failed, or
  check what you are allowed to evaluate and how much budget remains.
---

# evals: run evaluations and navigate their results

Your program is scored by a trusted evaluator you cannot see into. Everything
you *are* allowed to see lands in the read-only `.evals/` directory, and the
`evals` CLI is the structured way to work with it. `evals --help` lists every
subcommand.

## The loop

```bash
evals plan                             # backend+partition pairs, sizes, budget
evals run --backend B --evaluation-set S --partition P   # BLOCKS, returns the result
evals list --sort score --desc        # every past result, one row each
evals diff BASELINE_ID CANDIDATE_ID   # which cases improved / regressed
evals cases ID --sort score           # per-case scores for one result
evals trace ID CASE_ID                # trace summary + artifact files for a case
evals trace ID CASE_ID --span 3       # one span of the trace, char-windowed
evals submit --version COMMIT         # nominate your best candidate to ship
```

Evaluate the baseline first: a candidate only counts as an improvement against
a measured baseline on the same case selection.

**Results are persisted, so printed output is disposable.** Before `evals run`
returns, the complete record — overall metrics, every per-case score, and the
trial artifacts — is written under `.evals/results/` and stays there for the rest
of the run. So piping through `head`/`tail` loses nothing recoverable: use
`evals list` to find any past evaluation and `evals show` / `evals cases` /
`evals diff` to read it back. Never re-run an evaluation just to recover a number
you truncated — it is on disk.

## Blocking vs detached

By default `evals run` **blocks** until scoring finishes and returns the result
— one call, nothing to track. Evaluations can take many minutes; that is
expected, so let it block. This is what you want almost always.

Use `--detach` **only** to run several evaluations concurrently: it returns a
`job_id` immediately instead of blocking. Then `evals wait JOB_ID` blocks until
that job finishes and prints its result; or poll `evals status JOB_ID`, which
now also reports `elapsed_seconds` (and `requested_cases` for a subset) so you
can see it is progressing. To wait on two jobs, wait the first, then the second.

Run every `evals` call in the **foreground**. If you are a headless single-shot
run, nothing can wake you: putting a long call in a background task, scheduling
a wake-up, or promising to "report back when it finishes" ends the run right
there. A call that blocks for half an hour is working correctly.

## Run options worth knowing (`evals run --help` for all)

- **`--start N --stop M`** or repeated **`--case-id ID`** — score a *subset*.
  Iterate on a handful of cases (cheap, fast) before spending budget on the full
  set. A range past the end of the partition is rejected, so read the `cases`
  column of `evals plan` for the partition's size rather than guessing.
- **`--seed N`** — backend-dependent, and several backends reject it outright
  (you get an `invalid evaluation request` naming the reason). Where it is
  refused, replicate a noisy comparison by re-running the *identical selection*
  instead — same `--start/--stop` or same `--case-id` set.
- **`--max-concurrency`**, **`--case-timeout`**, **`--timeout`** — override
  parallelism and wall budgets for a run. Note the final held-out evaluation
  always uses the configured per-case budget, so don't loosen `--case-timeout`
  in search or your slow candidate will look fine, then get stopped on test.

## The `.evals/` tree

```
.evals/
├── README.md, manifest.json
├── plan.json     # evaluations you may invoke: selection rules, disclosure, budget
├── results/      # past evaluation results   -> evals list / show / cases / trace / diff
│   ├── index.json
│   └── <digest>/evaluation.json, cases/<case>/result.json, artifacts/
├── tasks/        # exposed task resources    -> evals tasks
└── candidates/   # prior program versions (Git refs for `git show` / `git diff`)
```

Everything is plain JSON and files — `evals` is navigation and aggregation;
read individual artifact or task files directly with your file tools once a
command hands you the path.

## Disclosure

Each result is projected to an authorized disclosure level before it reaches
you: `full` (per-case results, traces, artifacts), `aggregate` (overall metrics
only — `evals cases` will refuse), or `none` (bare acknowledgement). Budgets
are enforced by the evaluator; `evals plan` shows what remains.

## Common mistakes

- Comparing evaluations run on different case selections — deltas are then
  case-sampling noise, not signal. Use `evals diff`, which matches by case id.
- Claiming an improvement before the baseline's own evaluation finished.
- Dumping whole traces into context. Go metadata-first: `evals trace` summary,
  then `--span N`, then windowed chars; artifacts are files — `grep` them.
- Re-running an evaluation because you truncated its output. Every result is on
  disk under `.evals/results/`; read it back with `evals show` / `evals cases`.
- Confusing ids: evaluation ids come from `evals list`; case ids from
  `evals cases`; job ids only from `evals run --detach`.
- Mismatching `--backend` and `--partition`. Each partition is served by exactly
  one backend, and asking a different one is refused as `evaluation denied` — a
  denial is deliberately opaque, so it will not tell you that is the reason. Take
  both from the same row of `evals plan`.
- Scores may be noisy: replicate an important comparison before acting on it, by
  re-running the identical case selection.
- Backgrounding the wait. Putting `evals run`/`evals wait` in a background task
  (or scheduling a wake-up) looks like waiting but ends a headless run: the
  notification you are counting on never arrives. Block in the foreground.
- Finishing without `evals submit`: your best candidate must be nominated
  deliberately, or a fallback (auto-best, then your last commit) ships instead.
- Optimizing accuracy while ignoring per-case latency: a slower candidate can be
  stopped at the wall budget and score the failure value.
