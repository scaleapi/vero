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
evals plan                             # what you may run, rules, remaining budget
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

## Blocking vs detached

By default `evals run` **blocks** until scoring finishes and returns the result
— one call, nothing to track. Evaluations can take many minutes; that is
expected, so let it block. This is what you want almost always.

Use `--detach` **only** to run several evaluations concurrently: it returns a
`job_id` immediately instead of blocking. Then `evals wait JOB_ID` blocks until
that job finishes and prints its result; or poll `evals status JOB_ID`, which
now also reports `elapsed_seconds` (and `requested_cases` for a subset) so you
can see it is progressing. Never end your turn to "wait" for a detached job —
nothing resumes you.

## Run options worth knowing (`evals run --help` for all)

- **`--start N --stop M`** or repeated **`--case-id ID`** — score a *subset*.
  Iterate on a handful of cases (cheap, fast) before spending budget on the full
  set.
- **`--seed N`** — fix the seed to *reproduce a noisy comparison exactly* (the
  cleanest way to "replicate before acting").
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
- Confusing ids: evaluation ids come from `evals list`; case ids from
  `evals cases`; job ids only from `evals run --detach`.
- Scores may be noisy: replicate an important comparison before acting on it
  (`--seed N` reproduces one exactly).
- Ending your turn to "wait" for a `--detach`ed job — nothing resumes you. Block
  on `evals run` instead, or keep polling `evals status` within your turn.
- Finishing without `evals submit`: your best candidate must be nominated
  deliberately, or a fallback (auto-best, then your last commit) ships instead.
- Optimizing accuracy while ignoring per-case latency: a slower candidate can be
  stopped at the wall budget and score the failure value.
