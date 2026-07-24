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
evals run --backend B --evaluation-set S --partition P --detach   # returns job_id
evals status JOB_ID                    # poll the job (evals status = global view)
evals result JOB_ID                    # durable result once finished
evals list --sort score --desc        # every past result, one row each
evals diff BASELINE_ID CANDIDATE_ID   # which cases improved / regressed
evals cases ID --sort score           # per-case scores for one result
evals trace ID CASE_ID                # trace summary + artifact files for a case
evals trace ID CASE_ID --span 3       # one span of the trace, char-windowed
```

Evaluate the baseline first: a candidate only counts as an improvement against
a measured baseline on the same case selection.

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
- Scores may be noisy: replicate an important comparison before acting on it.
