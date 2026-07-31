# Harness engineering benchmarks

`harness-engineering-bench` contains end-to-end benchmarks for automatically
improving the harness of an agent or the code of a component used to build
agents. Each leaf directory pairs one editable target program with one immutable
Harbor dataset and compiles them into an outer Harbor optimization task.

The benchmark definitions intentionally keep three boundaries visible:

- `target/` is the program the optimization agent may edit.
- `partitions/` pins the cases and the development/validation/test split.
- `build.yaml` is trusted configuration: model, evaluator, access policy,
  budgets, and final scoring.

In each benchmark, the complete development tasks and attachments are mounted
read-only for the optimization agent. Development evaluations expose per-case
results and complete Harbor trial records, including exact failures and
target-agent logs. Validation remains aggregate-only, and
test is reachable only by the trusted final verifier.

The paper-era benchmark stack remains available on the `paper/v1` branch and
the `paper-v1` tag. New benchmarks should use this Harbor-native layout.

## First run in a fresh checkout: fetch the task data

Two benchmarks keep their task definitions out of git — officeqa's 246 and
browsecomp-plus's 830 directories are hundreds of megabytes and thousands of
files, and committing them once bloated the repository and timed out the
pre-commit secret scan. A fresh checkout therefore cannot run either benchmark
until the data is fetched, and the failure is not obvious from the error.

Ask what is missing, and what to run for it:

```bash
python3 scripts/task_data.py            # status per benchmark
python3 scripts/task_data.py --check    # exits 1 if anything is missing
```

It reports state and names the command; it deliberately fetches nothing, because
the fetchers are slow, network-bound and replace their output directory. The
other three benchmarks pin a registry digest and need nothing local.

A count that disagrees with the expected total means a partial fetch, which is
worth taking seriously: a half-vendored directory passes config validation and
then scores a subset of the benchmark without saying so.

## Benchmarks

Promoted benchmarks live at the top level. Task sets still under review live in
`candidates/`; we work through the list in the paper's `benchmark-scoping.md` and
promote a task set to the top level once it is ready.

### Promoted

| Benchmark | Editable target | Dataset | Split |
| --- | --- | --- | --- |
| [GAIA baseline](gaia/baseline/) | Tool-using Responses API agent | Harbor `gaia/gaia` | 20% / 40% / 40% |
| [OfficeQA baseline](officeqa/baseline/) | Grounded document-QA agent | Treasury Bulletin corpus | 20% / 40% / 40% |
| [SWE-Atlas-QnA baseline](swe-atlas-qna/baseline/) | Codebase investigation agent | Harbor `scale-ai/swe-atlas-qna` | 20% / 40% / 40% |
| [tau3 baseline](tau3/baseline/) | MCP customer-service agent | Harbor `sierra-research/tau3-bench` | 20% / 40% / 40% |
| [BrowseComp-Plus baseline](browsecomp-plus/baseline/) | Fixed-corpus deep-research agent | Pinned local Harbor tasks | 20% / 40% / 40% |
| [Terminal-Bench baseline](terminal-bench/baseline/) | Shell-loop terminal agent | Harbor `terminal-bench/terminal-bench-2-1` | 20% / 40% / 40% |

**`swe-bench-pro/` is at the top level but is not promoted, and its numbers are
not comparable to the five above.** It predates the normalization pass those five
went through and still differs on most of it: case budgets are 1x the partition
size rather than 4x, the held-out target has no `n_attempts: 3` / `mean`
override so it is scored once, there is no pinned `baseline_reward` (and
`score_baseline: true` adds a second full held-out pass), the agent clock runs at
0.6x the declared case timeout, gateway request and token caps are 20-30x tighter
than the sizing convention, there is no `agent_env` block, and telemetry goes to
its own W&B project. Treat it as a work in progress: launching it will produce a
number, but not a measurement of the same quantity. `CONFIGURATION.md` documents
the conventions it is missing.
