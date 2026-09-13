# HarnessOpt-Bench: Evaluating LLMs at Harness Optimization

[![Paper](https://img.shields.io/badge/arXiv-2608.06301-b31b1b.svg)](https://arxiv.org/abs/2608.06301)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Built on VeRO](https://img.shields.io/badge/built%20on-VeRO-2b6a4f.svg)](../vero/)

An LLM's usefulness in an agentic system depends on its **harness** as much as
on its weights: the prompts, tools, control flow, memory and orchestration code
around it. HarnessOpt-Bench measures how well frontier LLMs improve such a
harness themselves, under expensive and stochastic evaluation.

An **optimizer**, an LLM paired with a coding harness, receives a target agent's
seed harness, graded evaluation feedback and a fixed evaluation budget. It edits
the harness and nominates a final candidate, which is scored by its normalized
gain over the seed on a held-out test partition it never sees. A trusted
execution environment enforces that boundary, meters the target agent's
resource use, and keeps every candidate version for audit.

![Optimizer sandbox, trusted evaluation server and model gateway, evaluation sandboxes](docs/figure1-architecture.png)

Every benchmark here is one optimization task: an editable target agent, an
immutable Harbor dataset with a pinned development / validation / test split,
and a `build.yaml` that VeRO compiles into an outer Harbor task. The optimizer
runs inside that task with read access to the development cases, aggregate-only
access to validation, and no access to test.

- **Paper**: [HarnessOpt-Bench: Evaluating LLMs at Harness Optimization](https://arxiv.org/abs/2608.06301)
- **Framework**: [VeRO](../vero/), which runs the version-evaluate-select loop, the evaluation sidecar and the metered model gateway
- **Conventions**: [`CONFIGURATION.md`](CONFIGURATION.md) documents every shared setting and the per-benchmark values

## Layout of a benchmark

- `target/` is the program the optimizer may edit. The paper's GAIA variant uses
  `target-shell/`, a non-functional stub, so gain there is the raw held-out score.
- `partitions/` pins the cases and the split, as JSON lists of task ids.
- `baseline/build.yaml` is trusted configuration: target model, evaluator,
  access policy, budgets, gateway scopes, outer-trial limits and final scoring.
- `baseline/compiled.manifest.json`, where present, is the SHA-256 manifest of
  the compiled task; `vero harbor build --check` verifies a checkout reproduces it.

Development evaluations expose per-case results and complete Harbor trial
records, including exact failures and target-agent logs. Validation is
aggregate-only, and test is reachable only by the trusted final verifier.

## Running a cell

```bash
cd vero
uv run vero harbor run \
  --config ../harness-opt-bench/officeqa/baseline/build.yaml \
  --env-file secrets.env --environment modal \
  --agent opencode --model anthropic/claude-sonnet-5 \
  --param optimizer_model=claude-sonnet-5 \
  --param wandb_run=officeqa__claude-sonnet-5-opencode__r1 \
  -o ../runs/officeqa/claude-sonnet-5-opencode-r1/jobs
```

Runs take hours, so start them detached from the shell (`setsid`, `nohup` with a
new session, or a scheduler) and keep the process id. The env file carries the
gateway upstream key and base URL, Modal and W&B credentials, and optionally
`MODAL_ENVIRONMENT`. `skills/run-benchmark/SKILL.md` is the full runbook. See
`CONFIGURATION.md` for how each optimizer harness spells its model on the wire;
a mismatch with the producer allow-list is a 403 on the first request.

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

The four benchmarks the paper reports on live at the top level. Three more that
were wired and run but are not in the paper sit under [`archive/`](archive/),
with the reason for each in its README.

| Benchmark | Editable target | Dataset | Split |
| --- | --- | --- | --- |
| [GAIA baseline](gaia/baseline/) | Tool-using Responses API agent | Harbor `gaia/gaia` | 20% / 40% / 40% |
| [OfficeQA baseline](officeqa/baseline/) | Grounded document-QA agent | Treasury Bulletin corpus | 20% / 40% / 40% |
| [BrowseComp-Plus baseline](browsecomp-plus/baseline/) | Fixed-corpus deep-research agent | Pinned local Harbor tasks | 20% / 40% / 40% |
| [Terminal-Bench baseline](terminal-bench/baseline/) | Shell-loop terminal agent | Harbor `terminal-bench/terminal-bench-2-1` | 20% / 40% / 40% |
