---
name: run-benchmark
description: >-
  Run one HarnessOpt-Bench optimization from preflight through held-out scoring,
  and verify that the resulting measurement is complete.
---

# Run a HarnessOpt-Bench optimization

This runbook covers the benchmark workflow without assuming a particular model
provider, execution backend, or observability service. Keep deployment settings
and credentials in local, ignored files.

## How a run is structured

Each benchmark is defined by
`harness-opt-bench/<benchmark>/baseline/build.yaml`. A run contains three trusted
components:

- an inference gateway that enforces model and usage limits;
- an evaluation service that owns cases, scores, and candidate selection;
- an optimizer that edits `target/` and submits candidates through `evals`.

After optimization, the evaluator scores the selected candidate on the held-out
test partition and writes the final result. The optimizer receives only scoped
access and the disclosures allowed for each partition.

## Prerequisites

- A checkout containing both `vero/` and `harness-opt-bench/`.
- `uv`, with commands run from `vero/`.
- An inference endpoint that supports the models named by the run.
- A Harbor environment capable of running the nested evaluations.
- Any credentials required by those services, stored in a local env file.
- Vendored task data where required. Check it with:

```bash
python3 ../harness-opt-bench/scripts/task_data.py --check
```

Provider-specific packages, credentials, capacity settings, and environment
arguments belong in the launch environment, not in the benchmark definition.

## Launch

```bash
cd <repo-root>/vero
uv run vero harbor run \
  --config ../harness-opt-bench/<benchmark>/baseline/build.yaml \
  --env-file <local-env-file> \
  --environment <optimizer-environment> \
  --agent <optimizer-harness> \
  --model <launch-model> \
  --param inner_env=<evaluation-environment> \
  --param optimizer_model=<wire-model> \
  --yes \
  -o <output-directory>/jobs
```

`--model` is the identifier passed to the optimizer harness.
`optimizer_model` must exactly match the identifier that harness sends to the
gateway. Some harnesses normalize model names, so verify the wire value before a
large run.

The optimizer harness is installed at launch, at its current release. To
replicate a reported result with the harness it was produced with, add
`--pin-harness`, which installs the release recorded in the build's
`optimizer_harness_versions`. `--ak version=<release>` selects any other
release.

## Preflight

Compile before launching:

```bash
uv run vero harbor build \
  --config ../harness-opt-bench/<benchmark>/baseline/build.yaml \
  --param inner_env=<evaluation-environment> \
  --output /tmp/harness-opt-preflight
```

Inspect the generated files:

1. `environment/sidecar/serve.json`: partition attempt counts and aggregation.
2. `serve.json` and `environment/gateway/config.json`: target and optimizer
   model allow-lists.
3. `task.toml`: optimizer and verifier time limits.
4. `instruction.md`: the complete optimizer prompt.

Also send a small request through the configured endpoint using the exact wire
model. A new optimizer harness or model should pass
`vero/examples/harness-conformance/` before a full benchmark run.

A compiled task is deterministic. To prove two checkouts agree, write a manifest
from one with `vero harbor build --manifest <file>` and run `--check <file>` from
the other.

## Launch discipline

Long runs should outlive the invoking shell. Start them in a durable session,
capture logs and the process identifier, and verify that the process is alive.
Launch one cell first and define a health gate before increasing concurrency.

Use measured error rates and latency to set concurrency. Shared inference or
execution capacity can make healthy individual runs fail when launched together;
increase load gradually and replace completed cells instead of adding bursts.

## Health checks

After initialization, verify that:

- the outer process is alive and its log has no traceback;
- the gateway, evaluator, and optimizer are running;
- optimizer requests reach the producer scope without authorization errors;
- nested evaluations are completing without terminations;
- the evaluator's records or telemetry keep advancing; the outer log only shows a spinner.

If a run must be stopped, terminate both the launcher and the work owned by its
execution environment. Before stopping it, export the evaluator session if the
normal verifier archive has not yet been written. That session contains the
candidate repository, budgets, and evaluation records needed for recovery.

## Completion and audit

A complete run contains
`jobs/<timestamp>/task__*/verifier/finalization.json` with:

- `shipped: true`;
- a non-null selected candidate parent;
- the expected reward key;
- the pinned baseline reward;
- an empty `errors` block.

Also confirm that:

- the final error rate is zero, or every missing case is accounted for;
- search evaluations were not unexpectedly terminated;
- independent cells did not submit identical candidate trees;
- `session.tar.gz` and the final report were exported.

For per-trial usage and latency, run:

```bash
python3 ../harness-opt-bench/scripts/per_trial_tokens.py \
  <extracted-session-directory> --json
```

## Benchmark summary

Values can change; confirm them in each `baseline/build.yaml`.

| Benchmark | Task data | Target model | Pinned baseline | Development / validation cases |
| --- | --- | --- | --- | --- |
| `officeqa` | Locally fetched | See build | 0.3412 | 196 / 392 |
| `browsecomp-plus` | Locally fetched | See build | 0.4619 | 132 / 264 |
| `terminal-bench` | Registry digest | `grok-build-0.1` | 0.2407 | 68 / 144 |
| `gaia` shell seed | Registry digest | `gpt-5.4-mini` | 0.0 by construction | 132 / 264 |

All reported benchmarks select on validation, let the optimizer submit the final
candidate, run search cases once, and average three attempts per held-out case.
The GAIA end-to-end variant is a small machinery check; its score is not a
benchmark result.
