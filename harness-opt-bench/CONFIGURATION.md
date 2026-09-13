# Benchmark configuration

Each benchmark is defined by its `baseline/build.yaml`. Those files are the
source of truth; this page summarizes the common protocol, the values that differ
between benchmarks, and the rules for changing them.

## Evaluation protocol

HarnessOpt-Bench gives an optimizer controlled access to three partitions:

| Partition | Optimizer access | Purpose |
| --- | --- | --- |
| Development | Per-case scores, feedback, and the tasks' input files | Diagnose and improve the harness |
| Validation | Aggregate scores over at least five cases | Compare candidates without revealing held-out labels |
| Test | No optimizer access | Produce the final reported score |

The optimizer may evaluate any saved candidate, not just its current checkout.
Every evaluated version remains available for selection and audit.

### Selection and reward

The optimizer nominates a final candidate. If it does not submit one, VeRO falls
back to the best validation candidate and then to the latest candidate.

The nominated candidate is evaluated on test and compared with the pinned seed
baseline. The benchmark reports normalized improvement over that baseline.
Accuracy determines selection and reward; resource-use metrics are reported but
are not part of the objective.

### Budgets and disclosure

Development and validation each allow 100 evaluation requests and a case budget
equal to four complete passes over the partition. By default, the optimizer can
see its remaining request, case, and model-usage budgets.

The budget-blind ablation sets `disclose_budget: false`. Enforcement is unchanged,
but remaining budgets and model-usage metrics are removed from the results the
optimizer is shown. Latency remains visible.

Three model-usage budgets are kept apart: the optimizer's own model calls, the
evaluations it runs during search, and the final scoring. Search cannot consume
the capacity reserved for final scoring.

### Attempts and failures

Search and validation run each case once. Final test scoring runs each case three
times and averages the attempts, matching the procedure used to estimate the
pinned seed baselines.

Failure handling distinguishes the harness from the evaluation system:

- A candidate failure is an informative outcome and receives the configured
  failure score.
- A case that never ran because the evaluation system failed is excluded and
  counts toward the evaluation's error rate.
- An evaluation is invalid when infrastructure loss exceeds 10% of its cases.
- Whole-evaluation retries are reserved for trusted final scoring. They are not
  used during candidate search, where retrying candidate-triggered failures could
  bias selection.

### Reported measurements

Every evaluation reports accuracy, case count, error rate, and score variability.
It also records model usage and wall time. Because usage and latency can be
heavy-tailed, per-case summaries include mean, median, and maximum values.

Detailed traces and artifacts are available only where the partition's disclosure
policy permits them. These measurements support auditing and cost analysis but do
not affect candidate selection.

## Shared defaults

| Setting | Value | Reason |
| --- | --- | --- |
| Objective | Maximize `score` | Keeps optimization aligned with task quality |
| Selection mode | Optimizer submission | Measures the optimizer's ability to choose its result |
| Baseline floor | Disabled | Validation should not gate a reward measured on test |
| Seed evaluation | Pinned | Avoids repeating a full seed evaluation every run |
| Validation rescore pool | Top 3 candidates | Gives automatic fallback a small trusted comparison set |
| Search attempts | 1 per case | Preserves evaluation budget for exploring candidates |
| Test attempts | 3 per case, mean aggregation | Reduces final-score variance |
| Case budget | 4 partition passes | Allows broad evaluation plus targeted follow-up |
| Minimum aggregate | 5 cases | Prevents validation from revealing individual labels |
| Maximum concurrency | 24 trials | Shared throughput setting used by timeout calculations |
| Error-rate threshold | 10% | Rejects aggregates dominated by missing cases |
| Final-score retries | Up to 3 evaluation attempts | Recovers from failures outside the candidate |
| Feedback transcript limit | 16,000 bytes | Bounds context while retaining useful diagnostics |

## Per-benchmark values

### Experimental design

| Benchmark | Target model | Development / validation / test | Pinned seed baseline |
| --- | --- | --- | --- |
| [GAIA](gaia/baseline/build.shell.yaml) | `gpt-5.4-mini` | 33 / 66 / 66 | 0.0 by construction |
| [OfficeQA](officeqa/baseline/build.yaml) | `deepseek-v4-flash` | 49 / 98 / 99 | 0.341 ± 0.033 |
| [BrowseComp-Plus](browsecomp-plus/baseline/build.yaml) | `deepseek-v4-flash` | 33 / 66 / 66 | 0.462 ± 0.028 |
| [Terminal-Bench](terminal-bench/baseline/build.yaml) | `grok-build-0.1` | 17 / 36 / 36 | 0.241 ± 0.013 |

Target models are named without a provider. Each build maps the name to a
deployment under the evaluation scope's `model_aliases`; pass
`--param target_model_route=<provider/model>` to use a different one.

Each baseline is the mean of three independent test rounds; the reported
uncertainty is the standard deviation of those round means. GAIA uses a
multimodal target because some held-out cases contain images. The reported GAIA
variant, `build.shell.yaml`, starts from a seed that does not run, so its
baseline is a measured zero and improvement there equals the raw held-out
score. The original tool-using seed in `build.yaml` scores 0.621 ± 0.052 and is
not part of the reported suite.

### Search budgets

| Benchmark | Development budget | Validation budget |
| --- | --- | --- |
| GAIA | 100 runs / 132 cases | 100 runs / 264 cases |
| OfficeQA | 100 runs / 196 cases | 100 runs / 392 cases |
| BrowseComp-Plus | 100 runs / 132 cases | 100 runs / 264 cases |
| Terminal-Bench | 100 runs / 68 cases | 100 runs / 144 cases |

### Time limits

| Benchmark | Case limit | Evaluation limit | Finalizer limit |
| --- | --- | --- | --- |
| GAIA | 10 minutes | 2 hours | 4 hours |
| OfficeQA | 30 minutes | 8 hours | 15 hours |
| BrowseComp-Plus | 60 minutes | 11 hours | 21 hours |
| Terminal-Bench | Task-defined; 15-minute reference | 12 hours | 18 hours |

The exact values, in seconds, remain in each `build.yaml`. Terminal-Bench tasks
declare different limits; its reference value preserves those relative limits
rather than replacing them with one global timeout.

### Optimizer limits

The optimizer itself is bounded by wall clock, not only by evaluation budget.
Every benchmark gives it 20 hours of agent time inside a 24-hour sandbox that is
reclaimed after one idle hour. When the agent clock runs out, the optimizer is
stopped and final scoring still runs on the candidate it had submitted, so a run
cut off by the clock is scored rather than lost. The sandbox limit is the hard
ceiling: reaching it loses the run.

### Optimizer harness versions

The optimizer harness is installed from its public registry at the start of
every run, so by default a run uses the current release. Each build records
the releases its reported results were produced with under
`optimizer_harness_versions`; pass `--pin-harness` to `vero harbor run` to
install those instead and replicate a result with the same harness.

| Harness | Release |
| --- | --- |
| claude-code | 2.1.220 |
| codex | 0.146.0 |
| kimi-cli | 1.49.0 |
| goose | 1.45.0 |
| mini-swe-agent | 2.4.6 |
| opencode | 1.18.9 (OfficeQA), 1.18.10 (BrowseComp-Plus, Terminal-Bench), 1.18.11 (GAIA) |

opencode released three patch versions while the original rounds ran, so the
recorded release is the one most of that benchmark's rounds used. Pass
`--ak version=<release>` on the command line to run any specific release.

## Configuration rules

### Baselines

Pinned baselines must be refreshed whenever the seed harness, target model, test
partition, or scoring behavior changes. Candidate-caused failures count as zero;
evaluation-system failures are excluded. The same rule is applied to seed and
candidate attempts.

A pinned baseline is reported with every final result, so consumers can compute
improvement without joining against this document.

### Timeouts

Case limits follow the limits declared by the underlying dataset. Evaluation
limits are derived from the number of trials, maximum concurrency, and case
limit:

```text
worst-case evaluation = ceil(trials / max_concurrency) × case limit
```

The evaluation limit is set above that bound. The finalizer limit covers final
test scoring plus any validation rescoring. Search and finalization have
independent clocks, so time spent searching does not reduce the finalizer limit.

Changing a partition size, attempt count, case limit, or concurrency therefore
requires recalculating both limits.

### Concurrency

`max_concurrency` counts trials, not unique cases. With three attempts per test
case, a concurrency of 24 evaluates at most eight test cases simultaneously.

Concurrency is a throughput control, not part of the benchmark objective. Change
it only after confirming that the execution and model-serving layers remain
stable, then recompute the timeout bounds above.

### Model-usage caps

Model-usage caps are runaway safeguards, not the primary experiment budget. Case
budgets already bound the optimizer's authorized work, so these caps should sit
comfortably above the largest valid evaluation rather than truncate it.

Final evaluation has a separately reserved cap. Exhausting an optimizer-visible
search budget is a valid outcome; exhausting the reserved final-evaluation budget
invalidates the run.

### Isolation

The optimizer cannot access test cases, trusted evaluation state, or unrestricted
model credentials. Development and validation disclosures are generated by the
trusted evaluator according to the partition policy.

Some tasks depend on external services of their own and therefore use a reduced
isolation profile. That limitation must be declared in the benchmark configuration
and considered before running with an adversarial optimizer.

### Deployment settings

A few fields describe the services a run depends on rather than the benchmark
itself. They are the same in every build file, and the build files do not
repeat their meaning.

- `harbor_requirement` names the Harbor package the evaluation service installs.
  It carries the extra for the evaluation environment and can be overridden with
  `--param harbor_requirement=`.
- `vero_requirement` names the published VeRO release the compiled task installs.
  It must equal the version doing the compiling.
- `secrets` lists the environment variable names the run needs for its execution
  and telemetry services. `vero harbor run` refuses to launch while any is
  unset. Edit the list and the env file together.
- `wandb` configures telemetry and is optional. Remove the block to run without it.
- `extra_harbor_args` passes options to the evaluation sandboxes. The defaults
  group them under one application and reclaim idle ones; remove them for an
  environment that does not accept those options.
- `inference_gateway.request_log_attribution` stamps each gateway request with the
  trial it served, which is what makes per-trial usage attribution reliable.
- `agent_env` raises the optimizer's tool-call time limit so one full evaluation
  can complete inside a single foreground call, and disables background tasks.

The remaining shared values, such as attempts, budgets, time limits and
isolation, follow the rules above; a build file comments only on what is
specific to its benchmark.

## Compiled tasks

`vero harbor build` turns a `build.yaml` into a self-contained task directory,
and two compiles of the same sources produce identical output. Nothing specific
to one run is written into the compiled task: the per-run access tokens and the
optimizer model are supplied by the launcher through the environment and read
once when the services start, so a running evaluation cannot be altered from
outside.

`vero harbor build --manifest <file>` records a checksum of every compiled file,
and `--check <file>` recompiles and fails if the output differs. Use them to
prove that two checkouts produce the same task; nothing is committed.

By default the compiled task carries a copy of the VeRO source tree so its
images can be built without a package index. Setting `vero_requirement` to an
exact published version (for example `scaleapi-vero==0.6.0`) installs that
release instead; the version must match the VeRO doing the compiling.

## Adding or changing a benchmark

1. Update the benchmark's `baseline/build.yaml`.
2. Keep the development, validation, and test partitions immutable once results
   are reported.
3. Refresh the pinned baseline when the seed, target model, test set, or scoring
   behavior changes.
4. Recalculate timeouts whenever partition size, attempts, concurrency, or case
   limits change.
5. Run the benchmark-configuration tests and the harness conformance example.
6. Compile the benchmark and inspect the generated task.
7. Update the summary tables above only after the build configuration is final.

Useful checks:

```bash
cd vero
uv run pytest tests/test_v05_benchmark_configs.py

uv run vero harbor build \
  --config ../harness-opt-bench/<benchmark>/baseline/build.yaml \
  --param inner_env=<evaluation-environment> \
  --output <output-directory>
```

Deployment credentials, execution-provider settings, telemetry destinations, and
capacity planning are intentionally outside this document. They do not define the
benchmark and should not be committed to its public configuration.
