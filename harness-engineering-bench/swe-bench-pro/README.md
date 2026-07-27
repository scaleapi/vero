# SWE-bench-Pro

This benchmark optimizes a code-editing agent on SWE-bench-Pro tasks. Each task
checks a real project out at `/app` and describes a bug fix or feature. The
editable agent (`baseline/target/src/swebench_pro_agent/agent.py`) edits the
repository in place; the **task-source verifier** applies the resulting repo state
and runs the task's hidden test suite for the reward. The agent does not
self-grade and writes no answer file: reward comes entirely from the suite.

## Task source

SWE-bench-Pro ships as the **`swebenchpro` dataset in the default Harbor
registry**, version `1.0`, 731 instances across 11 upstream projects. It is a
*registry* dataset, not an `<org>/<name>@sha256:<digest>` package like
swe-atlas-qna or tau3, so the pin is a bare `name@version`:

```
task_source: swebenchpro@1.0
```

Its tasks resolve to git-backed task ids under
`laude-institute/harbor-datasets` at commit
`c8e8f3fac7097accaacf261d74c3d6f441de45b1`, path
`datasets/swebenchpro/instance_<owner>__<repo>-<sha>`. Each task ships an
`environment/Dockerfile` that `FROM`s a public prebuilt DockerHub image
(`jefzda/sweap-images:<instance>`, 0.5-4.8 GB), resets the repo to the base
commit, and clears `ENTRYPOINT` (so harbor's default keepalive works and no
`--ek keepalive` override is needed, unlike swe-atlas-qna).

Grading is entirely offline: `tests/test.sh` checks out the gold test files,
runs the official `run_script.sh`, parses results with the official `parser.py`,
and writes `1` to `/logs/verifier/reward.txt` only when every test in
`fail_to_pass` and `pass_to_pass` passed. **No judge model, so no verifier
credentials are needed** (`--ve` is unnecessary for this benchmark).

Two consequences of being a registry rather than a package dataset:

- `agent_access[development].expose_case_resources` must be `false`. VeRO
  materializes case resources through `PackageDatasetClient`, which only parses
  `<org>/<name>` refs and rejects a bare `swebenchpro@1.0`.
- The recorded per-task `ref` in `partitions/manifest.json` is
  `<git_commit>:<path>` rather than a `sha256:` content hash.

## Split design

The committed split is deterministic and stratified by source repository (so each
represented codebase appears across all three partitions):

- development: 146 (20%, full result disclosure to the optimizer)
- validation: 292 (40%, aggregate-only; used to select candidates)
- test: 293 (40%, held out until Harbor grades the completed outer task)

731 does not divide evenly: the exact ratios are 146.2/292.4/292.4, so the one
leftover case is assigned by largest remainder, and the .4/.4 tie between
validation and test breaks towards test. That matches how the sibling benchmarks
resolved the same tie (officeqa 246 -> 49/98/99, swe-atlas-qna 124 -> 25/49/50).

Regenerate and verify the split with:

```bash
harbor download dataset swebenchpro@1.0 --output-dir /tmp/swebenchpro   # or a
# sparse checkout of laude-institute/harbor-datasets at the pinned commit

uv run --python 3.12 \
  harness-engineering-bench/swe-bench-pro/scripts/partition_swe_bench_pro.py \
  --tasks-dir /tmp/swebenchpro --fetch-registry

uv run --python 3.12 \
  harness-engineering-bench/swe-bench-pro/scripts/partition_swe_bench_pro.py \
  --tasks-dir /tmp/swebenchpro --check
```

## Target model: which models the seed agent can actually run on

The seed agent drives the **OpenAI Responses API** and chains turns with
`previous_response_id`. Measured against the shared LiteLLM proxy
(`heb-litellm-proxy`), that shape constrains the usable target models:

| model | turn 1 | chained turn (`previous_response_id`) |
|---|---|---|
| gpt-4o | OK (with the reasoning gate) | OK |
| gpt-5.4-mini-2026-03-17 | OK | OK |
| gpt-5.6-sol | OK | OK |
| deepseek-v4-flash | OK | OK |
| gpt-oss-120b | OK | OK |
| **qwen-3.6-27b** | OK | **fails, deterministically** |
| gpt-4.1 | 400 on `reasoning` | n/a |
| gpt-5.3-codex | 404 on the proxy's Responses route | n/a |

`qwen-3.6-27b` answers the first turn but every chained call returns
`litellm.BadRequestError: Fireworks_aiException ... invalid_request_error`
(3/3 reproductions, plus an in-harbor trial). The same conversation works on
`/chat/completions`, so the fault is LiteLLM's Responses-to-ChatCompletions
bridge replaying a shape Fireworks rejects, not the model. Switching the agent
to a stateless full-history `input` list also works on qwen, but that changes
the seed harness being measured, so it has not been done here.

`reasoning: {"effort": ...}` is gated on `_is_reasoning_model()` (the same
capability predicate the other five benchmark agents use): sending it to gpt-4o
or gpt-4.1 is a hard 400 on the very first turn.

## Build the outer Harbor task

From the repository root:

```bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-engineering-bench/swe-bench-pro/baseline/build.yaml \
  --output ../harness-engineering-bench/swe-bench-pro/baseline/compiled
```

Omit `VERO_SKIP_SECRET_CHECK=1` for a real build so VeRO verifies that the OpenAI
and Modal credentials declared in `build.yaml` are present. Set `OPENAI_BASE_URL`
to your OpenAI-compatible endpoint. The `compiled/` directory is generated and
intentionally ignored.

## Measure the held-out baseline

A K=3 baseline is three independent single-attempt rounds over the 293-case
`test` partition, scored by harbor's own
`stats.evals.*.metrics[0].mean` (an errored trial counts as a zero). Each round
is a plain `harbor run` against the registry dataset, with
`--agent-timeout-multiplier 0.6` (`case_timeout_seconds` 1800 over the task's
declared `[agent] timeout_sec` 3000):

```bash
harbor run \
  -a swebench_pro_agent.agent:SweBenchProAgent \
  -m <target-model> -e modal \
  -d swebenchpro@1.0 \
  $(python3 -c "import json;print(' '.join('-i '+t for t in json.load(open('../../partitions/test.json'))))") \
  --n-attempts 1 --max-retries 3 -n 5 --yes \
  --ek app_name=harness-engineering-bench --ek sandbox_idle_timeout_secs=3600 \
  --agent-timeout-multiplier 0.6 \
  --env-file baseline/secrets.env \
  -o ../../../runs/baseline-swe-bench-pro/roundN
```

Run it three times into `round1/`, `round2/`, `round3/`; the pinned
`baseline_reward` is the mean of the three round means and the `±` is the
population stdev across them. Concurrency matters: `-n 12` against the shared
Azure gpt-4o deployment produced 45 `RateLimitError` retries in 15 minutes,
while `-n 5` produced none.

## Run the optimization

`vero harbor run` compiles the `build.yaml` to a temporary task, wires credentials
(relocating the upstream inference key to the gateway and handing the optimizer a
scoped producer token), and invokes Harbor. Copy the secrets template first:

```bash
cp harness-engineering-bench/swe-bench-pro/baseline/secrets.env.example \
   harness-engineering-bench/swe-bench-pro/baseline/secrets.env   # then edit it

cd vero
uv run vero harbor run \
  --config ../harness-engineering-bench/swe-bench-pro/baseline/build.yaml \
  --env-file ../harness-engineering-bench/swe-bench-pro/baseline/secrets.env \
  --environment modal \
  --agent codex \
  --model gpt-5.3-codex \
  --yes \
  -o ../runs/swe-bench-pro-full/jobs
```

`--env-file` values override the ambient shell and are passed only through the
subprocess environment, never on the command line. Anything after the known
options (here `--yes -o ...`) is forwarded verbatim to `harbor run`.

The producer scope's allow-list in `build.yaml` is `gpt-5.3-codex`, so pass
`--model gpt-5.3-codex` (or an aligned model) to keep the optimizer's model and
the gateway allow-list in lockstep and avoid a 403 mismatch. The evaluation scope
is pinned to `gpt-4o` (matching `model: openai/gpt-4o`).

**Modal endpoint requirement:** `--environment modal` runs eval sandboxes on
Modal, which must be able to reach the model endpoint in `OPENAI_BASE_URL`. Use a
Modal-reachable endpoint (for example the Azure
`https://<resource>.openai.azure.com/openai/v1` endpoint), **not** a VPN-internal
host that only resolves on the Scale network. Use `--environment docker` to run
the optimizer locally against Modal eval backends instead.
