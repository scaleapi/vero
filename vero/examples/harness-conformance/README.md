# harness-conformance

A conformance check for the *stack*, not the model. Run it before spending a real
benchmark on a new optimizer harness or a new model, and you find out in minutes
whether the pieces an optimizer depends on actually work.

It is deliberately the **same path** as `harness-engineering-bench`: the harbor
evaluation backend, a nested `harbor run` per case, a target agent metered through
the inference gateway's evaluation scope, and an optimizer metered through the
producer scope. Only the work is trivial — six arithmetic tasks, two per
partition, seconds each. A `command`-backend smoke would be cheaper still, but it
exercises different code and would not tell you what you need to know.

## Run

```bash
cd <repo>/vero
uv run vero harbor run \
  --config examples/harness-conformance/build.yaml \
  --env-file secrets.env \
  --environment docker \
  --agent claude-code --model claude-sonnet-5 \
  --yes -o /tmp/conformance
```

Swap `--agent` / `--model` for whatever is under test. Useful parameters:

| parameter | default | why |
|---|---|---|
| `--param inner_env=...` | `modal` | **docker does not work** — the inner evaluation runs `harbor run -e docker` inside the sidecar, which has no docker CLI or socket, so every case fails with `Docker is not installed or not on PATH` and the eval 502s |
| `--param target_model=...` | `fireworks_ai/deepseek-v4-flash` | check a target model's gateway wiring |
| `--param optimizer_model=...` | follows `--model` | when the adapter requests a different string than it is launched with |
| `--param wandb_mode=online` | `disabled` | a smoke test should not need W&B |

## What it proves

Nine checks, in the order the optimizer meets them. The instruction is a literal
step-through — see `description` in `build.yaml`.

| step | proves |
|---|---|
| 1 | the optimizer's `OPENAI_BASE_URL` is the compose-internal gateway, not a public endpoint — i.e. it is metered and cannot see upstream |
| 2 | `evals plan` exposes partitions, **case counts**, disclosure levels, and remaining budget |
| 3 | development task resources are mounted under `.evals/tasks/`; validation's are not |
| 4 | an evaluation runs, blocks in the foreground, and returns a score |
| 5 | **persistence** — the same result is recoverable from `.evals/results/` via `evals list` / `show` / `cases`, so truncated output is never lost |
| 5b | **traces** — `evals cases` reports `trace: true` and `evals trace ID CASE` returns per-phase spans with plausible durations |
| 6 | **disclosure is enforced** — `evals cases` refuses on an aggregate-only partition |
| 7 | budget accounting decrements by what was actually spent |
| 8 | the edit → commit → re-evaluate loop moves the score, and `evals diff` attributes it |
| 9 | `evals submit` accepts a nomination |

## Reading the result

Two independent signals, and they should agree:

1. **`conformance-report.json`**, committed at the target repo root by the
   optimizer. One entry per step plus a `summary`. This is also where the
   optimizer records anything that *would* handicap it — a confusing message, a
   missing number, a command that behaved oddly.
2. **The reward.** The seed target answers addition and abandons multiplication,
   so a healthy stack scores **~0.5 for the seed and 1.0 after the fix**. Scoring
   0.0 throughout usually means the target could not reach the gateway; the
   arithmetic is easy enough that no plausible model gets it wrong.

If the report says every step passed but the reward is 0.0, trust the reward —
and treat the disagreement itself as a finding.

## Why the seed is broken on purpose

`target/src/conformance_agent/agent.py` returns early on `*`, leaving those
tasks unanswered. That gap is what makes step 8 meaningful: without it the seed
would already score 1.0 and there would be no way to see whether the
edit → evaluate → submit loop actually works. The fix is one early return.

## Extending it

Adding a check usually means adding a step to `description` — no code. Add a
*task* only if you need a new failure mode in the target: `tasks/` entries are
generated from a template (`task.toml`, `instruction.md`,
`environment/Dockerfile`, `tests/{test.sh,verify.py}`, `solution/solve.sh`), and
`partitions/*.json` list them by directory name. Keep every partition at two
cases so the whole run stays a few minutes.

Timeouts follow the same rules as the benchmark suite (see
`harness-engineering-bench/CONFIGURATION.md`): `case_timeout_seconds` equals the
tasks' declared `[agent] timeout_sec` so harbor's agent-timeout multiplier is
exactly 1.0, and each ceiling sits above
`ceil(trials / max_concurrency) × case_timeout`.

## Known-good and known-bad results

The first run of this example found three real problems, which is roughly what it
is for:

- `inner_env=docker` cannot work (see the parameter table). Fixed by defaulting to
  `modal`.
- The checklist originally asked the optimizer to print `$OPENAI_API_KEY`. It
  declined, correctly, and said so in its report. Now it only checks the variable
  is non-empty.
- The optimizer committed its report *after* `evals submit`, so the submitted
  candidate did not contain it and the report never reached the operator. The
  checklist now fixes the order and ends with `cat conformance-report.json` so the
  findings survive in the transcript regardless.

It also confirmed the tooling is adequate for diagnosis: the agent-facing error was
a bare `502 evaluation failed`, but `evals show ID` surfaced harbor's verbatim
stderr, and the optimizer root-caused the Docker fault unaided and retried 7 times
to rule out transience.
