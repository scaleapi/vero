---
name: conformance-check
description: >-
  Verify a new optimizer harness or model against the real VeRO stack in ~10
  minutes before spending a benchmark on it — gateway scoping and metering, the
  evals CLI, result persistence, disclosure enforcement, budget accounting, and
  the edit → evaluate → submit loop. Use when introducing an agent harness or
  model, after changing gateway/sidecar/timeout configuration, or when a real run
  fails and you need to know whether the stack or the benchmark is at fault.
---

# Conformance-checking a harness or model

`vero/examples/harness-conformance` runs the **same path a real experiment takes**
— harbor evaluation backend, a nested `harbor run` per case, a target agent
metered through the gateway's evaluation scope, an optimizer through the producer
scope — over six arithmetic tasks that finish in seconds. It exists so that a
harness problem costs ten minutes rather than a benchmark.

Read `README.md` in that directory for why it is built the way it is. This file is
how to run it and how to diagnose it when it fails.

## When to run it

- Before the first real run with a **new `--agent`** or a **new `--model`**.
- After changing anything shared: gateway budgets or allow-lists, timeouts,
  `agent_env`, sidecar or harbor versions.
- When a benchmark run fails early and you cannot tell whether the fault is the
  stack or that benchmark.

## Run it

```bash
cd <repo>/vero
uv run vero harbor run \
  --config examples/harness-conformance/build.yaml \
  --env-file <your-secrets-file> \
  --environment docker \
  --agent <harness> --model <provider/model> \
  --yes -o <scratch-dir>/jobs
```

Parameters worth knowing:

| parameter | default | why |
|---|---|---|
| `--param optimizer_model=` | follows `--model` | set it when the harness **requests** a different model string than it is launched with; it controls the producer allow-list independently |
| `--param target_model=` | a cheap chat model | the model the *target* harness calls; also the evaluation-scope allow-list |
| `--param inner_env=` | `modal` | leave it. `docker` cannot work — the inner evaluation runs `harbor run -e docker` inside the sidecar, which has no docker client or socket |
| `--param wandb_mode=online` | `disabled` | a smoke test should not need telemetry |

Inner evaluations run on Modal, so the Modal credentials must be listed under
`secrets:` in the build — omit them and every case fails
`Modal requires authentication` with 0 trial groups.

## Reading the result

Two independent signals. **Check both, and if they disagree trust the reward.**

1. **`conformance-report.json`**, committed by the optimizer at the target repo
   root, one entry per step plus a `summary`. Recover it from the job's session
   archive:

   ```bash
   tar xzf <jobs>/*/task__*/verifier/session.tar.gz -C <tmp>
   cd <tmp>/session/candidates/repository.git
   git log --oneline --all
   git show <commit>:conformance-report.json
   ```

   The optimizer is asked to commit the report **before** submitting, because a
   commit made after `evals submit` is not part of the submitted candidate and
   never reaches you. It also `cat`s the report as its final action, so the
   findings survive in the agent transcript even if the commit does not.

2. **The reward.** The seed target answers addition and abandons multiplication,
   so a healthy stack reads **~0.5 for the seed and 1.0 after the fix**.
   `<jobs>/*/task__*/verifier/reward.json` holds the final number.

Interpreting the reward:

| reward | meaning |
|---|---|
| `1.0` | the loop works: the optimizer measured, fixed the gap, and submitted |
| `~0.5` | evaluation works but the optimizer never fixed the seed — a harness-behaviour problem, not infrastructure |
| `0.0` | usually the target could not reach the gateway, or evaluations never ran. The arithmetic is trivial, so no plausible model gets it wrong |

## What each step proves

| step | proves |
|---|---|
| 1 | the optimizer's base URL is the compose-internal gateway, not a public endpoint — i.e. metered, and it cannot see upstream |
| 2 | `evals plan` exposes partitions, **case counts**, disclosure levels, remaining budget |
| 3 | development task resources are mounted; validation's are not |
| 4 | an evaluation runs, blocks in the foreground, returns a score |
| 5 | **persistence** — the same result is recoverable from `.evals/results/`, so truncated output is never lost |
| 5b | **traces** — `evals cases` reports `trace: true` and `evals trace ID CASE` returns per-phase spans with plausible durations |
| 6 | **disclosure is enforced** — `evals cases` refuses on an aggregate-only partition |
| 7 | budget decrements by what was actually spent |
| 8 | the edit → commit → re-evaluate loop moves the score, and `evals diff` attributes it |
| 9 | `evals submit` accepts a nomination |

## When it fails: diagnose in this order

**1. The evaluation record's diagnostics — the single most useful artifact.**
An agent-facing `502 evaluation failed` is deliberately terse, but the record
carries the backend's verbatim stderr:

```bash
python3 -c "
import json,glob
for p in glob.glob('<tmp>/session/evaluations/*/evaluation.json'):
    d=json.load(open(p)); r=d.get('report') or {}
    if r.get('status')!='success':
        print(json.dumps(r.get('diagnostics'), indent=1)[:1200])
"
```

This is where "Docker is not installed", "Modal requires authentication", and
"Can't instantiate abstract class" all surfaced immediately.

**2. The gateway request log — which model, which endpoint, what status.**
Definitive for anything credential- or routing-shaped:

```bash
python3 -c "
import json,glob,collections
c=collections.Counter()
for p in glob.glob('<tmp>/session/artifacts/inference/requests/*.jsonl'):
    for l in open(p,errors='replace'):
        r=json.loads(l)
        if r.get('scope')=='producer':
            c[(r.get('status'), r.get('model'), r.get('endpoint'))]+=1
for k,v in c.items(): print(k,'->',v)
"
```

Read it as:

- **zero producer requests** → the harness bypassed the gateway entirely. It is
  calling a provider's public endpoint with a scoped token, so it fails closed
  with `401`; nothing leaks, but nothing works either.
- **`403 model_denied`** → the requested model string is not in the producer
  allow-list. Note the string it *actually requested*, which is often not the one
  you launched with, and set `--param optimizer_model=` to that.
- **`200` on an unexpected `endpoint`** → the harness is using an API surface you
  did not intend (e.g. `responses` rather than `messages`), which matters both for
  compatibility and for token attribution.

**3. The harness's own trajectory**, under
`<jobs>/*/task__*/agent/` — usually a `.txt` stream and/or `trajectory.json`.
Filter to errors rather than reading it whole:

```bash
tr -d '\000' < <jobs>/*/task__*/agent/<harness>.txt | grep -o '"type":"error".\{0,240\}' | tail -3
```

**4. The trial phase timings**, via `evals trace ID CASE_ID` or the per-case
`execution_trace`. A near-zero `agent_execution` span means the harness never
called a model on that case.

## Harness gotchas found this way

Each of these was discovered by this check rather than by a benchmark run, and
each is the kind of thing that would otherwise read as "the model is bad":

- **A secondary small model.** Several harnesses issue an auxiliary
  summarisation/title call with a small model of the *same provider family*. The
  producer allow-list usually holds one entry, so that call `403`s silently — it
  is invisible outside the gateway log. Allow a second model.
- **Provider-prefixed model names.** Some harnesses require `provider/model` and
  then request only the bare `model`, so the allow-list needs the bare form while
  the launch flag needs the prefixed one. That is what `--param optimizer_model=`
  is for.
- **Which API surface the harness drives.** Crossing model families through a
  proxy can produce responses a harness cannot parse (mixed id namespaces in one
  stream). Prefer the provider's native surface; vero supplies a gateway base URL
  for non-`openai` opencode providers so Anthropic models stay on Messages.
- **A seed harness that will not instantiate.** Missing abstract methods surface
  as `infrastructure_failure` on every case, not as a low score.

## Extending it

Adding a check is usually a step in the build's `description` — no code. Add a
*task* only to introduce a new failure mode in the target; keep every partition at
two cases so a run stays a few minutes. Timeouts follow the same rule as the
benchmark suite: `case_timeout_seconds` equals the tasks' declared
`[agent] timeout_sec` so harbor's agent-timeout multiplier is exactly 1.0.
