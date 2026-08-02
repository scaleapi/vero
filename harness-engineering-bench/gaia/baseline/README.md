# GAIA tool-using baseline

This leaf benchmark optimizes a small Harbor-native GAIA agent. The editable
program uses the OpenAI Responses API with three capabilities:

- web search;
- shell commands inside the GAIA task environment;
- image inspection and final-answer submission.

The trusted build fixes the evaluated model to
`gpt-5.4-mini-2026-03-17`. A candidate may change the agent's prompts, control
flow, tool definitions, file handling, or dependencies, but it cannot change
the dataset, verifier, split, model, access policy, or final test target.

## The shell-seed variant (`build.shell.yaml`)

`build.shell.yaml` runs the same benchmark against `target-shell/` instead of
`target/`. The skeleton there satisfies the Harbor interface, resolves the model
and constructs a client, but its `run` writes an empty answer and returns. It
scores zero.

**Why.** Every other optimization task in the suite starts from a seed that
already works, so what they measure is *tuning*: how much an optimizer can add
to a competent program. This variant asks a different question -- what an
optimizer does when there is nothing to tune and it has to write the program
first. The two runs are directly comparable because everything except the seed
is held fixed: same cases, same partitions, same target model, same gateway
scoping. The invariant test in `vero/tests/test_v05_benchmark_configs.py`
enforces that, and also asserts the skeleton never calls a model -- a shell that
scores above zero is not a shell.

**Why the plumbing is present.** The skeleton keeps the model resolution and the
client construction, including the `removeprefix("openai/")` that the gateway
allow-list requires. Those are properties of this harness, not of GAIA, and
making an optimizer rediscover them by trial and error would spend budget on the
wrong thing and add variance unrelated to the question being asked.

**Why it writes an empty answer rather than doing nothing.** A case that scores
zero and a case that errors are different events here. Errors count against
`error_rate_threshold`, and a wholly erroring evaluation comes back `invalid`.
Writing the file keeps every case scoreable, so the floor is a real 0.0 and a
half-built candidate gets a number rather than nothing.

**`baseline_reward: 0.0` is a claim, not a measurement.** It follows from the
skeleton writing an empty answer. Confirm it with one baseline round before
quoting deltas against it.

**The framing lives in a template, not in `description`.** The built-in
instruction opens with "Improve the program in ...", which is the first thing the
optimizer reads and is false here. `instruction_template:
instruction.shell.md.j2` replaces that opening with "Build the program, then
optimize it". The template `{% extends "instruction.md.j2" %}` and overrides only
the `framing` block, so the workflow, budget, inspection and rules sections are
inherited verbatim and cannot drift as the shared instruction evolves -- the
invariant test asserts both the extends and the byte-identical remainder.
`description` is left saying only what the program must *do*, which keeps task
shape and task content in separate places.

**Deliberately not in the target.** The rationale above lives in this README
because the optimizer mounts `target-shell/` and would read anything placed
there. Notes about what the seed lacks, or what a good implementation would
contain, are an answer key.

## Compile the outer Harbor task

From the repository root:

```bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-engineering-bench/gaia/baseline/build.yaml \
  --output ../harness-engineering-bench/gaia/baseline/compiled
```

Omit `VERO_SKIP_SECRET_CHECK=1` for a real build so VeRO verifies that the
OpenAI and Modal credentials declared in `build.yaml` are present. Set
`OPENAI_BASE_URL` to your OpenAI-compatible endpoint; use
`https://api.openai.com/v1` when calling OpenAI directly. The `compiled/`
directory is generated and intentionally ignored.

The compiled task does not expose the upstream OpenAI key to either editable
program. A separate inference-gateway container holds it. The outer coding
agent uses a producer-scoped token for `gpt-5.4` or `gpt-5.5`; GAIA candidates use an
evaluation-scoped token restricted to `gpt-5.4-mini-2026-03-17`. Their request
and token budgets are recorded separately and are visible through
`evals status`.

Run the outer optimization through VeRO so it can give Harbor's coding-agent
adapter the scoped credential while forwarding the upstream credential only to
the gateway. `vero harbor run` is the whole pipeline in one command: it compiles
the `build.yaml` to a temporary task, wires credentials (relocating the upstream
inference key to the gateway and handing the optimizer a scoped producer token),
and invokes Harbor. Put run secrets in a dotenv file rather than exporting them
by hand — copy `secrets.env.example` to `secrets.env` (gitignored) and fill it
in:

```bash
cp harness-engineering-bench/gaia/baseline/secrets.env.example \
   harness-engineering-bench/gaia/baseline/secrets.env   # then edit it

cd vero
uv run vero harbor run \
  --config ../harness-engineering-bench/gaia/baseline/build.yaml \
  --env-file ../harness-engineering-bench/gaia/baseline/secrets.env \
  --environment docker \
  --agent codex \
  --model gpt-5.6-sol \
  --yes \
  -o ../runs/gaia-full/jobs
```

`--env-file` values override the ambient shell and are passed only through the
subprocess environment, never on the command line. Anything after the known
options (here `--yes -o ...`) is forwarded verbatim to `harbor run`. The
`--model` value is fed to `${optimizer_model}` so the producer scope's
allow-list and the optimizer agent's model stay in lockstep (no 403 mismatch).
Use `--environment modal` to run the optimizer itself remotely, or `docker` to
run it locally against Modal eval backends.

The coding agent edits only `target/`; inner evaluations run the candidate
against the pinned GAIA tasks on Modal. Complete development tasks and
attachments are mounted read-only under `.evals/tasks/`. After each development
evaluation, complete Harbor trial records for every case—including exact
failures and target-agent logs—are available under
`.evals/results/`. Validation remains aggregate-only, and test remains
verifier-only.

Before Modal teardown, the shared verifier exports the complete VeRO session
and a self-contained `experiment.html` into Harbor's verifier artifacts. Keep
the session archive alongside the HTML: it is the canonical candidate and
evaluation history and can be rendered again with `vero report`.
