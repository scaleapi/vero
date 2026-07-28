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
