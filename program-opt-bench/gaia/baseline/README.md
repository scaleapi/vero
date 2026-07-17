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
  --config ../program-opt-bench/gaia/baseline/build.yaml \
  --output ../program-opt-bench/gaia/baseline/compiled
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
`vero harbor status`.

Run the outer optimization through VeRO so it can give Harbor's coding-agent
adapter the scoped credential while forwarding the upstream credential only to
the gateway:

```bash
cd vero
uv run vero harbor run \
  --config ../program-opt-bench/gaia/baseline/build.yaml \
  --environment modal \
  --agent codex \
  --model openai/gpt-5.5
```

The coding agent edits only `target/`; inner evaluations run the candidate
against the pinned GAIA tasks on Modal. Complete development tasks and
attachments are mounted read-only under `.vero/cases/`. After each development
evaluation, complete Harbor trial records for every case—including exact
failures and target-agent logs—are available under
`.vero/evaluations/`. Validation remains aggregate-only, and test remains
verifier-only.

Before Modal teardown, the shared verifier exports the complete VeRO session
and a self-contained `experiment.html` into Harbor's verifier artifacts. Keep
the session archive alongside the HTML: it is the canonical candidate and
evaluation history and can be rendered again with `vero report`.
