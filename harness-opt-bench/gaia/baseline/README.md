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
  --config ../harness-opt-bench/gaia/baseline/build.yaml \
  --output ../harness-opt-bench/gaia/baseline/compiled
```

Omit `VERO_SKIP_SECRET_CHECK=1` for a real build so VeRO verifies that the
OpenAI and Modal credentials declared in `build.yaml` are present. Set
`OPENAI_BASE_URL` to your OpenAI-compatible endpoint; use
`https://api.openai.com/v1` when calling OpenAI directly. The `compiled/`
directory is generated and intentionally ignored.

Run the resulting outer task with Harbor on Modal and the coding agent of your
choice. The coding agent edits only `target/`; inner evaluations run the
candidate against the pinned GAIA tasks on Modal.
