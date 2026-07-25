# BrowseComp-Plus baseline

This editable target is a Responses API deep-research agent with three tools:
search the pinned BM25 index, open a document, and submit a formatted response.
The optimization agent may change its prompts, control flow, tool use, or
dependencies, but not the dataset, index, split, evaluated model, or verifier.

Build the generated tasks first as described in the parent
[`README.md`](../README.md), then compile from the repository root:

```bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-engineering-bench/browsecomp-plus/baseline/build.yaml \
  --output ../harness-engineering-bench/browsecomp-plus/baseline/compiled
```

For a real run, copy `secrets.env.example` to the ignored `secrets.env`, fill it
in, and use `vero harbor run` in the same way as the other harness-engineering
benchmarks.
