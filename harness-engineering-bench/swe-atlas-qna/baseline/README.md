# SWE-Atlas-QnA codebase agent

This leaf benchmark optimizes a small Harbor-native agent that explores the
repository mounted at `/app` and writes its final answer to
`/logs/agent/answer.txt`. The editable program controls its prompt, search
strategy, shell tools, context management, and answer synthesis.

Two pinned target builds share this seed, split, budgets, and access policy;
they differ only in target model, and each pins the seed floor measured on
**its own** model — a delta against the other build's floor is a model
comparison, not an optimization result:

| build | target model | pinned `baseline_reward` |
| --- | --- | --- |
| `build.yaml` | `fireworks_ai/gpt-oss-120b` | 0.0676 (K=3, n=148) |
| `build.gpt54mini.yaml` | `gpt-5.4-mini` | 0.1324 (K=3, n=136) |

Both floors were measured with `scripts/rescore_candidate.py --seed`, the same
path that produced every other pinned baseline in the suite.

The Harbor tasks retain their canonical rubric-based verifier. That verifier
needs `OPENAI_API_BASE`; the target agent uses `OPENAI_BASE_URL`. They may
point to the same OpenAI-compatible endpoint.

Compile from the repository root:

```bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-engineering-bench/swe-atlas-qna/baseline/build.yaml \
  --output ../harness-engineering-bench/swe-atlas-qna/baseline/compiled
```

For a real run, provide `OPENAI_API_KEY`, `OPENAI_BASE_URL`,
`OPENAI_API_BASE`, and the Modal credentials declared in `build.yaml`.
