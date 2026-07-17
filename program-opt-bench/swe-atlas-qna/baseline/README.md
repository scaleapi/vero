# SWE-Atlas-QnA codebase agent

This leaf benchmark optimizes a small Harbor-native agent that explores the
repository mounted at `/app` and writes its final answer to
`/logs/agent/answer.txt`. The editable program controls its prompt, search
strategy, shell tools, context management, and answer synthesis.

The trusted build pins the target model to `gpt-5.4-mini-2026-03-17`, the
dataset version, split, budgets, access policy, and final test partition. The
Harbor tasks retain their canonical rubric-based verifier. That verifier needs
`OPENAI_API_BASE`; the target agent uses `OPENAI_BASE_URL`. They may point to
the same OpenAI-compatible endpoint.

Compile from the repository root:

```bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../program-opt-bench/swe-atlas-qna/baseline/build.yaml \
  --output ../program-opt-bench/swe-atlas-qna/baseline/compiled
```

For a real run, provide `OPENAI_API_KEY`, `OPENAI_BASE_URL`,
`OPENAI_API_BASE`, and the Modal credentials declared in `build.yaml`.
