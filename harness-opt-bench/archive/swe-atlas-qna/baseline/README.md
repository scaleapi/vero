# SWE-Atlas-QnA editable target

The seed explores the repository mounted at `/app` and writes its answer to
`/logs/agent/answer.txt`. The optimizer may change its prompt, search strategy,
shell tools, context management, answer synthesis, and dependencies.

Two builds share the same target, split, budgets, and access policy:

| Build | Target model | Pinned seed baseline |
| --- | --- | ---: |
| `build.yaml` | `gpt-oss-120b` | 0.0667 |
| `build.gpt54mini.yaml` | `gpt-5.4-mini` | 0.1216 |

Each baseline belongs to its configured model. Comparing a candidate against the
other build's baseline would mix harness improvement with a model change.
Candidate-caused failures score zero; evaluation-system failures are excluded.

## Compile

From the repository root:

~~~bash
cd vero
uv run vero harbor build \
  --config ../harness-opt-bench/archive/swe-atlas-qna/baseline/build.yaml \
  --param inner_env=<evaluation-environment> \
  --output <output-directory>
~~~

Use `build.gpt54mini.yaml` to compile the alternate target-model build.

Dataset and split details are in the
[SWE-Atlas-QnA overview](../README.md). For a real optimization run, follow the
shared [`run-benchmark` guide](../../../skills/run-benchmark/SKILL.md).
