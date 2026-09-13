# tau3 editable target

The seed connects to the MCP server declared by each task, obtains the first
simulated-user message, and carries the conversation through domain and
communication tool calls.

The optimizer may change the prompt, tool selection, context management,
conversation policy, and dependencies in `target/`. The task-owned runtime,
simulated user, evaluator, dataset, partitions, and target model remain fixed.

## Compile

From the repository root:

~~~bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../harness-opt-bench/archive/tau3/baseline/build.yaml \
  --param inner_env=<evaluation-environment> \
  --output <output-directory>
~~~

The `VERO_SKIP_SECRET_CHECK` setting is appropriate only for compile-time
validation. Dataset and split details are in the
[tau3 overview](../README.md). For a real optimization run, follow the shared
[`run-benchmark` guide](../../../skills/run-benchmark/SKILL.md).
