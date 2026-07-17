# tau3 MCP customer-service agent

This leaf benchmark optimizes a Harbor-native agent that connects to the MCP
server declared by each tau3 task, obtains the first simulated-user message,
and carries the conversation through domain and communication tool calls. The
editable program controls its prompt, tool selection, context management, and
conversation policy; the task-owned MCP runtime and canonical evaluator remain
outside the candidate repository.

The trusted build pins the target model to `gpt-5.4-mini-2026-03-17` and pins
the Harbor dataset, split, budgets, access policy, and final test partition.
The simulated user and natural-language assertion grader use the canonical
model defaults encoded in the pinned tau3 tasks.

Compile from the repository root:

```bash
cd vero
VERO_SKIP_SECRET_CHECK=1 uv run vero harbor build \
  --config ../program-opt-bench/tau3/baseline/build.yaml \
  --output ../program-opt-bench/tau3/baseline/compiled
```

For a real run, provide the OpenAI and Modal credentials declared in
`build.yaml`.
