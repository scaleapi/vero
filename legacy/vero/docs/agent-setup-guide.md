# Vero Task Setup Guide (for Coding Agents)

This guide is for coding agents (Claude, Cursor, Copilot, etc.) helping users set up vero evaluation tasks. Follow these steps in order.

## Before you start

### Read and understand

1. **The user's agent code** — understand what it does, its inputs/outputs, and how it's invoked.
2. **`src/vero/core/scaffolds/vero_tasks.py`** — the scaffold template. Read it to understand the `TaskOutput`/`TaskResult` contract.
3. **The user's dataset** — what fields are in each row? What split names exist (train, test, validation)?
4. **`examples/matmul-kernel/`** — a complete working example with inference, evaluation, and a dataset.

### Ask the user

Before writing any code, clarify:

1. **What does a single evaluation sample look like?** What fields are in the dataset? (e.g., `input`, `expected_output`, `question`, `answer`)
2. **How should the agent be invoked?** Is it a function call, an API call, a subprocess? Does it need specific imports?
3. **How should outputs be scored?** Exact match? Fuzzy match? LLM-as-judge? Numeric comparison? Timing?
4. **What environment variables does inference need?** API keys, base URLs, model names?
5. **Is evaluation separate from the agent project?** Should task code live in the agent package or a separate `task_project`?

## Step-by-step setup

### 1. Verify project structure

The agent project must be a uv-managed Python package with git:

```
my-agent/
├── pyproject.toml          # Must have [project] section
├── src/my_agent/
│   ├── __init__.py
│   └── ...                 # Agent code
└── .git/
```

Run `vero check` to validate:

```bash
vero check --project-path .
```

### 2. Initialize the task scaffold

```bash
cd my-agent
vero init tasks --task main
```

This creates:

```
src/my_agent/vero_tasks/
├── __init__.py     # Imports the task module
└── main.py         # Inference + evaluation functions
```

It also adds `scale-vero` as a dependency in `pyproject.toml`.

### 3. Implement run_inference

Edit `src/my_agent/vero_tasks/main.py`. The inference function receives one dataset row and returns a `TaskOutput`:

```python
@task.inference()
async def run_inference(
    task: dict, evaluation_parameters: EvaluationParameters
) -> TaskOutput:
    # task is a dict with the dataset row fields
    # Return TaskOutput wrapping your agent's output
    ...
    return TaskOutput(output=result)
```

**Key rules:**

- `task` is a raw dict from the dataset row. Access fields like `task["question"]`.
- `evaluation_parameters.task_params` contains extra params passed via `Policy(evaluation_parameters=BaseEvaluationParameters(task_params={...}))`. Use for model name, temperature, etc.
- Return value is `TaskOutput(output=<any serializable value>)`. The `output` can be a string, dict, list, number — whatever your evaluation function needs.
- Function must be `async`. Use `await` for async agent calls.
- Do NOT catch exceptions — let them propagate so vero records them as errors.

**Common patterns:**

```python
# Simple function call
from my_agent import run_agent
result = await run_agent(task["input"])
return TaskOutput(output=result)

# LLM call via litellm
from litellm import acompletion
response = await acompletion(model="gpt-4", messages=[{"role": "user", "content": task["question"]}])
return TaskOutput(output=response.choices[0].message.content)

# OpenAI Agents SDK
from agents import Agent, Runner
agent = Agent(name="my-agent", model=model, instructions=task["instructions"])
result = await Runner.run(agent, input=task["input"])
return TaskOutput(output=result.final_output)
```

### 4. Implement run_evaluation

The evaluation function scores a single inference output:

```python
@task.evaluation()
async def run_evaluation(
    task: dict,
    output: TaskOutput,
    evaluation_parameters: EvaluationParameters,
) -> TaskResult:
    # task: the original dataset row
    # output: the TaskOutput from run_inference
    # Return TaskResult with score and optional feedback
    ...
    return TaskResult(output=prediction, score=score, feedback=feedback)
```

**Key rules:**

- `output.output` is whatever you returned from `TaskOutput(output=...)` in inference.
- `score` should be a float. Convention: higher is better (0.0 = worst, 1.0 = best) for accuracy-style tasks. For timing tasks, lower is better.
- `feedback` is an optional string shown to the optimizer agent — make it informative.
- `metrics` is an optional dict for additional tracked values.
- Return `TaskResult(output=..., score=..., feedback=..., metrics={...})`.

**Common evaluation patterns:**

```python
# Exact match
expected = task["expected"]
score = 1.0 if output.output == expected else 0.0

# Contains / regex
import re
match = re.search(r"Answer: (\w+)", output.output)
score = 1.0 if match and match.group(1) == task["answer"] else 0.0

# Numeric tolerance
score = 1.0 if abs(float(output.output) - task["expected"]) < 0.01 else 0.0

# LLM-as-judge
from litellm import acompletion
response = await acompletion(
    model="gpt-4",
    messages=[{"role": "user", "content": f"Rate this answer 0-1:\nQuestion: {task['question']}\nAnswer: {output.output}"}],
)
score = float(response.choices[0].message.content)

# Timing (lower is better)
score = output.output["time_ms"]  # Optimizer minimizes this
```

### 5. Add required_env_vars

If inference or evaluation needs API keys, declare them:

```python
task = create_task("main", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])
```

This enables early validation — `vero check` and the evaluator will catch missing vars before launching inference.

### 6. Create a dataset

Vero uses HuggingFace `DatasetDict`. Create one and save to disk:

```python
from datasets import Dataset, DatasetDict

ds = DatasetDict({
    "test": Dataset.from_dict({
        "input": ["What is 2+2?", "Capital of France?"],
        "expected": ["4", "Paris"],
    }),
    "train": Dataset.from_dict({
        "input": ["What is 3+3?", "Capital of Germany?"],
        "expected": ["6", "Berlin"],
    }),
})
ds.save_to_disk("./data/my-dataset")
```

Or pass a `DatasetDict` directly to `Policy(dataset=ds)` in Python — no need to save to disk.

### 7. Configure filesystem access (optional)

`.veroaccess` controls what the optimizer agent can read, write, or must avoid. It protects evaluation code, test data, and secrets from modification.

**Ask the user:**

- Are there files the optimizer should never touch? (test data, credentials, config)
- Are there files it should read but not modify? (test suites, task definitions)
- Should the default be "write everything" or more restrictive?

**When to set up `.veroaccess`:**

- **Always recommended** if the project has test data, evaluation code, or secrets.
- **Skip** for simple projects where everything is fair game.

**How to set up:**

```bash
# Auto-generate from project structure (scans directories, applies sensible defaults)
vero init accesses --auto

# Or use the bundled defaults
vero init accesses --default

# Or interactive mode (walk through each directory)
vero init accesses --interactive
```

This creates a `.veroaccess` file in the project root:

```
[exclude]
# Agent cannot access these at all
tests/data/**
**/__pycache__/**
.env

[read]
# Agent can read but not modify
tests/**
vero_tasks/**
.veroaccess

[write]
# Agent has full access (everything not listed above)
src/**
```

**Rules:**

- Last matching rule wins (like `.gitignore`)
- Three access levels: `exclude` (no access), `read` (read-only), `write` (read + write)
- Default for unlisted paths is `write` (configurable via `Policy(filesystem_default_access=...)`)
- `.veroaccess` itself is always read-only (enforced programmatically)

### 8. Set up VeroResources (optional)

Resources let you mark specific functions or classes for targeted optimization. Instead of the optimizer editing files by path, it edits resources by name — constraining changes to specific, declared targets.

**Ask the user:**

- Are there specific functions the optimizer should focus on? (prompts, scoring logic, model config)
- Would they prefer the optimizer to only modify marked functions, not arbitrary files?
- What namespace grouping makes sense? (e.g., "prompts", "models", "evaluators")

**When to use resources:**

- When you want to **constrain** what the optimizer can change (e.g., only the prompt function)
- When the project has many files but only a few are optimization targets
- When using `ResourceControl` tool instead of `FileWrite`

**Skip if:** the optimizer should have free rein to edit any file.

**How to set up:**

1. Decorate target functions with `@resource(namespace)`:

```python
from vero.core.resource import resource

@resource("prompts")
def system_prompt() -> str:
    """The system prompt for the agent."""
    return "You are a helpful assistant..."

@resource("prompts")
def format_output(raw: str) -> str:
    """Post-process model output."""
    return raw.strip()
```

2. Use `ResourceControl` in the tool set:

```python
from vero.tools import ResourceControl

# Only allow editing resources in the "prompts" namespace
agent = VeroAgent(tool_sets=[
    ResourceControl(allowed_namespaces={"prompts"}),
    # ... other tools
])
```

3. Pass `--resources` flag in the matmul example to see it in action.

**Key rules:**

- `@resource` is a marker — it doesn't change the function's behavior at runtime
- Resources are discovered via AST parsing (no code execution needed)
- The optimizer sees resources by `namespace.name` (e.g., `prompts.system_prompt`)
- Resource integrity is validated — the decorator can't be removed or added by the optimizer
- Multiple namespaces can coexist in the same file

### 9. Verify with vero check

> Steps 7 and 8 are optional. Continue here after implementing inference + evaluation.

```bash
vero check --project-path . --task main --dataset ./data/my-dataset
```

Expected output:

```
  [OK]   uv project found
  [OK]   Git repo: /path/to/my-agent
  [OK]   Task discovery: my_agent.vero_tasks (1 task(s))
         - main: OK
  [OK]   Env vars for 'main': all set
  [OK]   Dataset: ['test', 'train'] {'test': 2, 'train': 2}

RESULT: All checks passed
```

### 10. Run a test evaluation

```bash
vero evaluate \
  --project-path . \
  --task main \
  --dataset ./data/my-dataset \
  --split test \
  --num-samples 2
```

This runs inference + evaluation on 2 samples. Check:

- No errors in output
- Scores make sense
- Feedback messages are informative

### 11. Run optimization

```bash
vero run \
  --project-path . \
  --task main \
  --dataset ./data/my-dataset \
  --train-budget 5 \
  --max-turns 100
```

Or in Python:

```python
from vero.policy import Policy
from vero.agents.vero import VeroAgent

policy = Policy(
    project_path="./my-agent",
    dataset="./data/my-dataset",
    agent=VeroAgent(),
    task="main",
    train_budget=5,
    max_turns=100,
)
best = await policy.run()
```

## What NOT to do

- **Don't modify `vero_tasks/` during optimization.** The optimizer agent edits the agent code, not the evaluation code. Keep scoring logic immutable. Use `task_project` for external eval code if needed.
- **Don't hardcode absolute paths** in task code. Use relative imports and `evaluation_parameters.task_params` for configuration.
- **Don't suppress exceptions** in inference. Let errors propagate — vero tracks them as error samples.
- **Don't import heavy dependencies at module level** in task files. Use in-function imports for LLM clients, large libraries, etc. The task module is imported during discovery which should be fast.
- **Don't put secrets in code.** Use `required_env_vars` and environment variables.

## Reference

| Type | Description |
|------|-------------|
| `TaskOutput(output=...)` | Wraps inference output. `output` can be any serializable value. |
| `TaskResult(output=..., score=..., feedback=..., metrics={...})` | Evaluation result. `score` is float, `feedback` is str, `metrics` is dict. |
| `EvaluationParameters` | Has `.task_params` (dict), `.timeout`, `.max_concurrency`, `.run` (candidate info). |
| `create_task(name, required_env_vars=[...])` | Register a task. Name must match `Policy(task=...)`. |
| `@task.inference()` | Decorator for inference function. Signature: `(task: dict, evaluation_parameters) -> TaskOutput`. |
| `@task.evaluation()` | Decorator for evaluation function. Signature: `(task: dict, output: TaskOutput, evaluation_parameters) -> TaskResult`. |
