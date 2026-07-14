# VeRO: A Harness for Agents to Optimize Programs

[![arXiv](https://img.shields.io/badge/arXiv-2602.22480-b31b1b.svg)](https://arxiv.org/abs/2602.22480)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-4b44ce.svg)](https://arxiv.org/abs/2602.22480)

VeRO gives coding agents a feedback loop for improving code:

1. An optimizer edits a program.
2. VeRO saves the candidate and runs your evaluator.
3. The optimizer sees the scores, errors, and execution traces.
4. The loop repeats until VeRO finds the best version within your budget.

The target does not have to be an agent—or even Python. It can be an LLM harness, a C
kernel, a prompt pipeline, a compiler pass, or any other program you can edit and
measure. Your evaluator can run tests, benchmarks, simulations, datasets, or sandboxed
tasks. You choose the objective; VeRO manages versioning, budgets, and observations.

VeRO stands for **Versioning, Rewards, and Observations**. It was introduced in the
ICML 2026 paper [**VeRO: A Harness for Agents to Optimize
Agents**](https://arxiv.org/abs/2602.22480), where the default VeRO-Agent improved the
average best score from a 0.50 baseline to 0.61 across five tasks. This repository
generalizes that same loop from optimizing agents to optimizing programs.

> **Want to see it work?** Try the [C matrix multiplication example](examples/c-matmul/),
> where VeRO improves compiled code without requiring the target to be an agent or a
> Python package.

## Quickstart

Give VeRO three things: the program it may edit, a command that evaluates the program,
and an optimizer that proposes changes. For example, this configuration optimizes a
program for latency while requiring it to remain correct:

```toml
# vero.toml
[target]
root = "./target"

[evaluation]
backend = "command"
harness_root = "./evaluator"
command = [
  "./evaluate",
  "--workspace", "{workspace}",
  "--request", "{request}",
  "--report", "{report}",
]
evaluation_set = "performance"

[objective]
metric = "latency_ms"
direction = "minimize"

[[objective.constraints]]
metric = "correct"
operator = "=="
value = 1.0

[optimizer]
root = "./optimizer"
command = ["./optimize", "--workspace", "{workspace}"]
max_candidates = 3
```

First, evaluate the current program. Then run the optimization loop:

```bash
vero evaluate --config vero.toml
vero run --config vero.toml
```

The evaluation command receives the candidate workspace and writes a JSON report. It
can compile, test, benchmark, simulate, or otherwise measure the candidate. The
optimizer can be a coding agent or any command that edits the workspace.

See [`examples/c-matmul`](examples/c-matmul/) for an end-to-end C target with no Python
package, VeRO dependency, dataset, or agent framework. The deterministic example
optimizer can be replaced by a coding-agent command.

### Agent optimization

The existing dataset/VeroTask path remains available for optimizing Python agents:

```python
from agents import Agent as OAIAgent
from vero.policy import Policy
from vero.agents.vero import VeroAgent

policy = Policy(
    project_path="/path/to/my-agent",
    dataset="/path/to/my-dataset",
    agent=VeroAgent(
        oai_agent=OAIAgent(name="VeroAgent", model="anthropic/claude-sonnet-4-5-20250929"),
    ),
    task="main",
    train_budget=10,
    max_turns=200,
    enable_wandb=True,
    wandb_project="my-optimization",
)

best = await policy.run()
print(f"Best commit: {best.commit}, score: {best.score}")
```

## Installation

### Pre-requisites

- Python 3.11+
- `uv` ([install](https://docs.astral.sh/uv/getting-started/installation/))
- `git`

An LLM provider is only required when the optimizer or target uses one. The command
backend and non-agent targets do not require an LLM dependency or API key.

### From PyPI

```bash
uv pip install scale-vero[optimize]
```

### From source

```bash
cd ~/vero/vero
uv sync --extra optimize
source .venv/bin/activate
```

### Optional Dependencies

| Group | Install Command | Description |
| ----- | --------------- | ----------- |
| `optimize` | `scale-vero[optimize]` | Full optimization machinery (agents, policies, tools) |
| `claude` | `scale-vero[claude]` | Claude Agent SDK for ClaudeCodeAgent |
| `jupyter` | `scale-vero[jupyter]` | Jupyter notebooks |
| `wandb` | `scale-vero[wandb]` | Weights & Biases experiment tracking |

Combine groups: `uv pip install scale-vero[optimize,claude,wandb]`

## Core Concepts

### Program policy

`ProgramPolicy` owns the single-producer optimization loop. It evaluates the baseline,
asks a candidate producer to edit the versioned workspace, commits each change,
evaluates it through an approved backend, and selects the best feasible result using
the declared objective. Evaluation is independent of candidate production, so future
population and evolutionary strategies can reuse the same engine.

### Policy

`Policy` supports the generic constructor (`optimizer`, `backends`, `evaluation_set`,
and `objective`) without a dataset or task. Its compatibility constructor still accepts
`agent`, `dataset`, and `task` for dataset-backed VeroTask optimization and optional W&B
logging. Both constructors write the same schema-v2 `EvaluationRecord`; the dataset API
returns a deprecated `Experiment` view of that record.

Canonical evaluations are also available directly:

```python
record = await policy.evaluate_candidate(
    commit="abc123",
    backend_id="performance",
    evaluation_set=evaluation_set,
)
print(record.objective.value, record.objective.feasible)
```

```python
from vero.policy import Policy

policy = Policy(
    project_path="/path/to/agent",     # Git repo with a uv package
    dataset="/path/to/dataset",         # HuggingFace DatasetDict on disk
    agent=agent,                        # VeroAgent or ClaudeCodeAgent
    task="main",                        # Task name from vero_tasks module
    train_budget=10,                    # Evaluation runs on train split
    validation_budget=10,               # Evaluation runs on validation split
    max_turns=200,                      # Max optimization turns
    enable_wandb=True,                  # Enable wandb logging
    instructions_template="instructions/few_shot_instructions.j2",
    prompt_template="prompts/simple_prompt.j2",
)
```

Run the full optimization loop:

```python
best = await policy.run()
```

Or for interactive use (e.g. notebooks), call `init()` and `finish()` manually:

```python
await policy.init()
await policy.step()
best = policy.get_best_version()
policy.finish()
```

### Agents

Agents are execution backends that implement the optimization step:

**VeroAgent** — Uses the OpenAI Agents SDK with orchestrator + sub-agent architecture:

```python
from agents import Agent as OAIAgent
from vero.agents.vero import VeroAgent, default_tool_sets

agent = VeroAgent(
    oai_agent=OAIAgent(name="VeroAgent", model="anthropic/claude-sonnet-4-5-20250929"),
    tool_sets=default_tool_sets(),
)
```

**ClaudeCodeAgent** — Uses the Claude Agent SDK (Claude Code):

```python
from vero.agents.claude_code import ClaudeCodeAgent
from vero.tools import DatasetViewer, ExperimentRunnerTool, ExperimentViewer

agent = ClaudeCodeAgent(
    tool_sets=[DatasetViewer(), ExperimentRunnerTool(), ExperimentViewer()],
)
```

### Session

`Session` is a lightweight context that agents and tools bind to. Policy creates it automatically during `init()`, but you can also create one directly for testing or standalone use:

```python
from vero.policy import Session

# Minimal session for testing (no workspace, db, or evaluator)
session = Session(session_id="test", project_path=Path("/my/project"))
agent.init(session)
await agent.step("optimize this code", max_turns=10)

# Session with workspace
session = Session(
    session_id="test",
    project_path=tmp_path,
    workspace=my_workspace,
    instructions="Be helpful.",
)
```

#### Agent State Serialization

Agents support state serialization for resumption:

```python
# Save state after a run
state = agent.serialize_state()
# VeroAgent: conversation history (list of message dicts)
# ClaudeCodeAgent: {"session_id": "..."} (server-side session reference)

# Restore state in a new agent
agent2 = VeroAgent(tool_sets=[])
agent2.init(session)
agent2.deserialize_state(state)
await agent2.step("continue from where you left off", max_turns=10)
```

### ToolSets

ToolSets are pre-created instances that self-wire to session resources via `bind()`.
They implement the `ToolSet` protocol and carry an `exclude_tools` field to control which tool methods are exposed:

| Tool | Description |
| ---- | ----------- |
| `BashTool` | Execute bash commands |
| `FileRead` | Read files |
| `FileWrite` | Write or edit files |
| `Grep` | Search files |
| `GitViewer` | View git state |
| `GitControl` | Create branches, commits |
| `ExperimentRunnerTool` | Run experiments on dataset subsets |
| `ExperimentViewer` | View experiment results |
| `DatasetViewer` | Explore dataset samples |
| `EvaluationRunnerTool` | Evaluate a generic program candidate |
| `EvaluationViewer` | View canonical summaries, reports, cases, and artifacts |
| `WebSearch` | Search the web |
| `WebFetch` | Fetch web pages |
| `ContextStore` | Key-value store for agent context |
| `TodoList` | Track tasks |
| `think` | Extended reasoning |
| `ResourceControl` | View and edit VeroResources |

Configure tool sets on VeroAgent:

```python
from agents import Agent as OAIAgent
from vero.agents.vero import VeroAgent
from vero.tools import BashTool, FileRead, ExperimentRunnerTool, ContextStore

agent = VeroAgent(
    oai_agent=OAIAgent(name="VeroAgent", model="anthropic/claude-sonnet-4-5-20250929"),
    tool_sets=[BashTool(), FileRead(), ExperimentRunnerTool(), ContextStore()],
)
```

### Sessions and Results

All data is stored under `~/.vero/sessions/{session_id}/`:

```
sessions/{session_id}/
├── experiments/{result_id}/
│   ├── evaluation.json              # Schema-v2 request, report, provenance, objective
│   ├── cases/                       # Hashed canonical case checkpoints
│   ├── artifacts/                   # Logs, profiles, compiler output, traces
│   └── backend/                     # Backend-private files
├── database.json                    # Schema-v2 evaluation index
├── agent_trace/                     # Per-turn agent event log
│   ├── turn_0000.json
│   ├── turn_0001.json
│   └── ...
├── config.json
└── result.json
```

Legacy VeroTask sessions using `evaluation_parameters.json` and `samples/` remain
readable through compatibility adapters.

The `experiments/` directory is the source of truth; `database.json` can be rebuilt from
the canonical manifests and case files.

### Resuming and Forking Sessions

Resume an existing session (reconnects to the same project and experiments):

```python
resumed = Policy.resume(
    session_id="abc-123",
    agent=VeroAgent(oai_agent=OAIAgent(name="VeroAgent", model="anthropic/claude-sonnet-4-5-20250929")),
    dataset="/path/to/dataset",
    task="main",
)
async with resumed:
    # DB is reconstructed from experiments on disk
    # Use skip_initial_eval=True since baseline already exists
    best = await resumed.run(skip_initial_eval=True)
```

Fork creates a copy of the project and experiments in a new session:

```python
forked = Policy.fork(
    source_session_id="abc-123",
    agent=VeroAgent(oai_agent=OAIAgent(name="VeroAgent", model="anthropic/claude-sonnet-4-5-20250929")),
    dataset="/path/to/dataset",
    task="main",
)
async with forked:
    # Independent copy — changes don't affect the source session
    best = await forked.run(skip_initial_eval=True)
```

### Exposing Artifacts to the Filesystem

Use `artifacts` to materialize data into the agent's worktree as read-only files under `_vero/`:

```python
from vero.artifacts import DatasetArtifact, TracesArtifact, SkillsArtifact

policy = Policy(
    ...,
    artifacts=[
        DatasetArtifact(),   # Write viewable splits to _vero/datasets/
        TracesArtifact(),    # Write experiment results to _vero/traces/ after each eval
        SkillsArtifact(),    # Copy skills to _vero/skills/
    ],
)
```

This creates a `_vero/` directory in the worktree:

```
<worktree>/
├── _vero/
│   ├── datasets/<dataset_id>/train/   # Only viewable splits
│   │   ├── 0.json
│   │   └── 1.json
│   ├── traces/<split>__<commit>/      # Appears after each eval
│   │   ├── summary.json
│   │   ├── 0.json
│   │   └── 1.json
│   └── skills/<namespace>/            # Copied from skills paths
│       └── *.md
└── ...
```

The agent can `cat` these files but cannot write to them (enforced by `.veroaccess` READ rules and Claude Code's `disallowed_tools`). Non-viewable splits are never materialized.

### Event Callbacks

Register callbacks to observe agent events in real-time:

```python
policy = Policy(
    ...,
    on_event=[my_callback],  # Called with serialized event dict for each agent turn
)
```

A built-in `SessionLogger` is auto-registered to write per-turn JSON files to `agent_trace/`. Events are flushed immediately for crash safety.

## Quickstart: Zero to First Eval

```bash
# 1. Create a new agent project
uv init my-agent && cd my-agent
uv add scale-vero

# 2. Scaffold the evaluation task
vero init tasks --task main

# 3. Create a test dataset
uv run python -c "
from datasets import Dataset, DatasetDict
ds = DatasetDict({
    'test': Dataset.from_dict({
        'input': ['hello', 'world'],
        'expected': ['hello', 'world'],
    })
})
ds.save_to_disk('./data')
"

# 4. Verify setup
vero check --task main --dataset ./data

# 5. Run evaluation (uses the scaffold's echo + exact-match default)
vero evaluate --project-path . --task main --dataset ./data --split test

# 6. Edit src/my_agent/vero_tasks/main.py with your real inference + evaluation logic

# 7. Run optimization
vero run --project-path . --task main --dataset ./data --train-budget 5
```

### Using a coding agent for setup

If you're using a coding agent (Claude Code, Cursor, Copilot, etc.) to set up your vero tasks, point it at [`docs/agent-setup-guide.md`](docs/agent-setup-guide.md). It contains a step-by-step workflow designed for coding agents: what to read first, what to ask you, how to implement inference/evaluation, common patterns, and verification steps.

### Filesystem access (optional)

Create a `.veroaccess` file to control what the optimizer agent can modify:

```bash
vero init accesses --auto
```

Works like `.gitignore` but for agent permissions — `[exclude]` blocks access, `[read]` allows read-only, `[write]` allows full access. See the [agent setup guide](docs/agent-setup-guide.md#7-configure-filesystem-access-optional) for details.

### VeroResources (optional)

Mark specific functions for targeted optimization with `@resource("namespace")`. The optimizer edits resources by name instead of by file path. See the [agent setup guide](docs/agent-setup-guide.md#8-set-up-veroresources-optional) for details.

## Running Optimization

```python
from agents import Agent as OAIAgent
from vero.policy import Policy
from vero.agents.vero import VeroAgent

policy = Policy(
    project_path="/path/to/my-agent",
    dataset="/path/to/my-dataset",
    agent=VeroAgent(
        oai_agent=OAIAgent(name="VeroAgent", model="anthropic/claude-sonnet-4-5-20250929"),
    ),
    task="main",
    train_budget=10,
    max_turns=200,
)

best = await policy.run()
print(f"Best commit: {best.commit}, score: {best.score}")
```

## Policy Configuration Reference

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `project_path` | `Path \| str` | **Required.** Path to the agent project |
| `dataset` | `Path \| str \| dict` | **Required.** Dataset path, or `{id: path}` dict |
| `agent` | `Agent` | **Required.** `VeroAgent` or `ClaudeCodeAgent` |
| `task` | `str` | Task name from `vero_tasks` module |
| `task_project` | `Path \| str` | Separate uv project for task/eval code (default: same as `project_path`) |
| `task_module` | `str` | Explicit Python module for task registration (default: auto-discover from `{package}.vero_tasks`) |
| `isolate` | `bool` | Copy project into a fresh git repo before optimizing (default: `False`) |
| `train_budget` | `int` | Evaluation runs on train split |
| `validation_budget` | `int` | Evaluation runs on validation split |
| `max_turns` | `int` | Max optimization turns (default: 200) |
| `enable_wandb` | `bool` | Enable wandb logging (default: `False`) |
| `wandb_project` | `str` | Wandb project name |
| `instructions_template` | `str` | Path to Jinja2 instructions template |
| `prompt_template` | `str` | Path to Jinja2 prompt template |
| `ref` | `str` | Branch, tag, or commit to start from (default: `"main"`) |
| `skills` | `Path \| str \| dict` | Skills/cookbook artifacts, or `{namespace: path}` dict |
| `evaluation_parameters` | `BaseEvaluationParameters` | Timeout, concurrency, retry settings |
| `filesystem_accesses` | `list[AccessRule]` | Programmatic filesystem access rules |
| `artifacts` | `list[FileSystemArtifact]` | Artifacts to materialize in `_vero/` — `DatasetArtifact()`, `TracesArtifact()`, `SkillsArtifact()` (default: `[]`) |
| `sandbox` | `Sandbox` | Custom sandbox for non-local environments (default: `LocalSandbox` at `~`) |
| `vero_home` | `Path \| str` | Vero home directory for sessions/datasets (default: `~/.vero`) |
| `optimizer_env_file` | `Path \| str` | Path to `.env` file loaded into the parent process at `init()` |
| `subprocess_env_vars` | `list \| Path \| str` | Env var names to forward, OR path to `.env` file for eval subprocesses |
| `on_event` | `list[Callable]` | Callbacks fired with serialized event dicts during agent execution |
| `metadata` | `dict` | Arbitrary metadata (logged to wandb, included in `as_dict()`) |

## Environment Variables

Vero runs LLM calls in two contexts that may need different credentials:

1. **Optimizer process** — the VeroAgent or ClaudeCodeAgent that reads code and decides what to change. Runs in the parent Python process.
2. **Evaluation subprocess** — the agent-under-test that runs inference on dataset samples. Runs in an isolated `uv run` subprocess.

By default, subprocesses inherit `os.environ`, so if your env vars are already set in the shell, everything works. For explicit control:

```python
# Load .env into the parent process (optimizer LLM calls)
policy = Policy(
    optimizer_env_file=".env.optimizer",
    ...
)

# Load .env into evaluation subprocesses only
policy = Policy(
    subprocess_env_vars=".env.eval",
    ...
)

# Or forward specific vars by name (existing behavior)
policy = Policy(
    subprocess_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"],
    ...
)
```

From the CLI:

```bash
vero run --env-file .env.optimizer --subprocess-env-file .env.eval ...
```

Only the file *path* is stored in session config — env var values are never serialized to logs, wandb, or trace files.

Tasks can declare required env vars for early validation:

```python
task = create_task("gpqa", required_env_vars=["LITELLM_BASE_URL", "LITELLM_API_KEY"])
```

The evaluator checks these after task discovery, before launching the subprocess — failing fast with a clear message instead of a cryptic error deep in the LLM call.

## External Task Projects

By default, evaluation tasks live inside the agent's package (`{agent_package}.vero_tasks`). For benchmarking integrity — where the optimizer should not be able to modify scoring logic — tasks can live in a separate uv project:

```python
policy = Policy(
    project_path="/path/to/my-agent",
    dataset="/path/to/dataset",
    agent=agent,
    task="math",
    task_project="/path/to/eval-tasks",         # Separate project with scoring logic
    task_module="my_eval_tasks.vero_tasks",      # Module to import for task registration
)
```

When `task_project` is set, the evaluator runs `uv run --project eval-tasks --with-editable /agent/worktree`, layering the agent code at the correct commit on top of the task project's environment. This ensures the task can import agent code while keeping scoring logic immutable.

## Sandbox and Workspace

**Sandbox** is a pure I/O layer representing the execution environment (the "computer"). It provides file read/write, shell execution, and file transfer — no access control. The default `LocalSandbox` wraps `pathlib.Path` and `asyncio.create_subprocess_exec`. Custom implementations can target containers, remote VMs, etc.

**Workspace** layers version control and access control on top of a Sandbox. `GitWorkspace` uses `sandbox.run(["git", ...])` for all git operations. Access rules (from `.veroaccess`) are configured on the workspace, not the sandbox — tools call `workspace.validate_read()` for access checks and `sandbox.read_file()` for I/O.

```python
from vero.sandbox import LocalSandbox
from vero.workspace.git import GitWorkspace

# Create sandbox (the computer) and workspace (the project)
sandbox = await LocalSandbox.create()
workspace = await GitWorkspace.from_path(sandbox, "/path/to/project")

# Access control is on the workspace
workspace.set_access(accesses=[...], default_access=AccessType.WRITE)
```

Pass a custom sandbox to Policy for non-local environments:

```python
policy = Policy(
    sandbox=my_docker_sandbox,
    project_path="/workspace/my-agent",
    ...
)
```

## VeroResources

VeroResources let you mark specific functions for agent optimization:

```python
from vero.core.resource import resource

@resource(namespace="prompts")
def system_prompt() -> str:
    return "You are a helpful assistant..."
```

The agent can discover and edit these resources directly, without needing to understand file structure.

Enable resource tools:

```python
from agents import Agent as OAIAgent
from vero.agents.vero import VeroAgent
from vero.tools import ResourceControl

agent = VeroAgent(
    oai_agent=OAIAgent(name="VeroAgent", model="anthropic/claude-sonnet-4-5-20250929"),
    tool_sets=[ResourceControl(allowed_namespaces={"prompts"})],
)
```

## Examples

See [`examples/matmul-kernel/`](examples/matmul-kernel/) for a complete runnable example that optimizes a matrix multiply kernel for speed. It demonstrates eval-only mode, full optimization with VeroAgent or Claude Code, filesystem artifacts, and resource-based editing.
