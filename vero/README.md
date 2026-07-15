# VeRO: a harness for agents to optimize programs

VeRO gives an optimizer a program to edit, a controlled way to evaluate it, and
durable memory of everything it tried. The target can be an agent, a prompt, a
compiler pass, a CUDA kernel, a matrix multiplication function, or any other
Git-versioned program.

The target and evaluator do not need to be Python. VeRO's built-in command
backend communicates with an external evaluation harness through versioned JSON,
and candidate changes can come from a coding agent, an external command, or a
custom optimization strategy.

```text
strategy proposes ideas
        ↓
producers edit isolated candidate workspaces (in parallel if desired)
        ↓
evaluation backend measures each version
        ↓
selection keeps the best candidate and the next round continues
```

## Quickstart

Install VeRO and make sure the target is a clean Git repository:

```bash
uv pip install scale-vero
git -C ./my-program status --short
```

Write an evaluation harness outside the target repository. VeRO invokes it with
an isolated candidate workspace and a path where it must write a report:

```python
# ../my-evaluator/evaluate.py
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
report_path = Path(sys.argv[2])

# Build, run, benchmark, call another service, etc.
latency_ms = measure(workspace)

report_path.write_text(json.dumps({
    "schema_version": 1,
    "status": "success",
    "metrics": {"latency_ms": latency_ms},
}))
```

Then choose how candidates are changed.

### Optimize with a coding agent

```bash
vero optimize ./my-program \
  --harness-root ../my-evaluator \
  --evaluate 'python3 evaluate.py {workspace} {report}' \
  --agent claude \
  --instruction 'Make the program faster without changing its output' \
  --metric latency_ms \
  --direction minimize \
  --max-candidates 5
```

Use `--agent vero` for VeRO's OpenAI Agents SDK implementation. Provider-specific
dependencies and credentials are required for either built-in coding agent.

### Optimize with any external producer

An external producer receives an isolated workspace and edits it in place:

```bash
vero optimize ./my-program \
  --harness-root ../my-evaluator \
  --evaluate 'python3 evaluate.py {workspace} {report}' \
  --producer-root ../my-optimizer \
  --produce 'python3 improve.py {workspace}' \
  --metric latency_ms \
  --direction minimize
```

Commands are parsed into argument vectors, not executed through a shell. Use
absolute executable paths when the executable is not on the standard system
`PATH`. Available evaluation placeholders are `{workspace}`, `{request}`,
`{report}`, and `{artifacts}`.

## Python API

The same pipeline can be assembled from backend-neutral interfaces:

```python
from pathlib import Path

from vero.evaluation import (
    CommandBackend,
    CommandBackendConfig,
    MetricSelector,
    ObjectiveSpec,
)
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
)
from vero.runtime import create_local_optimization_session

backend = CommandBackend(CommandBackendConfig(
    harness_root=str(Path("../my-evaluator").resolve()),
    command=["python3", "evaluate.py", "{workspace}", "{report}"],
))
producer = CommandCandidateProducer(CommandCandidateProducerConfig(
    root=str(Path("../my-optimizer").resolve()),
    command=["python3", "improve.py", "{workspace}"],
))

session = await create_local_optimization_session(
    project_path="./my-program",
    session_dir="~/.vero/sessions/my-run",
    backend_id="command",
    backend=backend,
    objective=ObjectiveSpec(
        selector=MetricSelector(metric="latency_ms"),
        direction="minimize",
    ),
    producers={"default": producer},
    max_candidates=5,
)
result = await session.run()
print(result.best.request.candidate.version, result.best.objective.value)
```

### Python benchmark tasks

Python targets can use the optional `scale-vero-tasks` package instead of
writing the JSON command contract directly. It provides only task definition
and execution types; target programs do not depend on the VeRO optimizer.

```python
# ../my-evaluator/benchmark.py
from vero_tasks import TaskOutput, TaskResult, create_task
from my_program import run_program

task = create_task("quality")

@task.inference()
async def run(case, context):
    return TaskOutput(output=run_program(case["input"]))

@task.evaluation()
async def score(case, output, context):
    return TaskResult.from_task_output(
        output,
        score=float(output.output == case["expected"]),
    )
```

Connect it with `PythonTaskBackend`. Keep the task module and cases in a trusted
external uv project; VeRO overlays each isolated candidate with
`uv --with-editable` so the harness imports the exact program version being
measured without making evaluator code editable.

```python
from vero.evaluation import PythonTaskBackend, PythonTaskBackendConfig

backend = PythonTaskBackend(PythonTaskBackendConfig(
    harness_root=str(Path("../evaluation-state").resolve()),
    cases_path=str(Path("../cases.jsonl").resolve()),
    module="benchmark",
    task="quality",
))
```

### Harbor tasks

Harbor is also an evaluation backend, rather than a separate optimization
runtime. Map each `EvaluationSet` case to one Harbor task and pin the Harbor
package used to orchestrate the nested run:

```json
{"id": "task-1", "task_name": "org/terminal-task-1"}
{"id": "task-2", "task_name": "org/terminal-task-2"}
```

```python
from vero.harbor import HarborBackend, HarborBackendConfig

backend = HarborBackend(HarborBackendConfig(
    task_source="org/terminal-benchmark@1.0",
    agent_import_path="my_program.agent:Agent",
    cases_path=str(Path("../harbor-cases.jsonl").resolve()),
    harbor_requirement="harbor==0.1.17",
    evaluation_set_name="terminal-benchmark",
    partition="test",
    passthrough_environment=["ANTHROPIC_API_KEY"],
))
```

VeRO invokes `harbor run` without importing Harbor into the core library,
collates verifier rewards into schema-v1 case results, zero-fills dead attempts
for mean aggregation, and preserves Harbor output as evaluation artifacts. The
pinned Harbor overlay protects against a candidate changing its dependency pin;
it is not a process isolation boundary because candidate code still runs inside
the nested Harbor process. Use Harbor's external verifier/sidecar deployment
when the candidate itself is adversarial.

`EvaluationBackend`, `CandidateProducer`, `OptimizationStrategy`, and
`SelectionPolicy` are protocols. Implement them to connect a remote evaluator,
a non-Git version store, an evolutionary search algorithm, or an orchestrator
that delegates proposals to several specialized producers.

## Core concepts

| Concept | Meaning |
| --- | --- |
| `Candidate` | A program identity plus an opaque workspace version and lineage |
| `EvaluationSet` | A backend-owned collection or selection of evaluation cases |
| `EvaluationRecord` | The durable request, report, provenance, and objective result |
| `EvaluationBackend` | Measures a candidate without assuming its language or framework |
| `CandidateProducer` | Edits one isolated workspace to realize a proposed idea |
| `OptimizationStrategy` | Chooses parents, ideas, and producers for the next batch |
| `SelectionPolicy` | Chooses the best feasible evaluation for the configured objective |
| `OptimizationSession` | Owns lifecycle, events, artifacts, budgets, and durable state |

Coding agents receive a scoped `AgentContext`. They can edit only their supplied
workspace and request evaluation through `evaluate_current()`. Authorization,
budgeting, and disclosure are enforced by the evaluation engine; an agent may
receive a full record, an aggregate summary, or only an acknowledgement.
Intermediate checkpoints are real candidates and remain eligible for selection,
even if the agent later makes the program worse.

Strategies can propose a batch of candidates and route each proposal to a named
producer. Set `max_concurrency` to produce and evaluate independent candidates in
parallel. This supports sequential hill climbing, evolutionary search, and
orchestrator/sub-agent designs without changing the evaluation model.

## Durable sessions

Session state is stored outside the target repository:

```text
sessions/<session-id>/
├── manifest.json
├── database.json
├── budgets.json          # when evaluation is metered
├── events.jsonl
├── artifacts/
└── evaluations/
    └── <evaluation-id>/
        ├── evaluation.json
        ├── cases/
        └── artifacts/
```

The evaluation directories are the source of truth; the database can be rebuilt
from them. Reusing a session directory resumes its compatible baseline and
evaluation history.

```bash
vero session list
vero session inspect ~/.vero/sessions/<session-id>
```

## Safety boundaries

- Candidate changes happen in isolated Git worktrees; the original target is not
  edited.
- The target must be clean so its baseline has an unambiguous version.
- Session state, evaluation harnesses, and external producers must live outside
  the editable target repository.
- Command execution uses argument vectors and an explicit environment.
- Evaluation secrets are passed through backend configuration and redacted from
  diagnostics; they cannot be embedded in evaluation parameters.
- Budgets are reserved atomically before backend execution.

## Paper and reproduction

VeRO was introduced in [*VeRO: A Harness for Agents to Optimize
Agents*](https://arxiv.org/abs/2602.22480), accepted at ICML 2026. The paper
studies agent-harness optimization; the current library generalizes the same
version/evaluate/select loop to programs more broadly.

The frozen code for reproducing the paper is preserved on the `paper/v1` branch
and at the `paper-v1` tag:

```bash
git checkout paper-v1
```

## Development

```bash
uv sync --all-extras
uv run pytest tests/test_v05_*.py
```

VeRO is licensed under the MIT License.
