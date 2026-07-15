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

Install VeRO, then try the checked-in C matrix multiplication example. Its
editable target contains only C; a trusted external harness compiles it, checks
correctness, and measures latency.

```bash
uv pip install scale-vero
cd examples/c-matmul/target
git init -b main
git add .
git -c user.name=vero -c user.email=vero@localhost commit -m baseline
cd ..

vero evaluate --config vero.toml
vero run --config vero.toml
```

The example is deterministic and needs no model credentials. VeRO evaluates the
baseline, gives an isolated worktree to the configured producer, evaluates its
commit, selects the faster feasible result, and leaves the original target
untouched. See [`examples/c-matmul`](examples/c-matmul/) for the complete target,
harness, optimizer, and config.

## Configure an optimization

`vero.toml` is the shortest path from a program to a repeatable optimization:

```toml
[target]
root = "./my-program"
ref = "HEAD"

[evaluation]
harness_root = "../my-evaluator"
command = ["python3", "evaluate.py", "{workspace}", "{report}"]
evaluation_set = "performance"

[objective]
metric = "latency_ms"
direction = "minimize"

[[objective.constraints]]
metric = "correct"
operator = "=="
value = 1.0

[optimizer]
kind = "claude"
instruction = "Make the program faster without changing its output"
max_candidates = 5

[session]
directory = "../runs/my-program"
```

Run `vero evaluate` to measure only the baseline or `vero run` to produce and
evaluate candidates. Paths are resolved relative to the config file. A target
must be a clean Git repository, while the session directory, evaluation harness,
and command producer must live outside it.

The evaluator receives an isolated candidate workspace and paths for a
versioned request and report:

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
`{report}`, `{artifacts}`, and `{harness}`. External producers additionally get
`{producer}`. The latter two resolve to staged sandbox paths when the target is
not host-visible.

The flag-based `vero optimize` command exposes the same objective constraints,
case selection, target ref, timeouts, environments, and concurrency controls;
run `vero optimize --help` for the full surface.

## Track runs with Weights & Biases

Install the optional integration and add a section to `vero.toml`:

```bash
uv pip install 'scale-vero[wandb]'
```

```toml
[wandb]
project = "program-optimization"
entity = "my-team"       # optional
name = "matmul-v1"       # optional
mode = "online"          # online, offline, or disabled
tags = ["c", "latency"]
```

Each VeRO session maps to a stable W&B run. It logs canonical report metrics,
objective value and feasibility, case counts, candidate and evaluation IDs, and
the final baseline/best summary. Resuming the VeRO session resumes the same W&B
run. The direct CLI equivalent is `--wandb-project`, with optional entity, name,
and mode flags.

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

### Run the target in a remote sandbox

The local factory above is a convenience wrapper. For containers, remote VMs,
or another execution environment, provision a `Workspace` in that sandbox and
pass it to the generic factory:

```python
from vero.runtime import create_optimization_session
from vero.sandbox import DockerSandbox
from vero.workspace import GitWorkspace

sandbox = await DockerSandbox.create(image="gcc:14-bookworm")
try:
    # The repository is copied into the container; it is not bind-mounted.
    await sandbox.upload("./my-program", "/workspace/my-program")
    workspace = await GitWorkspace.from_path(
        sandbox,
        "/workspace/my-program",
    )

    backend = CommandBackend(CommandBackendConfig(
        harness_root=str(Path("../my-evaluator").resolve()),
        command=[
            "sh",
            "{harness}/evaluate.sh",
            "{workspace}",
            "{report}",
            "{artifacts}",
        ],
    ))
    producer = CommandCandidateProducer(CommandCandidateProducerConfig(
        root=str(Path("../my-optimizer").resolve()),
        command=["sh", "{producer}/improve.sh", "{workspace}"],
    ))
    session = await create_optimization_session(
        workspace=workspace,
        session_dir="~/.vero/sessions/remote-run",
        backend_id="command",
        backend=backend,
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="latency_ms"),
            direction="minimize",
        ),
        producers={"default": producer},
    )
    result = await session.run()
finally:
    await sandbox.close()
```

Session manifests, databases, budgets, W&B logging, and durable artifacts stay
on the host. Git worktrees, candidate commands, compilation, and evaluation
commands run in the sandbox. Requests and reports use a temporary staging area;
VeRO transfers them explicitly and removes it after each operation.

Remote command harnesses and producer directories must be self-contained. An
executable named in a command must either be installed in the sandbox or live
under `{harness}` or `{producer}`. `ClaudeCodeAgent` requires a host-visible
workspace because its SDK takes a local `cwd`; `VeroAgent` and custom agents
whose tools operate through `Sandbox` can work without one. Incompatible agents
are rejected when the session is created.

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
    harbor_requirement="harbor[modal]==0.18.0",
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
the nested Harbor process. The sidecar isolates the optimizer process and keeps
budget/finalization state trusted, but it does not by itself contain malicious
target code loaded by the nested runner. When targets are adversarial, execute
them in a separate sandbox that cannot read verifier data or credentials.

For optimization-as-a-Harbor-task, `EvaluationSidecar` exposes the same engine
across a process boundary. `EvaluationAccessPolicy` maps each backend and
evaluation-set partition to canonical full, aggregate, or acknowledgement-only
disclosure; the canonical budget ledger meters agent calls. The sidecar can
host several backends at once. `GitCandidateTransport` imports agent commits
under durable trusted refs, and `CanonicalVerifier` selects and re-scores a
candidate before producing Harbor rewards. Hidden final evaluations use the
same backend contracts with unmetered admin authorization, so this deployment
does not introduce a parallel evaluation model.

Install the optional server dependencies with `scale-vero[harbor]`. A sidecar
image provides a trusted `module:factory` callable that accepts its JSON config
and returns `SidecarComponents`; start it with:

```bash
vero harbor serve \
  --factory trusted_deployment:build_components \
  --config /etc/vero/sidecar.json \
  --admin-token /shared/admin-token
```

The optimizer uses `vero harbor eval`, `status`, and `submit` through
`VERO_EVAL_URL`. Harbor's trusted verifier uses `vero harbor finalize` with the
root-readable token file and writes only the final reward mapping to
`reward.json`.

The built-in Harbor compiler supplies that factory and container topology for
nested Harbor evaluations. A minimal build file looks like:

```yaml
name: example/optimize-agent
agent_repo: ../my-program
task_source: example/terminal-benchmark@1.0
agent_import_path: my_program.agent:Agent
harbor_requirement: harbor[modal]==0.18.0

partitions:
  validation: [example/task-a, example/task-b, example/task-c,
               example/task-d, example/task-e]
  test: [example/task-hidden]

agent_access:
  - partition: validation
    disclosure: aggregate
    total_runs: 10
    total_cases: 50
    max_cases_per_run: 5

selection_partition: validation
targets:
  - partition: test
    reward_key: reward
```

Compile it with `vero harbor build --config build.yaml --output task`, then run
the generated task with Harbor. The test partition and task source exist only in
the sidecar image; the optimizer container receives the editable baseline, the
agent-facing CLI, and approved result projections. Exact Harbor and registry
task-source versions are required so the measurement substrate is reproducible.

Evaluation budgets meter attempts, not only successful reports: a failed,
timed-out, or cancelled backend run remains charged. Aggregate disclosure is a
feedback-control mechanism, not a privacy guarantee; allowing arbitrary,
overlapping subsets can reveal individual scores through differencing, so use
canonical selections and finite budgets when validation data is sensitive.
The generated shared-container topology protects the admin credential with
Unix ownership and permissions. It assumes candidate code cannot gain root in
that container; higher-assurance deployments should keep finalization
credentials outside the candidate workbench entirely.

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
from them. Reusing a session directory resumes its compatible baseline,
candidate lineage, evaluation history, completed rounds, and supported coding
agent state. VeRO rejects a resume if its backend configuration, evaluation set,
parameters, limits, seed, objective, or baseline is incompatible with the
manifest.

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
