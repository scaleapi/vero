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

[backend]
id = "command"
kind = "command"
harness_root = "../my-evaluator"
command = ["python3", "evaluate.py", "{workspace}", "{request}", "{report}"]

[backend.staged_inputs]
train_cases = "../my-evaluator/train.jsonl"
validation_cases = "../my-evaluator/validation.jsonl"
test_cases = "../my-evaluator/test.jsonl"

[backend.agent_context_inputs]
train = ["train_cases"]

[[evaluations]]
name = "train"
partition = "train"
agent_can_evaluate = true
agent_visible = true
agent_selection = "arbitrary"
disclosure = "full"
expose_case_resources = true

[[evaluations]]
name = "validation"
partition = "validation"
agent_can_evaluate = true
agent_visible = true
agent_selection = "arbitrary"
disclosure = "aggregate"

[evaluations.agent_budget]
total_runs = 50
max_cases_per_run = 100

[[evaluations]]
name = "test"
partition = "test"
agent_can_evaluate = false
agent_visible = false
agent_selection = "fixed"
disclosure = "none"

[protocol]
selection_evaluation = "validation"
final_evaluation = "test"
max_proposals = 5
error_rate_threshold = 0.1

[protocol.retry]
max_attempts = 3
initial_delay_seconds = 4
maximum_delay_seconds = 120

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

[session]
directory = "../runs/my-program"
```

Run `vero evaluate` to measure only the baseline or `vero run` to produce and
evaluate candidates. Paths are resolved relative to the config file. A target
must be a clean Git repository, while the session directory, evaluation harness,
and command producer must live outside it.

Retries wrap each individual case's inference or scoring call. By default VeRO
retries provider rate limits, HTTP 429/503/529 responses, and timeouts up to
three attempts with bounded exponential backoff. A successful retry remains a
successful case, while its earlier failed attempts are retained in the case's
structured error history. Set `max_attempts = 1` to disable retries.

An otherwise successful evaluation becomes failed when 10% or more of its
selected cases end in error; configure `protocol.error_rate_threshold` to
change that boundary. For objectives aggregated from case metrics, set
`aggregation = "mean"` (or another case aggregation) and
`case_failure_value` to assign a direction-appropriate penalty to errored,
skipped, or metric-less cases. This prevents a candidate from improving its
score by failing difficult cases.

The protocol ranks every candidate on the fixed base selection of
`validation`, regardless of which cheaper subsets the agent explored. The
agent sees only aggregate validation feedback. `test` is evaluated by the
trusted runtime after selection and never enters agent context. `train` cases
are explicitly mounted read-only because that evaluation opts into case
resources. Run `vero init` for this starter profile and `vero check` before an
expensive run.

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
  --max-proposals 5
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
`{producer}` and `{context}`; `VERO_CONTEXT_PATH` contains the same context path.
The harness and producer roots resolve to staged sandbox paths when the target
is not host-visible.

`staged_inputs` are trusted evaluator inputs available to the evaluation
command through `{input:NAME}` placeholders. They remain hidden from candidate
producers unless their names are also explicitly listed in
`agent_context_inputs` for a specific named evaluation, as in the example
above. This per-evaluation allowlist prevents exposing a test input merely
because train and test share one command backend.

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
    EvaluationPlan,
    EvaluationSet,
    MetricSelector,
    ObjectiveSpec,
)
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
)
from vero.runtime import create_local_optimization_session
from vero.sandbox import LocalSandbox

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
    evaluation_plan=EvaluationPlan.single(EvaluationSet(name="performance")),
    producers={"default": producer},
    max_proposals=5,
)
result = await session.run()
print(result.best.request.candidate.version, result.best.objective.value)

# Every candidate remains available after its producer workspace is gone.
inspection_sandbox = await LocalSandbox.create()
for candidate in session.candidate_repository.list():
    async with session.candidate_repository.checkout(
        candidate,
        sandbox=inspection_sandbox,
        name=f"inspect-{candidate.id}",
    ) as candidate_workspace:
        print(candidate.id, candidate_workspace.project_path)
```

`vero session inspect SESSION_DIR` includes the same durable candidate records
alongside the manifest and evaluation summaries.

### Run the target in a remote sandbox

The local factory above is a convenience wrapper. For containers, remote VMs,
or another execution environment, provision a `Workspace` in that sandbox and
pass it to the generic factory:

```python
from vero.runtime import create_optimization_session
from vero.candidate_repository import GitCandidateRepository
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
    session_dir = Path("~/.vero/sessions/remote-run").expanduser()
    candidate_repository = await GitCandidateRepository.create(
        session_dir / "candidates",
        workspace=workspace,
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
        candidate_repository=candidate_repository,
        session_dir=session_dir,
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

Session manifests, databases, budgets, W&B logging, artifacts, and the bare Git
candidate repository stay on the host. Candidate commands, compilation, and
evaluation run in isolated checkouts inside the sandbox. VeRO transfers Git
bundles between remote checkouts and the durable repository, then removes each
temporary checkout after use.

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
from vero.evaluation import (
    PythonTaskBackend,
    PythonTaskBackendConfig,
    PythonTaskEvaluationConfig,
)

backend = PythonTaskBackend(PythonTaskBackendConfig(
    harness_root=str(Path("../evaluation-state").resolve()),
    module="benchmark",
    task="quality",
    evaluations=[
        PythonTaskEvaluationConfig(
            name="train",
            partition="train",
            cases_path=str(Path("../train.jsonl").resolve()),
        ),
        PythonTaskEvaluationConfig(
            name="validation",
            partition="validation",
            cases_path=str(Path("../validation.jsonl").resolve()),
        ),
    ],
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
across a process boundary. `SidecarEvaluationPolicy` maps each backend and
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
environment_name: modal
secrets: [MODAL_TOKEN_ID, MODAL_TOKEN_SECRET]

partitions:
  validation: [example/task-a, example/task-b, example/task-c,
               example/task-d, example/task-e]
  test: [example/task-hidden]

agent_access:
  - partition: validation
    disclosure: aggregate
    expose_case_resources: false
    total_runs: 10
    total_cases: 50
    max_cases_per_run: 5

selection_partition: validation
targets:
  - partition: test
    reward_key: reward
```

Compile it with `vero harbor build --config build.yaml --output task`. The
`environment_name` selects Modal for each nested evaluation; the listed secrets
are forwarded as environment references, never embedded in the compiled task.
Run the outer optimization on Modal as well:

```bash
harbor run --path task --env modal --agent codex --model openai/gpt-5
```

Configure the outer agent's model credentials through Harbor's agent environment.
The test partition and task source exist only in the sidecar image; the optimizer
container receives the editable baseline, the agent-facing CLI, and approved
result projections. Exact Harbor and registry task-source versions are required
so the measurement substrate is reproducible.

Agent-triggered and system-triggered evaluations use independent budgets.
Reservations for cancelled runs or execution failures are durably refunded;
completed failure reports remain measurements and stay charged. Harbor retries
whole-run infrastructure failures and surfaces exhausted outages separately so
they fail the session instead of becoming candidate regressions. Aggregate
validation is optimization data, not a privacy guarantee: arbitrary subsets are
allowed by default, while the separate final evaluation remains unreachable.
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
| `EvaluationPlan` | Named evaluations, agent access, independent budgets, canonical selection, and optional hidden final evaluation |
| `EvaluationRecord` | The durable request, report, provenance, and objective result |
| `EvaluationBackend` | Measures a candidate without assuming its language or framework |
| `CandidateProducer` | Edits one isolated workspace to realize a proposed idea |
| `OptimizationStrategy` | Chooses parents, ideas, and producers for the next batch |
| `SelectionPolicy` | Chooses the best feasible evaluation for the configured objective |
| `OptimizationSession` | Owns lifecycle, events, artifacts, budgets, and durable state |

Coding agents receive a scoped `AgentContext`. They can edit only their supplied
workspace and call `evaluate(evaluation=..., selection=..., candidate_id=...)`.
The current workspace is saved as a candidate when `candidate_id` is omitted;
supplying an existing candidate ID re-evaluates that durable version. The tool
returns a compact receipt with the evaluation ID, status, approved summary, and
path to the filesystem result. Large case records, traces, and artifacts stay
out of the tool response. Intermediate checkpoints are real candidates and
remain eligible for selection, even if the agent later makes the program worse.

Each producer workspace contains a generated, read-only `.vero/` directory:

```text
.vero/
├── README.md
├── manifest.json
├── evaluations.json # available evaluations, selections, disclosure, budgets
├── cases/          # only backend-approved case resources
├── candidates/     # metadata, parent patches, and repository-native refs
└── evaluations/    # authorized summaries or full case/trace/artifact trees
```

The agent can inspect this with ordinary filesystem and Git commands. Full
evaluation disclosure splits potentially long traces into separate files;
aggregate disclosure includes only aggregate metrics and counts; none includes
only status. Candidate history includes durable Git refs, so siblings do not
need to be ancestors of the current checkout. A proposal sees the candidates
from the start of its generation, then immediately sees its own evaluated
checkpoints. Parallel siblings become visible in the next generation.

The directory is excluded from Git, protected by workspace read rules and file
permissions, and rejected if a candidate force-adds it anyway. Its contents are
a disposable view: durable candidate and evaluation stores remain the trusted
source. In Harbor, the sidecar owns the writable volume and the coding-agent
container mounts the same directory read-only. Case resources are exported only
when the evaluation authorization explicitly permits them and the backend
provides a safe export.

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
agent state. VeRO rejects a resume if its backend configuration, evaluation
plan, run protocol, parameters, limits, seed, objective, or baseline is
incompatible with the schema-v3 manifest.

```bash
vero session list
vero session inspect ~/.vero/sessions/<session-id>
vero session fork OLD_SESSION NEW_SESSION --max-proposals 20 --reset-budgets
vero session export ~/.vero/sessions/<session-id>
vero session clear ~/.vero/sessions/<session-id> --yes
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
- Agent-visible cases, histories, and evaluation details are projected into a
  generated `.vero/` view according to the same authorization boundary.
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
