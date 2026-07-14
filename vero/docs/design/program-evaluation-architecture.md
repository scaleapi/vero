# Program-neutral evaluation architecture

Status: accepted design and execution plan

Date: 2026-07-14

Implementation PR: new Phase A PR (`program-optimization-foundation`, targeting `main`)

Implementation branch: `program-optimization-foundation`

Detailed Phase A proposal: `docs/design/phase-a-evaluation-primitives-plan.md`

## Executive decision

VeRO should support two independent extensions:

1. **Program optimization** broadens what VeRO can optimize.
2. **Population/evolutionary optimization** broadens how VeRO searches.

The Harbor integration and program optimization overlap directly in the evaluation
foundation. Evolutionary optimization mostly sits above that foundation and should be
implemented later.

The immediate work should land the generic program-optimization foundation in a fresh
Phase A PR based on `main`. Harbor PR #3 should then be reconstructed on top of that
foundation, retaining its Harbor-driven access, staging, scrubbing, and durability work
without retaining its dataset-bound `EvalStrategy` architecture.

Intended stack:

```text
main
  -> program-optimization-foundation     # new Phase A PR
  -> harbor-1-core                     # reconstructed Harbor PR #3
  -> harbor-2-sidecar                  # PR #4, adapted to reconstructed PR #3
  -> harbor-3-compiler                 # PR #5, adapted to the contract
  -> harbor-4-docs                     # PR #6
```

The core rule is:

> VeRO core evaluates an opaque versioned workspace. It must not require the target
> program to be Python, an agent, a `uv` project, or dataset-driven.

## Product model

The generic optimization loop is:

```text
optimizer
    -> edits a versioned workspace
    -> produces a candidate version
    -> requests an evaluation
    -> receives structured measurements
    -> compares candidates using an objective
```

In compact form:

```text
versioned workspace + evaluation request
    -> evaluation backend
    -> evaluation report
    -> objective and constraints
    -> candidate comparison
```

The target is just a program represented by `Workspace`. There should be no
`PythonTarget`, `RustTarget`, or `AgentTarget` hierarchy. Languages and build systems
belong behind an executable evaluation boundary.

## Why Phase A should precede PR #3

Harbor PR #3 already introduces the same seam that program optimization needs:

- `EvaluationEngine`
- `EvalRequest`
- `EvalStrategy`
- `BudgetLedger`
- a reorganized `Evaluator`
- staged and resumable task evaluation

However, the new core interface still assumes that every evaluation has:

- a Hugging Face dataset,
- a dataset split,
- integer sample IDs,
- a Python `VeroTask`,
- per-sample `SampleResult` files, and
- a scalar score aggregated with a mean.

The current `EvalStrategy` protocol also returns `None` and requires the implementation
to persist `SampleResult` files through VeRO-specific storage helpers. This makes it a
storage extension point rather than a general evaluation boundary.

If that interface becomes the stable foundation for the Harbor stack, adding command
evaluation will require refactoring the same core seam immediately afterward. But
folding the entire program-optimization migration into PR #3 would mix two large review
stories and make the generic architecture appear Harbor-specific. Phase A should land
first. PR #3 should then be rebuilt on that foundation with its useful split visibility,
budget hardening, staged evaluation, and label-scrubbing work. Harbor-specific security
and packaging remain in PRs #4 and #5.

## Boundary and ownership

### `Workspace`

`Workspace` remains the target-program abstraction. It owns:

- the project root,
- version creation and restoration,
- isolated copies/worktrees,
- filesystem permissions, and
- access to a `Sandbox` for execution.

It should remain unaware of datasets, metrics, objectives, and optimization strategy.

### `EvaluationEngine`

The engine owns the shared lifecycle:

- resolve a caller-selected ID from its registry of approved backends,
- resolve the candidate version,
- authorize the requested evaluation set,
- reserve evaluation budget,
- create a unique run/result directory,
- evaluate from a clean isolated workspace,
- persist normalized results and provenance,
- invoke callbacks, and
- expose a stable result to local tools or remote frontends.

It must not know how a candidate is built or measured.

### `EvaluationBackend`

The backend owns target-specific execution:

- build or prepare the candidate when required,
- invoke the evaluator or benchmark,
- interpret ecosystem-specific output,
- retain backend-specific resumable artifacts, and
- return a normalized `EvaluationReport`.

Backends should not own Git history, global budget accounting, database insertion,
authorization, or candidate selection.

### Frontends

Frontends expose the engine to a caller and apply presentation or transport policy.

Examples:

- the in-process optimizer tool,
- the Harbor HTTP sidecar,
- the config-driven CLI or a future remote evaluation service.

The Harbor sidecar additionally owns its trust boundary: commit transfer,
authentication, protected volumes, result redaction, and final verification.

### Optimization strategy

Optimization strategy sits above all of the above. The current single-agent `Policy`
and a future evolutionary strategy consume the same candidate/evaluation APIs. Search
strategy is deliberately not part of Phase A.

## Canonical primitives

The following names and responsibilities are canonical for Phase A.

### Candidate

```python
class Candidate(BaseModel):
    commit: str
    repo_name: str
    parent_commit: str | None = None
    created_at: datetime
    message: str | None = None
```

Reuse the existing `Candidate`; a second reference type would duplicate its identity
fields and create conversion overhead. Rich population lineage (`parents`,
`generation`, `proposal_id`) is deferred until the search-strategy work and should be
represented by a separate candidate record if needed.

### Evaluation set and case selection

```python
class AllCases(BaseModel):
    kind: Literal["all"] = "all"


class CaseIds(BaseModel):
    kind: Literal["ids"] = "ids"
    ids: list[str]


class CaseRange(BaseModel):
    kind: Literal["range"] = "range"
    stop: int
    start: int = 0


CaseSelection = Annotated[
    AllCases | CaseIds | CaseRange,
    Field(discriminator="kind"),
]


class EvaluationSet(BaseModel):
    name: str = "default"
    partition: str | None = None
    selection: CaseSelection = Field(default_factory=AllCases)
```

`EvaluationSet` identifies the cases a backend should evaluate without describing how
those cases are stored or executed. Its name is meaningful within a backend, and the
backend resolves it. Examples:

- `EvaluationSet(name="performance")`
- `EvaluationSet(name="matmul", selection=CaseIds(ids=["512", "1024", "2048"]))`
- `EvaluationSet(name="gaia", partition="validation")`
- `EvaluationSet(name="gaia", partition="train", selection=CaseRange(stop=10))`

The generic model does not encode datasets. The VeroTask compatibility adapter maps
`dataset_id` to `name`, `split` to `partition`, `sample_ids` to `CaseIds`, and
`num_samples` to `CaseRange(stop=num_samples)`. Dataset-specific validation stays in
`VeroTaskBackend`.

`CaseRange` uses stop-exclusive positions in a backend-defined deterministic ordering;
`start` defaults to zero. It does not treat numeric-looking case IDs as positions.

### Evaluation request

```python
class EvaluationRequest(BaseModel):
    candidate: Candidate
    evaluation_set: EvaluationSet
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)
    seed: int | None = None
```

Authorization, budget waivers, and administrator privileges are not caller-controlled
fields in this request. They come from trusted engine context.

An individual evaluation uses exactly one backend, but an engine may register multiple
approved backends. The caller selects a registered backend ID through the engine API;
it cannot provide arbitrary backend configuration in the request.

### Backend context

```python
@dataclass
class EvaluationContext:
    workspace: Workspace       # clean and checked out at candidate.commit
    result_dir: Path            # unique to this evaluation
    session_id: str
    evaluation_id: str
```

The backend may write logs and resumable artifacts below `result_dir`. It must not be
required to use VeRO's legacy per-sample cache helpers.

### Backend protocol

```python
@runtime_checkable
class EvaluationBackend(Protocol):
    async def evaluate(
        self,
        *,
        context: EvaluationContext,
        request: EvaluationRequest,
    ) -> EvaluationReport:
        ...
```

Use `EvaluationBackend`, not `EvalStrategy`. The word `strategy` is reserved for the
outer optimization/search policy.

Initial implementations:

```text
VeroTaskBackend   current Python/uv task pipeline
HarborBackend     nested `harbor run` and result collation
CommandBackend    arbitrary executable producing a versioned JSON report
```

The current staged inference/scoring work belongs inside `VeroTaskBackend`; it is a
valuable backend capability but not a requirement of every evaluator.

### Evaluation artifacts, case result, and report

```python
class EvaluationArtifact(BaseModel):
    path: str
    media_type: str | None = None
    description: str | None = None


class CaseError(BaseModel):
    message: str
    code: str | None = None
    phase: str | None = None
    attempt: int | None = None
    retryable: bool | None = None
    terminal: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationDiagnostic(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"]
    phase: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CaseResult(BaseModel):
    case_id: str
    status: Literal["success", "error", "skipped"]
    metrics: dict[str, float] = Field(default_factory=dict)
    output: JsonValue | None = None
    feedback: str | None = None
    errors: list[CaseError] = Field(default_factory=list)
    artifacts: list[EvaluationArtifact] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    schema_version: Literal["1"] = "1"
    status: Literal["success", "failed"]
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)
    artifacts: list[EvaluationArtifact] = Field(default_factory=list)
    diagnostics: list[EvaluationDiagnostic] = Field(default_factory=list)
```

An `EvaluationArtifact` describes a file produced by an evaluation, such as a log,
trace, profile, plot, transcript, or compiler diagnostic. The file itself remains in
the evaluation's artifact directory; the model stores its relative path and descriptive
metadata.

Case errors are structured and chronological. A successful case may retain
non-terminal errors from failed retries. An error result contains at least one terminal
error, and each entry may identify its phase, attempt, stable code, and retryability.

Evaluation diagnostics are structured, evaluation-wide operational explanations that
do not belong to a particular case. They are ordered, private by default, and are not
objective metrics. Report status remains authoritative when a diagnostic has error
severity.

Cases are optional. This supports both a single whole-program benchmark and a
dataset/task suite.

The report should preserve measurement qualifications needed by Harbor, such as:

- number of cases requested, scored, and errored,
- standard error or other uncertainty estimates,
- attempt counts and dead-attempt causes,
- whether an aggregate covers the full evaluation set,
- evaluator/backend identity and version.

Sensitive case details and safe aggregate metrics should be separable so a frontend
can apply disclosure policy without reconstructing measurements from legacy files.

### Objective

```python
class ObjectiveSpec(BaseModel):
    metric: str
    direction: Literal["maximize", "minimize"]
    aggregation: Literal["mean", "median", "min", "max"] = "mean"
    constraints: list[MetricConstraint] = Field(default_factory=list)
```

Example:

```text
minimize latency_ms
subject to correct == 1
and max_error <= 1e-5
```

The evaluator reports measurements; the objective decides what is better. One shared
objective must drive:

- `Policy.get_best_version()`,
- Harbor `auto_best`,
- final verification,
- dashboard ranking, and
- future evolutionary fitness.

Multi-objective/Pareto selection is explicitly deferred.

### Evaluation record

The canonical persisted entity in Phase A is an evaluation record containing:

```text
evaluation ID
candidate
request and evaluation set
backend identity/config digest
report
objective result
timestamps and provenance
```

Phase A writes schema-v2 evaluation records and uses them for selection. Preserve
`Experiment`, `ExperimentRun`, `SampleResult`, and existing dataframes through legacy
readers and deprecated adapters rather than keeping them in canonical logic.

## Compatibility adapters

### Dataset request adapter

The existing public request surface remains available during migration:

```python
experiment = await policy.evaluate_version(
    commit="abc123",
    dataset_id="benchmark",
    split="validation",
    sample_ids=[0, 1, 2],
)
```

It validates dataset/sample selection, then creates the flat `EvaluationSet` and
canonical request. Existing `ExperimentRunnerTool.evaluate_commit()` remains a wrapper.
No standalone `DatasetEvaluationRequest` class exists on `main`, so Phase A does not
introduce one solely as a deprecated alias.

### VeroTask result adapter

Mapping into the generic report:

```text
DatasetSample.sample_id -> CaseResult.case_id
TaskResult.score        -> case.metrics["score"]
TaskResult.metrics      -> case.metrics
TaskResult.error        -> CaseError(phase="execution")
TaskResult.eval_error   -> CaseError(phase="scoring")
feedback/traces         -> corresponding case fields or artifacts
ExperimentResult.score -> ObjectiveSpec(metric="score", aggregation="report")
```

Legacy `SampleResult` files may continue to be emitted for compatibility, but they
should be produced by the engine/persistence adapter rather than required by the
backend protocol.

### API compatibility

- Preserve `Policy(dataset=..., task=...)` by registering a default `VeroTaskBackend`.
- Make `dataset` optional when an explicit backend registry is supplied.
- Introduce `optimizer=` in the canonical Policy path and keep `agent=` as a
  compatibility alias.
- Preserve the old `vero evaluate` path while adding config-driven command evaluation.

## Harbor integration

### New Phase A PR: generic program optimization

The new foundation PR supersedes these concepts currently introduced by PR #3:

| Current | Intended |
| --- | --- |
| `EvalStrategy` | `EvaluationBackend` |
| `produce_sample_results(...) -> None` | `evaluate(...) -> EvaluationReport` |
| default strategy inline in `Evaluator` | explicit `VeroTaskBackend` |
| `EvalRequest` as wire and core type | generic request plus Harbor/dataset wire adapter |
| backend writes `SampleResult` files | engine persists normalized report |
| engine resolves all samples | dataset adapter resolves dataset cases |

Phase A implements the generic ledger, engine, backend, persistence, command-evaluation,
and optimization contracts from `main`. It may reuse implementation ideas from PR #3,
but it does not take a code dependency on that branch.

### Reconstructed PR #3: Harbor core

Rebuild `harbor-1-core` on the Phase A branch rather than mechanically rebasing its
combined evaluator commit. Port only the behavior that remains additive:

- three-tier split visibility and its compatibility mapping to evaluation sets,
- fail-closed access policy,
- Harbor-oriented budget reconciliation and ledger durability hardening,
- staged and resumable VeroTask inference/scoring, and
- label scrubbing before inference.

Do not port `EvalStrategy`, the dataset-bound engine request, the alternative evaluator
layout, or backend-owned `SampleResult` persistence. Those responsibilities are
replaced by Phase A.

### PR #4: sidecar

The Harbor sidecar remains a privileged frontend over the engine. It owns:

- commit transfer from the untrusted optimizer workspace,
- token authentication,
- protected evaluator/scorer state,
- evaluation-set authorization,
- disclosure/redaction,
- agent-visible and admin-visible result routing,
- submission, and
- final verification.

Its response should expose a generic metrics map and qualified case summary. Keep
`mean_score`, `dataset_id`, and `split` as compatibility fields where required, not as
the canonical core schema.

The later Harbor access-hardening work demonstrates that budget bypass and evaluation-set
authorization are separate capabilities. Replace a broad `admin` switch over time with
trusted execution policy such as:

```python
class EvaluationAuthorization(BaseModel):
    may_evaluate: bool
    meter_budget: bool
    disclosure: Literal["full", "aggregate", "none"]
```

The caller must not be allowed to request these capabilities itself.

### PR #5: nested Harbor and compiler

`HarborRunner` becomes `HarborBackend` and returns an `EvaluationReport`:

```text
Harbor task       -> case
Harbor reward     -> metric
Harbor transcript -> feedback/artifact
Harbor exception  -> `CaseError`/diagnostic
```

The compiler remains Harbor-specific. Its internal configuration should use a
discriminated backend union rather than allowing Mode A/Mode B fields to silently
coexist:

```text
VeroTaskEvaluationConfig
HarborEvaluationConfig
CommandEvaluationConfig (Phase A)
```

Harbor may retain Mode A and Mode B terminology in its public compatibility layer.
Prefer `target_repo`; accept `agent_repo` as a deprecated alias.

### PR #6: docs and example

Use `target program` in generic architecture documentation and reserve `target agent`
for examples such as GAIA where the target truly is an agent.

### Harbor follow-up work that should feed the generic model

- Feedback transcripts and attempt detail belong in case feedback/errors/diagnostics.
- Qualified means, error counts, standard errors, and versioned re-evaluations belong
  in the generic report and evaluation record.
- Durable fail-closed budget restoration belongs in the shared budget implementation.
- Per-target model/executor overrides belong in request parameters.
- Hidden evaluation-set subset rules belong in authorization/disclosure policy.
- Infrastructure failure classification must remain diagnostic and must not silently
  alter the booked objective value.

## Budget and access model

The current PR #3 `BudgetLedger` is a strong implementation reference but is keyed by
`(split, dataset_id)` and charges runs plus samples. The Phase A model supports:

```python
class EvaluationCost(BaseModel):
    runs: int = 1
    cases: int = 0
    wall_time_seconds: float = 0
```

```text
Budget key: backend identity plus evaluation set, or a policy-defined group
Limits: runs, cases, wall time
Reservation: atomic and durable
Failure policy: explicitly configured and auditable
```

Dataset-specific `SplitBudget` remains an adapter. Token and monetary optimizer budgets
can remain separate until population search requires unified accounting.

There must be one budget authority. Local tools must not bypass the engine ledger and
then mutate a second copy of the budget. This is required before parallel evaluation is
considered safe.

Access and budget are independent:

- whether an evaluation set may be used,
- whether the call consumes budget,
- what result detail may be disclosed.

Do not encode all three decisions in one `admin` boolean.

## Concurrency requirements

Program optimization does not require population search, but the evaluation contract
should be safe for concurrent candidate evaluation:

- every run has a unique ID and result directory,
- backend instances do not rely on process-global working directories,
- candidates are evaluated in isolated workspaces,
- budget reservations are atomic,
- database/result writes are transaction-safe or serialized,
- backend concurrency limits are explicit,
- hardware resources may use named semaphores/capacity limits,
- cache keys include candidate version, backend/config digest, evaluation set,
  parameters, and seed.

Add a contract test that evaluates two distinct candidate versions concurrently. A
backend may declare concurrency `1`; it must not corrupt or conflate results.

## Phase A command backend and product proof

The program-optimization release is not complete until a target with no Python package
and no VeRO dependency can be optimized.

Proposed configuration:

```toml
[target]
root = "./target"

[evaluation]
backend = "command"
harness_root = "./evaluator"
command = [
  "./evaluate",
  "--candidate", "{workspace}",
  "--request", "{request}",
  "--output", "{report}",
]
working_directory = "."
timeout_seconds = 600

[objective]
metric = "latency_ms"
direction = "minimize"

[[objective.constraints]]
metric = "correct"
operator = "=="
value = 1
```

Use argv arrays, not shell strings. The harness and active configuration live outside
the editable target workspace and should be mounted read-only. VeRO writes a versioned
request JSON and validates the returned report JSON.

Acceptance demonstration:

> A C matrix-multiplication repository, containing no Python package and no
> VeRO dependency, is optimized by minimizing latency subject to correctness, while all
> existing VeroTask examples remain unchanged.

## Evolutionary optimization: deferred layer

The evaluator is the fitness-function boundary needed by evolutionary search, but the
Harbor work does not provide the search layer.

After program optimization, split the current `Policy` into:

```text
OptimizationRuntime
    workspace and candidate management
    evaluator
    objective
    budget manager
    history/artifacts

OptimizationStrategy
    decides what candidates to produce and select
```

Preserve current behavior as `SingleAgentStrategy`. The first population-oriented
strategy should be `BestOfNStrategy`:

```text
planner emits N structured proposals
    -> N isolated worktrees
    -> N fresh implementation agents
    -> N candidate commits
    -> scheduled evaluations
    -> objective comparison
    -> best valid candidate
```

Only after Best-of-N should VeRO add population state, generations, parent selection,
elitism, and multi-generation evolution.

Harbor contributes useful infrastructure to this later work—commit-oriented remote
evaluation, durable budgets, protected metrics, and versioned results—but it does not
yet provide worker-agent isolation, candidate lineage, or population scheduling.

## Implementation sequence

### Phase A: generic program optimization (new PR from `main`)

1. Add canonical request, flat evaluation-set, case-selection, context, case, report,
   artifact, and objective models.
2. Add the backend registry, `EvaluationBackend`, and `VeroTaskBackend`.
3. Make the engine own report persistence, authorization, budgeting, and lifecycle.
4. Add schema-v2 records plus schema-v1 and dataset API adapters.
5. Implement `CommandBackend` and its versioned JSON file contract.
6. Make datasets and VeroTask optional in the canonical Policy/tool/CLI path.
7. Add configuration for target workspace, candidate producer, backend, evaluation set,
   and objective.
8. Add an end-to-end non-Python matrix-multiplication optimization example.
9. Update product documentation and positioning to describe program optimization.

Exit criteria:

- Existing VeroTask tests pass unchanged or through compatibility adapters.
- A command backend returns a report without importing VeRO or writing legacy sample
  files.
- The engine can evaluate a report with no cases.
- Objective direction supports both maximize and minimize.
- No module in the generic backend protocol imports Hugging Face datasets or `uv`.
- A C target containing no Python package, VeRO dependency, or agent framework
  is improved through the ordinary VeRO optimization loop.
- VeRO can accurately claim to be a generic program optimizer, with the documented
  qualification that an approved evaluation backend or command harness is configured.

### Phase B: reconstruct Harbor PR #3

1. Create a replacement `harbor-1-core` branch from the Phase A head.
2. Port split visibility and fail-closed access behavior.
3. Port staged/resumable VeroTask evaluation and label scrubbing into
   `VeroTaskBackend`.
4. Port Harbor-oriented budget reconciliation and durability hardening onto the generic
   ledger.
5. Exclude the current `EvalStrategy`, dataset-bound request, and evaluator refactor.
6. Update PR #3 to the reconstructed branch after its tests pass.

Exit criteria:

- PR #3 is a consumer of Phase A rather than an alternative evaluation foundation.
- All access, staging, scrubbing, budget, and VeroTask behavior retained from the
  original PR #3 has parity coverage.
- PR #3 contains no `EvalStrategy` or second canonical evaluation model.

### Phase C: adapt the downstream Harbor stack

1. Keep PR #4 based on reconstructed `harbor-1-core` and adapt the sidecar to canonical
   requests, reports, evaluation sets, and disclosure projections.
2. Rebase PR #5 after PR #4 is adapted.
3. Convert `HarborRunner` to `HarborBackend`.
4. Preserve current Harbor wire/config compatibility.
5. Fold qualified metrics and durable ledger fixes into shared primitives where
   appropriate.

Exit criteria:

- Mode A and Mode B retain current behavior.
- Hidden details never cross the sidecar disclosure boundary.
- Harbor can rank candidates through `ObjectiveSpec`.
- Existing Harbor task configs still load.

### Phase D: search strategies

1. Extract `OptimizationRuntime` and `OptimizationStrategy`.
2. Wrap current behavior as `SingleAgentStrategy`.
3. Add isolated worker factories and concurrency-safe scheduling.
4. Implement Best-of-N.
5. Add multi-generation evolutionary search later.

## Tests required at the contract boundary

Every backend should pass shared tests for:

- clean candidate checkout,
- success with aggregate-only metrics,
- optional case results,
- invalid report rejection,
- timeout handling,
- non-zero process/backend failure,
- stdout/stderr and artifact capture,
- unique repeated-evaluation records,
- concurrent evaluation isolation,
- deterministic objective comparison,
- constraint violations,
- sensitive-detail redaction by the Harbor frontend,
- resume behavior without conflating an earlier evaluation,
- budget reservation on success and failure according to policy.

Additional compatibility tests:

- existing `VeroTask` examples,
- current `ExperimentRunnerTool` calls,
- existing `Policy(dataset=..., task=...)`,
- Harbor Mode A,
- Harbor Mode B task-to-case/reward-to-metric mapping,
- Harbor verifier `submit` and `auto_best` behavior.

## Explicit non-goals for Phase A

- Rewriting existing schema-v1 session data in place.
- Removing `SampleResult` or dataset APIs.
- Harbor split tiers, VeroTask staging/resume, label scrubbing, ledger hardening,
  sidecar, verifier, compiler, and nested runner integration.
- Multi-objective/Pareto selection.
- Distributed scheduling.
- Containerizing every evaluator.
- Build caching beyond defining safe cache identity.
- Multiple implementation agents.
- Population or evolutionary search.
- Renaming every public `agent` symbol immediately.

## Phase A decisions

The implementation-level decisions are resolved in
`docs/design/phase-a-evaluation-primitives-plan.md`. In particular, Phase A uses one
flat `EvaluationSet` with discriminated case selection, migrates canonical persistence
to `EvaluationRecord`, preserves deprecated APIs and schema-v1 reads, treats full
reports as private, introduces a trusted authorization decision with independent
access/metering/disclosure fields, and places the objective domain model in core
evaluation while keeping search algorithms outside it. It ships from a fresh branch
based on `main` and includes `CommandBackend`, a dataset-free Policy/CLI path, and a
non-Python end-to-end optimization proof.

## Merge and branch mechanics

- Create `program-optimization-foundation` from `main` and open it as the new Phase A PR.
- Do not merge Phase A until the command backend, generic Policy/CLI path, and
  non-Python end-to-end proof satisfy the product claim.
- Reconstruct `harbor-1-core` from the Phase A head; do not mechanically preserve PR
  #3's combined evaluator commit.
- During review, retarget PR #3 to the Phase A branch. After Phase A merges, its base
  becomes `main` while its diff remains Harbor-specific.
- Keep PR #4 based on `harbor-1-core`; update it only after reconstructed PR #3 is
  stable.
- Rebase PR #5 and later Harbor branches in order after PR #4 is adapted.
- Keep compatibility adapters until the Harbor stack has shipped and downstream users
  have a migration path.

## Reference PRs

- Harbor core foundation: https://github.com/scaleapi/vero/pull/3
- Harbor evaluation sidecar: https://github.com/scaleapi/vero/pull/4
- Harbor nested runner and compiler: https://github.com/scaleapi/vero/pull/5
- Harbor docs and GAIA example: https://github.com/scaleapi/vero/pull/6
- Feedback and multi-fidelity follow-up: https://github.com/scaleapi/vero/pull/30
- Discriminated Mode A/Mode B configs: https://github.com/scaleapi/vero/pull/31
- Access-tier integrity: https://github.com/scaleapi/vero/pull/34
- Qualified measurements and versioned re-evaluation: https://github.com/scaleapi/vero/pull/35
- Durable fail-closed ledger recovery: https://github.com/scaleapi/vero/pull/36
- Transfer-target evaluator overrides: https://github.com/scaleapi/vero/pull/38
