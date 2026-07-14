# Phase A: generic program optimization

Status: implemented; ready for PR review

Date: 2026-07-14

Implementation branch: `program-optimization-foundation`

Pull request: new Phase A PR, targeting `main`

Implementation workspace: fresh branch from `origin/main`

## Summary

Phase A makes generic program optimization a shipped capability, not only an internal
architecture. It replaces dataset/sample evaluation as VeRO's canonical model, adds an
executable command boundary for arbitrary target programs, and routes the ordinary
optimization loop through it. Existing dataset-oriented APIs remain callable as
deprecated adapters, and existing session data remains readable.

The completed vertical slice is:

```text
versioned program workspace
    -> optimizer proposes and commits a candidate
    -> approved backend ID + Candidate + EvaluationSet + runtime limits
    -> registered EvaluationBackend
    -> EvaluationReport
    -> ObjectiveSpec
    -> EvaluationRecord
    -> best candidate selection
```

Phase A is a full canonical migration, not merely a set of unused types:

- the current Python/`uv` path becomes `VeroTaskBackend`,
- the evaluator and engine consume program-neutral requests and reports,
- database and result-directory writes use schema v2,
- selection and internal consumers use objective results rather than `mean_score`,
- legacy experiment types become compatibility views or backend implementation details,
- `CommandBackend` evaluates programs through a versioned JSON file contract,
- Policy, tools, and CLI no longer require a dataset or `VeroTask`, and
- a non-Python target is optimized end to end as the release proof.

Phase A ships in a fresh PR based on `main`. Harbor PR #3 is reconstructed on top after
the Phase A contracts stabilize. This makes program optimization an independent VeRO
capability and keeps Harbor-specific access, staging, and durability behavior in the
Harbor stack.

## Decisions already made

These decisions are part of the proposal and are not left to the implementer:

1. Reuse the existing `Candidate`; do not introduce `CandidateRef`.
2. Use one flat, backend-neutral `EvaluationSet` with a discriminated case-selection
   model; do not introduce named or dataset-specific evaluation-set subtypes.
3. Include single-objective comparison, direction, aggregation, failure value, and
   constraints in Phase A.
4. Migrate all in-repository consumers to the new canonical record.
5. Preserve old Python APIs through deprecated adapters.
6. Read schema v1 data, write only schema v2, and never rewrite old sessions implicitly.
7. Keep the existing `experiments/` directory name to avoid unnecessary filesystem
   churn, even though its new contents are evaluation records.
8. Treat a full `EvaluationReport` as trusted/private data. Frontends disclose a
   structural projection of it rather than redacting arbitrary dictionaries in place.
9. Include command evaluation, generic Policy/tool/CLI configuration, and a non-Python
   end-to-end optimization proof in Phase A.
10. Implement Phase A in a fresh PR from `main`; reconstruct Harbor PR #3 on top rather
    than preserving its current evaluator architecture.
11. Allow multiple approved backends per engine while keeping each individual
    evaluation bound to exactly one backend.
12. Treat “VeRO is a generic program optimizer” as a Phase A acceptance claim, qualified
    by the requirement that an approved backend or command harness be configured.

## Review focus

Reviewers should focus on these boundaries:

- whether the flat evaluation-set model is sufficient for current datasets and future
  programs,
- whether objective metric selection and failure ordering are unambiguous,
- whether backend ownership stops at producing a report,
- whether the backend registry supports multiple evaluators without allowing callers
  to inject backend configuration,
- whether v1 compatibility can be maintained without allowing old models back into
  core logic,
- whether full versus aggregate disclosure is structurally safe for Harbor,
- whether the command contract and non-Python example substantiate the product claim,
- whether the implementation slices keep the PR reviewable.

## Current state being replaced

On `main`, evaluation still encodes the old model:

- `Evaluator.evaluate()` constructs `DatasetSubset` and `ExperimentRun`.
- the Python/`uv` implementation is inline in `Evaluator` instead of being a backend.
- `ExperimentResult.score()` hardcodes mean aggregation and a minimum fill score.
- `Policy`, tools, traces, artifacts, and dashboards query `dataset_subset_split` and
  `mean_score` columns.
- evaluation and optimization require a dataset and Python `VeroTask`.

Current Harbor PR #3 contains useful split visibility, ledger, staged task evaluation,
and label-scrubbing work, but combines them with a dataset-bound `EvalStrategy` and
alternative evaluator layout. Phase A is implemented independently from `main`. PR #3
is then reconstructed to port only the additive Harbor behavior onto the final
contracts.

## Architecture and ownership

```text
Caller / frontend
    |
    | approved backend ID + EvaluationRequest + trusted EvaluationAuthorization
    v
EvaluationEngine
    | resolve registered backend
    | authorize backend and evaluation set
    | resolve and reserve budget
    | invoke canonical evaluator
    | insert record and fire callbacks
    v
Evaluator
    | create evaluation ID and directories
    | isolate and checkout candidate
    | invoke backend
    | validate report
    | compute objective
    | persist EvaluationRecord
    v
EvaluationBackend
    | build/run/measure target using checked-out workspace
    | checkpoint cases when useful
    v
EvaluationReport
```

### `Workspace`

`Workspace` remains the target-program abstraction. It owns versioning, copies,
checkout, filesystem access, and sandbox execution. It does not know about evaluation
sets, metrics, datasets, objectives, or budgets.

### `EvaluationEngine`

The engine is the policy boundary. It owns an approved backend registry, authorization,
budget reservation, database insertion, and callbacks. It does not discover datasets
itself and does not run `uv`. The caller may select a registered backend ID but cannot
provide arbitrary backend configuration. Evaluation-set-specific resolution is
provided by the resolved backend or compatibility adapter.

### `Evaluator`

The evaluator is the execution lifecycle. It owns isolation, result directories,
backend invocation, report validation, objective computation, persistence, and failure
recording. It does not interpret task modules, Harbor jobs, or command output.

### `EvaluationBackend`

A backend owns ecosystem-specific preparation and measurement. It receives an already
checked-out workspace and returns a report. It may checkpoint through the canonical
case store, but the protocol never requires legacy `SampleResult` files.

### Frontends

Frontends translate local tool calls or remote requests into canonical requests and
trusted authorization decisions. Harbor-specific authentication, commit transfer, and
redaction remain outside Phase A and outside the generic engine.

## Detailed domain model

The following declarations specify field names and behavior. Minor Pydantic syntax may
change during implementation, but serialized field names and semantics should not.

### Candidate

Use the existing model:

```python
class Candidate(BaseModel):
    commit: str
    repo_name: str
    parent_commit: str | None = None
    created_at: datetime
    message: str | None = None

    @property
    def id(self) -> tuple[str, str]:
        return (self.repo_name, self.commit)
```

Evaluation identity uses `repo_name + commit`. Timestamps and messages are persisted
metadata but do not change identity. Multi-parent evolutionary lineage is a future
candidate-record concern, not a reason to duplicate this model now.

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

An evaluation set identifies the cases to run without encoding how they are stored or
executed. The set name is resolved within the selected backend. For example, the same
model can identify the GAIA validation set for `VeroTaskBackend` or a large-matrix
performance set for `CommandBackend`.

```python
CaseRange(stop=10)             # positions 0 through 9
CaseRange(start=10, stop=20)   # positions 10 through 19
```

Validation:

- names, partitions, and case IDs must be non-empty,
- `CaseIds.ids` must be non-empty and unique,
- `CaseRange.start` must be non-negative and `stop` must be greater than `start`,
- a case range selects positions `[start, stop)` in the backend's stable case ordering,
- a backend that accepts `CaseRange` must define deterministic ordering,
- Phase A does not support a range step; use `CaseIds` for sparse selection,
- `AllCases` selects the complete set,
- list order is preserved because it is meaningful for deterministic reporting.

Stable budget keys:

```text
<backend-id>:<evaluation-set-name>:<partition-or-empty>
```

Case selection affects reservation cost but not budget identity. Parameters do not
participate in budget keys. Both selection and parameters participate in provenance
and evaluation fingerprints.

The deprecated dataset adapter performs this translation:

```text
dataset_id  -> EvaluationSet.name
split       -> EvaluationSet.partition
sample_ids  -> CaseIds(ids=[str(sample_id), ...])
num_samples -> CaseRange(stop=num_samples)
no selector -> AllCases()
```

It validates non-negative integer sample IDs before converting them to strings.

### Runtime limits and request

```python
class EvaluationLimits(BaseModel):
    timeout_seconds: int = 600
    case_timeout_seconds: int = 180
    max_concurrency: int = 100
    retry_config: RetryConfig = Field(default_factory=RetryConfig)


class EvaluationRequest(BaseModel):
    candidate: Candidate
    evaluation_set: EvaluationSet
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)
    seed: int | None = None
```

Validation:

- timeouts and concurrency must be positive,
- parameters must be JSON-serializable,
- request fingerprints use canonical JSON with sorted object keys,
- authorization, budget exemptions, result paths, backend instances, and backend
  configuration are never caller-controlled request fields.

Backend selection is an engine argument rather than a request field. A caller may
select only an ID already present in the engine's approved registry. The resolved
backend and configuration digest are persisted in `BackendProvenance`.

`parameters` is the backend-specific escape hatch. Existing `task_params` map directly
to it. Values that change evaluator behavior are recorded in the request and therefore
in provenance and cache identity.

### Evaluation artifacts

```python
class EvaluationArtifact(BaseModel):
    path: str
    media_type: str | None = None
    description: str | None = None
```

An `EvaluationArtifact` describes a file produced by an evaluation, such as a log,
trace, profile, plot, transcript, or compiler diagnostic. It contains metadata, not the
file contents. Paths are relative to the evaluation's `artifacts/` directory. Absolute
paths, `..`, empty segments, and symlink escapes are rejected when the record is
persisted or materialized.

### Case results

```python
class CaseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class CaseError(BaseModel):
    message: str
    code: str | None = None
    phase: str | None = None
    attempt: int | None = None
    retryable: bool | None = None
    terminal: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CaseResult(BaseModel):
    case_id: str
    status: CaseStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    input: JsonValue | None = None
    output: JsonValue | None = None
    feedback: str | None = None
    errors: list[CaseError] = Field(default_factory=list)
    execution_trace: list[JsonValue] | None = None
    evaluation_trace: list[JsonValue] | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: list[EvaluationArtifact] = Field(default_factory=list)
```

Validation and conventions:

- `case_id` is non-empty and unique within a report,
- metric keys are non-empty and values must be finite floats,
- errors are ordered chronologically,
- error messages are non-empty; optional codes and phases are non-empty when present,
- attempt numbers are positive when present,
- `terminal=True` means the error ended the overall case, not merely one retry attempt,
- `SUCCESS` may contain errors from failed attempts but none may be terminal,
- `ERROR` requires at least one terminal error and may contain multiple errors,
- `SKIPPED` must not contain a terminal error,
- `SKIPPED` does not contribute to case aggregation,
- inputs, outputs, feedback, errors, and traces are considered sensitive by default,
- backend-specific structured details go in `metadata`, not in invented top-level
  fields.

### Evaluation report

```python
class EvaluationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EvaluationDiagnostic(BaseModel):
    code: str
    message: str
    severity: DiagnosticSeverity
    phase: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    schema_version: Literal["1"] = "1"
    status: EvaluationStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)
    diagnostics: list[EvaluationDiagnostic] = Field(default_factory=list)
    artifacts: list[EvaluationArtifact] = Field(default_factory=list)
    error: str | None = None
```

Semantics:

- top-level metrics are aggregate measurements over the requested evaluation set,
- backends may omit aggregate metrics if the configured objective derives them from
  cases,
- top-level metrics must be safe for aggregate disclosure; case data and diagnostics
  are private unless a frontend grants full disclosure,
- diagnostics are chronological evaluation-wide operational explanations, not metrics
  or case errors,
- diagnostic codes, messages, and optional phases are non-empty,
- report status is authoritative; error-severity diagnostics explain failures or
  recovered problems but do not determine status by themselves,
- `FAILED` represents an evaluation that produced a valid failure report,
- a backend crash or invalid report is caught by the evaluator, converted into a
  persisted failure record, and then re-raised to preserve current caller behavior,
- an empty `cases` list is valid for whole-program aggregate evaluation,
- a successful report may contain errored cases; the objective and backend's aggregate
  metrics determine whether the candidate is useful.

Phase A does not add a third `partial` status. Qualification comes from explicit case
counts and diagnostics instead of ambiguous status semantics.

### Backend provenance

```python
class BackendProvenance(BaseModel):
    name: str
    version: str
    config_digest: str
```

Every backend exposes a serializable configuration model. `config_digest` is SHA-256 of
canonical JSON containing the backend name, backend version, and configuration. Secrets
must be represented by stable secret names or redacted fingerprints, never plaintext.

### Metric selectors and constraints

```python
class MetricAggregation(str, Enum):
    REPORT = "report"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"


class MetricSelector(BaseModel):
    metric: str
    aggregation: MetricAggregation = MetricAggregation.REPORT


class ConstraintOperator(str, Enum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="


class MetricConstraint(BaseModel):
    selector: MetricSelector
    operator: ConstraintOperator
    value: float


class ObjectiveSpec(BaseModel):
    selector: MetricSelector
    direction: Literal["maximize", "minimize"]
    failure_value: float | None = None
    constraints: list[MetricConstraint] = Field(default_factory=list)
```

Selector rules:

- `REPORT` reads exactly one top-level report metric,
- case reducers include only successful cases that contain the named metric,
- skipped cases never contribute,
- missing metrics do not silently become zero,
- reducers over an empty set are missing metrics,
- all selected and constraint values must be finite.

### Objective result and ordering

```python
class ConstraintViolation(BaseModel):
    constraint: MetricConstraint
    observed: float | None
    reason: str


class ObjectiveResult(BaseModel):
    value: float | None
    feasible: bool
    violations: list[ConstraintViolation] = Field(default_factory=list)
```

Evaluation rules:

1. Resolve all constraint selectors and record every violation.
2. Resolve the objective selector.
3. A failed report, missing objective metric, or any constraint violation is infeasible.
4. If the objective metric is missing or the report failed, use `failure_value` as the
   stored value when configured; otherwise store `None`.
5. `failure_value` does not make an evaluation feasible.
6. Feasible evaluations always outrank infeasible evaluations.
7. Feasible evaluations compare by direction.
8. Equal values break ties by candidate `created_at` descending, then commit hash
   ascending.
9. Automatic best-candidate selection returns no candidate when every non-baseline
   evaluation is infeasible. Existing baseline fallback remains a caller-level policy.

The VeroTask compatibility objective is:

```python
ObjectiveSpec(
    selector=MetricSelector(metric="score", aggregation="report"),
    direction="maximize",
    failure_value=0.0,
)
```

`VeroTaskBackend` computes the top-level score with errored legacy samples filled by
the failure value before the mean is calculated. Selecting that report metric preserves
current pessimistic score behavior without weakening the generic case-reducer rule that
only successful cases contribute.

### Evaluation record

```python
class EvaluationRecord(BaseModel):
    id: str
    request: EvaluationRequest
    report: EvaluationReport
    backend_id: str
    backend: BackendProvenance
    objective_spec: ObjectiveSpec | None = None
    objective: ObjectiveResult | None = None
    created_at: datetime
    completed_at: datetime
```

Rules:

- IDs are UUID strings generated by the evaluator before backend execution,
- `backend_id` is the approved registry key selected by the engine; it is distinct
  from the backend implementation name in `BackendProvenance`,
- repeated evaluation of the same request creates a new record,
- request fingerprints group comparable repeated measurements but are not record IDs,
- objective fields are both absent for measurement-only evaluation,
- optimization and Harbor selection require an objective,
- the objective result is persisted rather than recomputed implicitly under a later
  configuration,
- candidate, request, backend digest, and objective spec form the comparison provenance.

## Disclosure projection

The full report is trusted/private. Phase A defines a structural projection API for
frontends, even though the Harbor sidecar begins consuming it in Phase C.

```python
class DisclosureLevel(str, Enum):
    FULL = "full"
    AGGREGATE = "aggregate"
    NONE = "none"


class EvaluationSummary(BaseModel):
    evaluation_id: str
    candidate_commit: str
    backend_id: str
    evaluation_set: EvaluationSet
    status: EvaluationStatus
    metrics: dict[str, float]
    objective: ObjectiveResult | None
    total_cases: int
    successful_cases: int
    errored_cases: int
    skipped_cases: int
```

Projection rules:

- `FULL` returns the full record through trusted internal APIs,
- `AGGREGATE` returns `EvaluationSummary` only,
- `NONE` returns only an acknowledgement containing evaluation ID and status,
- aggregate projection never includes cases, inputs, outputs, feedback, traces,
  diagnostics, errors, or artifact paths,
- top-level metrics are the only backend-supplied values allowed into aggregate output.

Phase C may add statistical qualifiers such as standard error to top-level metrics or a
typed summary section; it must not weaken these disclosure boundaries.

## Authorization

Replace combined `admin` behavior with a trusted decision object:

```python
class EvaluationAuthorization(BaseModel):
    may_evaluate: bool
    meter_budget: bool = True
    disclosure: DisclosureLevel = DisclosureLevel.FULL
    reason: str | None = None
```

Rules:

- frontends or trusted policy resolvers create this object,
- request payloads cannot set it,
- denied backend/evaluation-set combinations fail before candidate checkout, budget
  debit, or backend work,
- `meter_budget=False` does only that; it does not bypass evaluation-set authorization,
- disclosure does not change what the trusted evaluator persists,
- local in-process evaluation defaults to allowed, metered, full disclosure,
- dataset split access maps to this object through the compatibility resolver,
- Harbor admin/finalize maps to allowed, unmetered, full disclosure,
- a future free baseline maps to allowed, unmetered, normal agent disclosure rather
  than administrator authority.

## Budget model

```python
class EvaluationCost(BaseModel):
    runs: int = 1
    cases: int | None = None


class EvaluationBudget(BaseModel):
    backend_id: str
    evaluation_set_key: str
    total_runs: int | None = None
    remaining_runs: int | None = None
    total_cases: int | None = None
    remaining_cases: int | None = None
    max_cases_per_run: int | None = None
```

`BudgetLedger.reserve((backend_id, evaluation_set_key), cost)` remains atomic and
durable. A reservation is never automatically refunded after backend work begins,
preserving current behavior.

Cost resolution:

- `CaseIds`: number of IDs,
- `CaseRange`: resolved positional count, including any valid truncation at the end of
  the available set,
- `AllCases`: total size resolved by the backend,
- an evaluation set whose size cannot be resolved uses `cases=None` and is charged one
  run with no case debit
  when no case limit applies,
- an unresolved size is rejected when its budget has a case limit because the engine
  cannot safely reserve an unknown cost.

`SplitBudget` remains a deprecated adapter to `EvaluationBudget`. There is one ledger
per session. `ExperimentRunnerTool` must stop maintaining and decrementing a second
copy.

Schema-v2 ledger persistence restores remaining values. Missing ledgers mean a fresh
session; malformed durable ledgers fail closed when the Harbor durability changes are
rebased.

## Backend protocol

The engine registry maps stable backend IDs to preconfigured `EvaluationBackend`
instances. One evaluation resolves exactly one backend. A session may use several
registered backends, and the same `EvaluationSet` shape may be interpreted differently
by each. Backend ID and configuration digest namespace authorization, budget keys, and
evaluation fingerprints.

```python
@dataclass
class EvaluationContext:
    workspace: Workspace
    session_id: str
    evaluation_id: str
    result_dir: Path
    artifact_dir: Path
    case_store: CaseCheckpointStore


@runtime_checkable
class EvaluationBackend(Protocol):
    @property
    def provenance(self) -> BackendProvenance: ...

    async def resolve_cost(
        self,
        evaluation_set: EvaluationSet,
    ) -> EvaluationCost: ...

    async def evaluate(
        self,
        *,
        context: EvaluationContext,
        request: EvaluationRequest,
    ) -> EvaluationReport: ...
```

Contract requirements:

- `context.workspace` is clean and checked out at `request.candidate.commit`,
- the backend validates and resolves its evaluation-set names, partitions, and
  selections,
- the backend does not change Git history or select another candidate,
- all returned values are JSON-serializable and pass model validation,
- filesystem artifacts stay inside the supplied directories,
- a backend may checkpoint canonical cases but must return a complete final report,
- the backend does not insert into the database, debit budget, compute the objective,
  invoke global callbacks, or apply frontend disclosure,
- backend objects must not rely on process-global current directories,
- backend-specific concurrency limits may be enforced internally in Phase A; a generic
  resource scheduler is deferred.

## Case checkpoint store

The engine-owned checkpoint store supports resumable and long-running backends without
making file persistence part of the backend protocol.

```python
class CaseCheckpointStore:
    async def save(self, result: CaseResult) -> None: ...
    async def load(self, case_id: str) -> CaseResult | None: ...
    async def load_all(self) -> list[CaseResult]: ...
```

Persistence rules:

- filenames use SHA-256 of UTF-8 `case_id`, not raw IDs,
- the actual case ID is stored and verified on load,
- saves use write-to-temp plus atomic replace,
- duplicate IDs within a completed report are rejected,
- checkpoint writes are serialized per evaluation,
- a corrupt checkpoint fails that evaluation rather than being silently skipped,
- checkpoint order does not define report order; the returned report does.

## VeroTask backend

`VeroTaskBackend` preserves the existing dataset/task path as one of the two Phase A
production backends.

Configuration:

```python
class VeroTaskBackendConfig(BaseModel):
    task: str
    task_project: str | None = None
    task_module: str | None = None
    hooks: list[str] = ["setup_logging"]
    subprocess_env_vars: list[str] = []
```

Behavior:

1. Interpret `EvaluationSet.name` as `dataset_id` and require `partition` as the dataset
   split.
2. Convert `CaseIds` to non-negative integer sample IDs, `CaseRange(start=0)` to the
   legacy prefix selection, and `AllCases` to the full split.
3. Resolve nonzero range starts by selecting the corresponding ordered dataset slice;
   reject a start beyond the split and truncate `stop` at the split size.
4. Construct legacy `DatasetSubset`, `ExperimentRun`, and `EvaluationParameters`
   privately for the current task subprocess.
5. Preserve `main` behavior for task discovery, required environment checks,
   `uv --with-editable`, dataset upload, retries, and timeouts.
6. Place legacy task parameters and staging files under `backend/vero-task/`, not at the
   canonical result root.
7. Load backend-private `SampleResult`s and convert them to `CaseResult`s.
8. Copy canonical cases through the checkpoint store.
9. Return a report containing case metric `score`, all existing custom metrics, error
   qualification, and compatible aggregate statistics.

Mapping:

```text
DatasetSample.sample_id -> str(CaseResult.case_id)
TaskResult.score        -> case.metrics["score"]
TaskResult.metrics      -> remaining case metrics
input/output            -> case input/output
feedback                -> case feedback
error                   -> CaseError(phase="execution")
eval_error              -> CaseError(phase="scoring")
execution/eval traces   -> corresponding trace fields
```

When both legacy error fields are present, preserve both in execution-then-scoring
order. Mark only errors that end the final case outcome as terminal; retry errors on an
eventually successful case remain non-terminal.

If a custom metric is also named `score`, the explicit task score wins and a warning is
recorded as an `EvaluationDiagnostic` in report diagnostics.

The backend derives top-level compatibility statistics but the default objective uses
case aggregation, so legacy error-fill behavior is explicit and auditable.

Staged inference/scoring, label scrubbing, and their resume behavior remain in Harbor
PR #3. When that PR is reconstructed, they are added inside `VeroTaskBackend` without
changing the generic backend contract.

## Command backend

`CommandBackend` is the Phase A proof that target programs need not be Python,
dataset-driven, or VeRO-aware.

Trusted configuration:

```python
class CommandBackendConfig(BaseModel):
    harness_root: str
    command: list[str]
    working_directory: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)
```

`harness_root` is a trusted path outside the editable target workspace and is mounted
read-only when the sandbox supports it. `working_directory` is relative to that root.
The command is an argv array, never a shell string. Configuration may use these exact
placeholders:

```text
{workspace}  clean candidate checkout
{request}    input request JSON path
{report}     output report JSON path
{artifacts}  writable artifact directory
```

The engine writes a schema-v1 envelope:

```python
class CommandEvaluationInput(BaseModel):
    schema_version: Literal["1"] = "1"
    request: EvaluationRequest
```

The envelope contains only the canonical request: candidate identity, evaluation set,
parameters, limits, and seed. It never contains authorization, backend configuration,
secrets, or host-only paths. The command writes a schema-v1 `EvaluationReport`. Both
schemas are ordinary JSON; the target and harness do not import VeRO.

Execution rules:

- run through the workspace sandbox from the configured working directory,
- load backend configuration once before optimization and never reload it from a
  candidate checkout,
- expand only the four documented placeholders and pass every argv element directly,
- inherit only the sandbox baseline environment, configured non-secret values, and
  runtime values named by `passthrough_environment`,
- never place secret values in persisted configuration or provenance,
- capture stdout and stderr as `EvaluationArtifact`s,
- require exit code zero and a present, valid report file for success,
- convert non-zero exit, timeout, missing output, malformed JSON, and invalid report
  data into the canonical failure lifecycle,
- terminate the process tree on timeout or cancellation,
- validate every returned artifact path before persistence, and
- include the command, non-secret environment representation, and backend version in
  the configuration digest.

`resolve_cost()` uses explicit `CaseIds` or `CaseRange` directly. For `AllCases`, the
harness may expose a trusted preflight count; otherwise only run-limited budgets are
allowed because case cost is unknown before execution.

## Generic Policy, configuration, and CLI path

The canonical Policy constructor no longer requires `dataset` or `task`:

```python
Policy(
    workspace=workspace,
    optimizer=optimizer,
    backends={"default": CommandBackend(config)},
    default_backend_id="default",
    evaluation_set=EvaluationSet(name="performance"),
    objective=objective,
)
```

Keep `agent=`, `dataset=`, and `task=` as deprecated compatibility inputs that construct
the current VeroTask setup.

Candidate production is intentionally separate from evaluation:

```python
class CandidateProducer(Protocol):
    async def propose(self, context: OptimizationContext) -> str | None: ...
```

The producer edits the supplied versioned workspace and returns a commit message. Policy
owns the commit, constructs the `Candidate`, invokes the evaluation engine, and selects
the best feasible objective result. `AgentCandidateProducer` adapts the existing single
coding-agent step; `CommandCandidateProducer` provides the config-driven boundary used
by arbitrary coding agents and deterministic tests. Population scheduling remains a
later optimization-strategy layer.

Add a config-driven path to both `vero evaluate` and `vero run`. The canonical filename
is `vero.toml`; both commands accept `--config PATH` and default to `./vero.toml`. A
minimal generic configuration is:

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
evaluation_set = "performance"

[objective]
metric = "latency_ms"
direction = "minimize"

[[objective.constraints]]
metric = "correct"
operator = "=="
value = 1

[optimizer]
root = "./optimizer"
command = ["./optimize", "--workspace", "{workspace}"]
max_candidates = 3
```

The CLI creates the backend registry, evaluation set, and objective from this trusted
configuration. Existing flags remain supported and translate to the VeroTask
compatibility path. The optimizer may edit only `target.root`; the active harness and
configuration are not part of candidate commits.

## Non-Python release proof and product claim

Add a runnable C matrix-multiplication target with:

- no Python package, VeRO import, dataset dependency, or agent framework,
- a deliberately improvable baseline implementation,
- a command harness that builds the checked-out candidate, verifies numerical
  correctness across multiple matrix sizes, and reports latency,
- `ObjectiveSpec` minimizing latency subject to correctness,
- at least one candidate commit created through the ordinary Policy optimization loop,
  and
- durable baseline, candidate, objective, best-version, and artifact records using the
  canonical models.

The automated end-to-end test uses a deterministic local optimizer fixture so CI does
not depend on an external model. The documented example uses the same `vero run`
configuration path available to users and may use a normal coding agent.

The test must demonstrate that the selected best candidate is feasible and improves
the baseline objective. Merely evaluating a static non-Python repository is
insufficient.

After this proof passes, VeRO may claim:

> VeRO is a generic program optimizer: it can optimize a versioned program repository
> independently of its implementation language or agent framework, provided the target
> has an approved evaluation backend or command harness configured.

This does not claim that VeRO can infer an arbitrary program's build, test, or objective
without configuration.

## Canonical evaluator lifecycle

`Evaluator.evaluate(request, backend, objective_spec=None, use_copy=None)` performs:

1. Validate the request and objective.
2. Generate evaluation ID and initialize its result directories.
3. Persist a minimal `running` manifest containing request and provenance.
4. Resolve the candidate commit through the workspace.
5. Reject a dirty direct workspace before backend work.
6. Create a temporary worktree when configured; otherwise use the current workspace.
7. Enter the candidate version and construct `EvaluationContext`.
8. Invoke the backend under the overall timeout.
9. Validate the report and every artifact path.
10. Compute objective result when configured.
11. Persist the complete schema-v2 evaluation atomically.
12. Return `EvaluationRecord`.

Failure behavior:

- a backend-returned failed report produces and returns a failed record,
- timeout, cancellation, backend exception, invalid JSON/model output, or persistence
  failure produces a best-effort failed manifest with sanitized error information,
- after recording a thrown failure, raise `ExperimentRunFailedError` with the new
  evaluation ID for compatibility,
- cancellation is re-raised after cleanup and must not be swallowed,
- callbacks are an engine responsibility and run only after the record is durable,
- callback failures are logged and do not change the evaluation record.

## Evaluation engine lifecycle

`EvaluationEngine.evaluate()` accepts `backend_id`, `request`, optional
`objective_spec`, and optional `authorization`. It performs:

1. Resolve `backend_id` from the approved backend registry.
2. Resolve authorization for the backend and evaluation set when not supplied by a
   trusted internal frontend.
3. Reject denied combinations before commit transfer, checkout, or budget debit.
4. Resolve case selection and `EvaluationCost` through the backend.
5. Atomically reserve budget when `meter_budget=True`.
6. Invoke the evaluator with the resolved backend.
7. Insert the durable record into `EvaluationDatabase`.
8. Fire evaluation callbacks.
9. Return the disclosure projection requested by the trusted authorization context;
   internal full-evaluation methods return the complete record.

Provide separate internal methods for full record access and frontend projection so a
caller cannot obtain full data merely by casting an aggregate response.

## Schema-v2 persistence

Continue using:

```text
<vero-home>/sessions/<session-id>/experiments/<evaluation-id>/
```

New layout:

```text
evaluation.json
cases/
  <sha256(case-id)>.json
artifacts/
backend/
  <backend-name>/...
```

`evaluation.json` is a manifest containing:

```json
{
  "schema_version": 2,
  "id": "...",
  "request": {},
  "report": {
    "schema_version": "1",
    "status": "success",
    "metrics": {},
    "diagnostics": [],
    "artifacts": [],
    "error": null
  },
  "case_files": [
    {"case_id": "...", "path": "cases/<digest>.json"}
  ],
  "backend": {},
  "objective_spec": {},
  "objective": {},
  "created_at": "...",
  "completed_at": "..."
}
```

Cases are excluded from the embedded report and reconstructed from `case_files` in
manifest order. The in-memory `EvaluationRecord` always contains the complete report.

Writes use a temporary manifest and atomic replace. A complete manifest is written only
after every referenced case file exists. A running or failed best-effort manifest may
omit cases but must state its lifecycle status distinctly from report status.

### Database schema v2

```json
{
  "schema_version": 2,
  "id": "session-id",
  "candidates": {},
  "evaluations": {},
  "datasets": {}
}
```

`EvaluationDatabase` provides:

- `add_evaluation(record)`,
- `get_evaluation(id)`,
- `get_evaluations(...)`,
- `get_evaluations_df(...)`,
- `get_best(objective_spec, backend_ids=None, evaluation_sets=None,
  exclude_candidate=None)`, and
- schema-v1 and schema-v2 deserialization.

Database records may include full case results, matching current `database.json`
behavior. The experiment directory remains the reconstruction source of truth when the
database file is missing.

Database writes are serialized and atomic. Concurrent evaluations may execute in
parallel but must not overwrite one another's database dump.

## Legacy-data conversion

Schema v1 data is converted in memory and never rewritten automatically.

Conversion rules:

```text
Candidate                    -> unchanged Candidate
DatasetSubset                -> flat EvaluationSet + CaseSelection
EvaluationParameters limits  -> EvaluationLimits
task_params                  -> request.parameters
SampleResult                 -> CaseResult
ExperimentResult.status      -> EvaluationReport.status
ExperimentResult statistics  -> report.metrics
legacy operational messages   -> report.diagnostics
result ID                    -> EvaluationRecord.id
```

Use `str(sample_id)` as canonical case ID and preserve the original integer in case
metadata. Preserve legacy timestamps and messages when present. Attach VeroTask backend
provenance marked `version="legacy-v1"` and the compatibility objective.

Convert legacy `error` and `eval_error` fields into separate execution and scoring
`CaseError`s. If a failed legacy case has no error message, synthesize a terminal error
with `code="legacy_missing_error"`, `phase="legacy"`, and a non-sensitive explanatory
message so canonical status invariants remain valid.

If a legacy record cannot be converted, log its result directory and continue loading
other records; do not synthesize a successful empty evaluation.

New writes never emit `evaluation_parameters.json`, root `samples/`, or
`result_metadata.json`. The VeroTask backend may retain those formats under its private
backend staging directory.

## Internal consumer migration

### Policy and selection

- Add `Policy.objective`, defaulting to the VeroTask compatibility objective.
- Let `Policy` configure an approved backend registry and a default backend ID. The
  legacy `dataset=..., task=...` constructor registers one `VeroTaskBackend` as the
  default.
- Internal evaluation uses `evaluate_candidate()` and receives `EvaluationRecord`.
- Best-version selection queries persisted `ObjectiveResult` rather than sorting a
  dataframe `mean_score` column.
- Dataset split preferences become evaluation-set filters.
- `BestVersion.score` remains as a compatibility field populated from objective value;
  add objective metric and evaluation ID to its canonical representation.

### Tools

- Introduce canonical `EvaluationRunnerTool.evaluate_candidate()` accepting candidate,
  approved backend ID, evaluation set, and parameters.
- Keep `ExperimentRunnerTool.evaluate_commit()` as a deprecated dataset wrapper.
- Budget status comes from the engine's single ledger.
- Viewer tools expose evaluation summary, report metrics, cases, and artifacts.
- Legacy sample-table methods operate only on dataset-backed records and emit a
  deprecation warning.

### Artifacts, traces, CLI, and logging

- Replace direct `dataset_subset` access with evaluation-set-aware labels.
- Replace `mean_score` assumptions with objective value and metric name.
- Preserve legacy trace-analysis column names only in compatibility export views.
- W&B logs objective value, feasibility, selected metric, evaluation-set key, and report status.
- Callbacks move from `on_experiment` to `on_evaluation`; the old callback name receives
  a legacy view for dataset evaluations.

### Public exports

Export canonical models from `vero.evaluation` and the package root where current DB
models are exported. Keep legacy imports working with `DeprecationWarning` rather than
removing them in Phase A.

## Compatibility API behavior

### New canonical API

```python
record = await policy.evaluate_candidate(
    commit="abc123",
    backend_id="performance",
    evaluation_set=EvaluationSet(name="large-matrices"),
)
```

This returns `EvaluationRecord`.

### Existing dataset API

```python
experiment = await policy.evaluate_version(
    commit="abc123",
    split="validation",
    sample_ids=[0, 1, 2],
)
```

This remains callable and returns a deprecated `Experiment` view backed by the new
record. It is available only when the record was produced by `VeroTaskBackend` and its
evaluation set can be converted to the legacy dataset representation.

Legacy view mapping:

- case IDs must be convertible to integer sample IDs,
- case metric `score` becomes `SampleResult.score`,
- execution and scoring errors map back to `error` and `eval_error`; when a phase is
  unavailable, the final terminal error becomes the legacy `error`,
- missing fields use the existing legacy defaults,
- `ExperimentResult.score()` delegates to the persisted compatibility objective result,
- legacy dataframe columns are generated by the adapter rather than stored canonically.

`main` did not expose an `EvalRequest` or `DatasetEvaluationRequest` public class, so
Phase A does not create a second request alias. The existing `Policy.evaluate_version()`
and `ExperimentRunnerTool.evaluate_commit()` calls are the deprecated dataset request
adapters and convert immediately into `EvaluationRequest`.

## Concurrency and isolation

Phase A must support safe concurrent evaluations even though it does not implement
population search:

- every evaluation gets a UUID and independent directory,
- temporary worktrees are created per evaluation when `use_copy=True`,
- direct-workspace evaluation remains prohibited on dirty state,
- no backend uses `os.chdir` or mutable process-global result paths,
- case checkpoint writes are per-evaluation and atomic,
- budget reservations are locked and occur before work begins,
- database insertion and database-file dumps are serialized,
- callbacks cannot mutate or replace the persisted record,
- a backend may serialize its own scarce resource with a semaphore,
- generic resource scheduling is deferred.

Evaluation fingerprints include candidate identity, evaluation set, parameters, limits that
affect results, seed, backend digest, and objective spec. Phase A records fingerprints
for provenance but does not add result caching.

## Security invariants

- Backends cannot choose or rewrite the candidate commit being evaluated.
- Backends receive only the checked-out workspace and their configured evaluator state.
- Evaluator/harness code and active backend configuration live outside the writable
  target workspace.
- Request parameters cannot grant authorization, budget exemptions, or disclosure.
- Aggregate output cannot contain case details, diagnostics, errors, traces, or artifact
  paths.
- Artifact paths and case checkpoint paths cannot escape their evaluation directory.
- Secrets never appear in requests, reports, provenance config, diagnostics, or errors.
- Error strings included in persisted records are sanitized for configured secret
  values before writing.

Harbor's OS/container trust boundary remains Phase C; Phase A supplies the types and
projection rules it will enforce.

## Implementation sequence in the new Phase A PR

Implement the foundation as one pull request, using the following logical commits to
keep review and bisectability manageable. The PR is merged only when the complete
sequence is green; none of these commits defines a separately merged architecture.

### 0. Establish the independent baseline

- Create `program-optimization-foundation` directly from `main` before implementation.
- Record the current VeroTask, Policy, CLI, persistence, and end-to-end behavior with
  characterization tests.
- Reuse ideas or isolated implementation details from PR #3 only when they fit the new
  contracts; do not merge or depend on its evaluator architecture.
- Keep Harbor imports and split-tier policy out of Phase A.

### 1. Domain models and objective evaluator

- Add flat evaluation sets, case selection, request/limits, cases, report, provenance,
  objective, record, disclosure, authorization, and generic budget models.
- Add validation, fingerprinting, objective evaluation, comparison, and summary
  projection tests.
- Do not wire production evaluation yet.

### 2. Schema-v2 persistence and legacy loading

- Add checkpoint store, manifest store, and `EvaluationDatabase`.
- Add atomic persistence and reconstruction.
- Add schema-v1 database and result-directory converters while keeping the existing
  `ExperimentDatabase` import available to compatibility callers.
- Test v1 fixtures before changing the evaluator.

### 3. VeroTask backend extraction

- Move Python/`uv` discovery and task execution out of `Evaluator`.
- Preserve current `main` behavior and keep legacy files backend-private.
- Return canonical reports and add parity tests against current expected results.

### 4. Evaluator, engine, authorization, and budget migration

- Route production evaluation through `EvaluationBackend`.
- Replace the inline evaluator implementation with explicit backends.
- Add the approved backend registry and resolve one backend per evaluation.
- Establish the single ledger and trusted authorization object.
- Persist failure records and callback only after durability.

### 5. Command backend and generic configuration

- Implement the versioned request/report JSON file contract and sandboxed argv
  execution.
- Add command failure, timeout, cancellation, artifact, and cost-resolution tests.
- Add trusted config loading for target, backend, evaluation set, and objective.

### 6. Internal consumers, Policy, CLI, and compatibility adapters

- Migrate Policy, tools, viewers, artifacts, traces, CLI, W&B, and package exports.
- Make the canonical path dataset-free and add `vero evaluate` and `vero run` config
  entry points.
- Add canonical APIs and deprecated VeroTask/experiment views.
- Remove internal reads of dataset dataframe columns except inside adapters.

### 7. Non-Python proof, documentation, and Harbor handoff

- Delete obsolete internal experiment code that is no longer an adapter or VeroTask
  implementation detail.
- Run the full suite and add concurrent-evaluation coverage.
- Add the C matrix-multiplication example and deterministic end-to-end test.
- Document the generic program-optimizer quickstart and precise product claim.
- Prepare reconstruction notes for Harbor PR #3.

## Test plan

### Model and objective tests

- Round-trip `EvaluationSet` with every case-selection variant through JSON.
- Reject conflicting selectors, duplicate IDs, negative samples, empty names, unsafe
  artifacts, non-finite metrics, and invalid status/error-history combinations.
- Reject negative range starts and stops that are not greater than their start.
- Cover multiple errors, retry histories on successful cases, multiple terminal errors,
  and legacy failures with synthesized error details.
- Evaluate maximize and minimize objectives from report and case metrics.
- Cover mean, median, min, and max reducers.
- Cover every constraint operator, multiple violations, missing metrics, failed reports,
  failure values, infeasible ordering, and deterministic ties.
- Verify aggregate projection contains no sensitive fields.

### Backend contract tests

Use a fake backend to cover:

- aggregate-only success with no cases,
- case-based success,
- successful report with some errored cases,
- backend-returned failure,
- backend exception,
- timeout and cancellation,
- invalid report and unsafe artifact,
- checkpoint/resume,
- repeated identical requests producing distinct records,
- concurrent requests producing isolated directories and cases.

### VeroTask parity tests

- Task discovery and missing-task errors.
- Required environment variables.
- Same-project and separate `task_project` execution.
- `uv --with-editable` behavior.
- Dataset full split, explicit sample IDs, zero-based ranges, and nonzero range starts.
- Error threshold, retry, timeouts, threading, and custom task parameters.
- Existing score/statistics results and legacy view output.

Staged inference/scoring, label scrubbing, and resume parity are tested when Harbor PR
#3 is reconstructed on top of Phase A.

### Command backend tests

- Placeholder expansion without shell interpretation.
- Working-directory and environment isolation.
- Valid aggregate-only and case-based report JSON.
- stdout and stderr artifact capture.
- Non-zero exit, missing output, malformed JSON, invalid report, and unsafe artifact
  paths.
- Overall timeout, process-tree termination, and cancellation cleanup.
- Known and unknown case-cost resolution.
- Zero-based and nonzero `CaseRange` execution and cost resolution.
- Configuration digest excludes secret values while changing for behaviorally relevant
  configuration.

### Persistence and migration tests

- Schema-v2 manifest, cases, database, and experiment-directory round trips.
- Atomic write interruption never exposes a complete manifest referencing missing cases.
- Corrupt case files fail only the affected record with a clear path.
- Schema-v1 database and result-directory fixtures convert deterministically.
- Loading v1 followed by saving creates v2 without mutating the v1 source.
- Missing database file reconstructs from evaluation directories.
- Concurrent database inserts do not lose records.

### Engine, budget, and authorization tests

- Denied request performs no checkout, reservation, backend call, or persistence.
- Unknown backend IDs fail before checkout, reservation, or persistence.
- Two registered backends may resolve the same evaluation-set name independently.
- Metered request reserves exactly once.
- Unmetered request does not bypass access.
- Backend failure consumes a reservation after work begins.
- Unknown case count is rejected for a case-limited budget.
- Backend-qualified evaluation-set keys remain stable.
- SplitBudget adapter reproduces current budget messages and status.

### Consumer and compatibility tests

- `Policy.run()` baseline, step, best selection, and final evaluation.
- Maximize and minimize best selection.
- No feasible non-baseline candidate behavior.
- Canonical and deprecated evaluation tools.
- Viewer summaries and case tables.
- Artifact and trace materialization.
- CLI check/evaluate/session output.
- Config-driven dataset-free `vero evaluate` and `vero run`.
- W&B summary fields.
- Legacy imports, callbacks, dataframe columns, and serialized experiment views.
- Existing end-to-end optimization suite passes without Harbor PR #4 or #5.
- A deterministic optimizer improves and selects a feasible non-Python
  matrix-multiplication candidate through the normal Policy loop.

## Acceptance criteria

Phase A is complete only when all of the following are true:

- `EvaluationBackend` contains no import or field requiring datasets, `uv`, Python, or
  agents.
- A fake backend evaluates a candidate and returns aggregate metrics without cases.
- `CommandBackend` evaluates a target using only the versioned JSON file contract and
  sandbox filesystem/process boundary.
- The production VeroTask path runs exclusively through `VeroTaskBackend`.
- All new durable records use schema v2.
- Existing schema-v1 sessions load without being rewritten.
- Internal best-candidate selection never sorts `mean_score` directly.
- Minimize and maximize objectives behave correctly with constraints.
- There is one authoritative budget ledger and no tool-owned decrement path.
- Full and aggregate disclosure are structurally different models.
- Concurrent evaluations cannot collide in worktrees, result paths, checkpoints, or
  database writes.
- Existing dataset-oriented public calls remain callable with deprecation warnings.
- The canonical Policy constructor and config-driven CLI path work without `dataset`,
  `task`, or a VeroTask backend.
- The full non-Harbor test suite passes.
- Harbor PR #3 can be reconstructed as a consumer of these contracts without adding a
  second evaluation model.
- An engine can register two backends, evaluate the same neutral set shape through
  either one, and retain distinct provenance, budget identity, and results.
- The non-Python matrix-multiplication target is improved end to end, and the persisted
  objective selects the improved feasible candidate over its baseline.
- The README quickstart substantiates the qualified claim that VeRO is a generic
  program optimizer.

## Explicit non-goals

- Harbor split tiers, staged inference/scoring, label scrubbing, sidecar, verifier,
  compiler, Mode A/Mode B configuration, or HarborRunner conversion.
- Result caching.
- Build caching.
- Generic CPU/GPU/distributed scheduling.
- Multi-objective or Pareto selection.
- Atomic multi-backend evaluation plans and objectives that combine separate reports.
- Candidate populations, proposals, worker agents, Best-of-N, or evolutionary search.
- Removing deprecated experiment APIs in the same release.

## Harbor handoff after Phase A

After the new Phase A PR stabilizes:

1. Preserve the current `harbor-1-core` head as a backup reference.
2. Create a replacement Harbor core branch from the Phase A head.
3. Port three-tier split visibility, fail-closed access, VeroTask staging/resume, label
   scrubbing, and ledger hardening with focused commits and parity tests.
4. Do not port `EvalStrategy`, the dataset-bound engine/request, or the alternative
   evaluator layout.
5. Update PR #3 to the replacement branch and review its diff relative to Phase A.
6. Keep PR #4 based on `harbor-1-core`; adapt it only after reconstructed PR #3 is
   stable.
7. Make the sidecar construct canonical requests and trusted authorization decisions,
   use aggregate projection for non-viewable evaluation sets, and query objective
   results for `auto_best`.
8. Rebase PR #5 and later Harbor branches in order, converting `HarborRunner` into an
   `EvaluationBackend` implementation when its turn is reached.

## Branch and merge mechanics

- Create `program-optimization-foundation` directly from `main`, preserving these design documents.
- Open `program-optimization-foundation` as a fresh Phase A PR targeting `main`.
- During stacked review, set PR #3's base to `program-optimization-foundation`. Once Phase
  A merges, PR #3 can target `main` with only its Harbor-specific diff.
- Reconstruct PR #3 instead of blindly rebasing its combined evaluator commit. Update
  the existing remote branch only after the replacement passes its focused and full
  test suites.
- Keep PR #4 based on `harbor-1-core` and adapt it after reconstructed PR #3 stabilizes.
- Rebase PR #5 and later branches in their existing order after PR #4 is adapted.
- Merge Phase A only when the generic program-optimizer acceptance checklist passes.
- Never merge the temporary `EvalStrategy` architecture merely to unblock the stack.

## Reference material

- Overall architecture and phased roadmap:
  `docs/design/program-evaluation-architecture.md`
- Harbor core foundation: https://github.com/scaleapi/vero/pull/3
- Harbor evaluation sidecar: https://github.com/scaleapi/vero/pull/4
- Harbor nested runner/compiler: https://github.com/scaleapi/vero/pull/5
- Harbor docs/example: https://github.com/scaleapi/vero/pull/6
