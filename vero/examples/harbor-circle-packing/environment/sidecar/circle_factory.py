"""Custom Harbor sidecar factory for circle-packing.

Harbor drives the OUTER loop (a coding agent edits packing.py); the target is
scored by a plain CommandBackend (circle-packing's evaluate.py) — NOT a nested
harbor run. Same shape as the proven harbor-live demo, swapped to the
sum_radii / valid objective.
"""

from __future__ import annotations

from pathlib import Path

from vero.candidate_repository import GitCandidateRepository
from vero.evaluation import (
    BackendRegistry,
    BudgetLedger,
    CaseRange,
    ConstraintOperator,
    EvaluationBudget,
    EvaluationDatabase,
    EvaluationSet,
    Evaluator,
    MetricConstraint,
    MetricSelector,
    ObjectiveSpec,
)
from vero.evaluation.backends.command import CommandBackend, CommandBackendConfig
from vero.evaluation.engine import EvaluationEngine
from vero.sandbox import LocalSandbox
from vero.sidecar.serve import SidecarComponents
from vero.sidecar.sidecar import EvaluationSidecar, SidecarEvaluationPolicy
from vero.sidecar.transport import GitCandidateTransport
from vero.sidecar.verifier import (
    CanonicalVerifier,
    VerificationSelection,
    VerificationTarget,
)
from vero.workspace import GitWorkspace

SET = "circle-packing"
OBJ = ObjectiveSpec(
    selector=MetricSelector(metric="sum_radii"),
    direction="maximize",
    failure_value=0.0,
    constraints=[
        MetricConstraint(
            selector=MetricSelector(metric="valid"),
            operator=ConstraintOperator.EQ,
            value=1.0,
        )
    ],
)


def _backend(harness_root: str) -> CommandBackend:
    return CommandBackend(
        CommandBackendConfig(
            harness_root=harness_root,
            command=[
                "uv", "run", "--python", "3.12", "python", "{harness}/evaluate.py",
                "--workspace", "{workspace}",
                "--request", "{request}",
                "--report", "{report}",
                "--artifacts", "{artifacts}",
            ],
            environment={"PYTHONHASHSEED": "0"},
            passthrough_environment=["PATH", "HOME", "UV_CACHE_DIR"],
        )
    )


async def build(config: dict) -> SidecarComponents:
    repo_path = config["repo_path"]
    agent_repo_path = config["agent_repo_path"]
    session_dir = Path(config["session_dir"])
    harness_root = config["harness_root"]
    session_dir.mkdir(parents=True, exist_ok=True)

    sandbox = await LocalSandbox.create(root=Path(repo_path).parent)
    workspace = await GitWorkspace.from_path(sandbox, repo_path)
    candidate_repository = await GitCandidateRepository.create(
        session_dir / "candidates", workspace=workspace
    )
    database = EvaluationDatabase.load_reconciled(
        database_path=session_dir / "database.json",
        evaluations_dir=session_dir / "evaluations",
        database_id="circle",
    )
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="cmd",
                evaluation_set_key=f"cmd:{SET}:validation",
                total_runs=10,
            )
        ],
        path=session_dir / "budgets.json",
    )
    ledger.save()

    engine = EvaluationEngine(
        evaluator=Evaluator(
            candidate_repository=candidate_repository,
            sandbox=workspace.sandbox,
            session_dir=session_dir,
            session_id="circle",
        ),
        backends=BackendRegistry({"cmd": _backend(harness_root)}),
        database=database,
        database_path=session_dir / "database.json",
        budget_ledger=ledger,
    )
    transport = GitCandidateTransport(
        workspace=workspace,
        candidate_repository=candidate_repository,
        agent_repo_path=agent_repo_path,
    )
    baseline = await transport.trusted_candidate("HEAD")

    selection = VerificationSelection(
        mode="auto_best",
        backend_id="cmd",
        evaluation_set=EvaluationSet(
            name=SET, partition="validation", selection=CaseRange(start=0, stop=1)
        ),
        objective=OBJ,
        baseline_candidate=baseline,
    )
    sidecar = EvaluationSidecar(
        engine=engine,
        candidate_transport=transport,
        access_policies=[
            SidecarEvaluationPolicy(
                backend_id="cmd", evaluation_set_name=SET, partition="development",
                disclosure="full", expose_case_resources=True, objective=OBJ,
            ),
            SidecarEvaluationPolicy(
                backend_id="cmd", evaluation_set_name=SET, partition="validation",
                disclosure="aggregate", min_aggregate_cases=1, objective=OBJ,
            ),
        ],
        admin_volume=session_dir / "admin",
    )
    await sidecar.initialize_context()

    verifier = CanonicalVerifier(
        engine=engine,
        selection=selection,
        targets=[
            VerificationTarget(
                reward_key="sum_radii", backend_id="cmd",
                evaluation_set=EvaluationSet(name=SET, partition="test"),
                objective=OBJ, max_attempts=1,
            )
        ],
        admin_volume=session_dir / "admin",
        score_baseline=True,
    )
    return SidecarComponents(sidecar=sidecar, verifier=verifier)
