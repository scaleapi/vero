"""Custom Harbor sidecar factory for ALE-Bench (program optimization).

vero drives the OUTER loop: a coding agent edits /work/agent/solution.cpp to
maximize the ALE-Bench score for a fixed AHC problem. Each candidate is scored by
a plain CommandBackend (harness/score.py -> ALE-Bench's deterministic judge),
NOT a nested `harbor run`. Disclosure maps onto ALE-Bench's public/private split:
development+validation are scored on public seeds, test on private seeds.
"""

from __future__ import annotations

from pathlib import Path

from vero.candidate_repository import GitCandidateRepository
from vero.evaluation import (
    BackendRegistry,
    BudgetLedger,
    CaseRange,
    EvaluationBudget,
    EvaluationDatabase,
    EvaluationSet,
    Evaluator,
    MetricSelector,
    ObjectiveSpec,
)
from vero.evaluation.command import CommandBackend, CommandBackendConfig
from vero.evaluation.engine import EvaluationEngine
from vero.harbor.serve import SidecarComponents
from vero.harbor.sidecar import EvaluationSidecar, SidecarEvaluationPolicy
from vero.harbor.transport import GitCandidateTransport
from vero.harbor.verifier import CanonicalVerifier, VerificationSelection, VerificationTarget
from vero.sandbox import LocalSandbox
from vero.workspace import GitWorkspace

SET = "ale-bench"
# ahc011 is a maximization problem; ALE-Bench zeroes invalid/CE/TLE submissions.
OBJ = ObjectiveSpec(
    selector=MetricSelector(metric="score"),
    direction="maximize",
    failure_value=0.0,
)


def _backend(harness_root: str) -> CommandBackend:
    return CommandBackend(
        CommandBackendConfig(
            harness_root=harness_root,
            command=[
                "python", "{harness}/score.py",
                "--workspace", "{workspace}",
                "--request", "{request}",
                "--report", "{report}",
                "--artifacts", "{artifacts}",
            ],
            # The judge runs the submission in Docker; the scorer needs daemon access.
            passthrough_environment=["PATH", "HOME", "DOCKER_HOST", "HF_TOKEN"],
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
        database_id="ale",
    )
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="cmd",
                evaluation_set_key=f"cmd:{SET}:development",
                total_runs=40,
            ),
            EvaluationBudget(
                backend_id="cmd",
                evaluation_set_key=f"cmd:{SET}:validation",
                total_runs=40,
            ),
        ],
        path=session_dir / "budgets.json",
    )
    ledger.save()

    engine = EvaluationEngine(
        evaluator=Evaluator(
            candidate_repository=candidate_repository,
            sandbox=workspace.sandbox,
            session_dir=session_dir,
            session_id="ale",
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
                reward_key="score", backend_id="cmd",
                evaluation_set=EvaluationSet(name=SET, partition="test"),
                objective=OBJ, max_attempts=1,
            )
        ],
        admin_volume=session_dir / "admin",
        score_baseline=True,
    )
    return SidecarComponents(sidecar=sidecar, verifier=verifier)
