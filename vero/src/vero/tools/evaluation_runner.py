"""Agent tools for canonical, dataset-free candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from vero.evaluation import DisclosureLevel, project_evaluation
from vero.tools.utils import is_tool
from vero.workspace import GitWorkspace


@dataclass
class EvaluationRunnerTool:
    """Evaluate candidates through a Policy's approved backend and objective."""

    exclude_tools: list[str] = field(default_factory=list)
    policy: object | None = None

    def bind(self, session) -> None:
        self.policy = session.policy

    @is_tool
    async def evaluate_candidate(self, commit: str = "HEAD") -> str:
        """Evaluate a candidate commit and return aggregate measurements.

        Args:
            commit: Git commit or ref to evaluate. Defaults to HEAD.
        """
        if self.policy is None:
            raise ValueError("EvaluationRunnerTool requires a generic program Policy")
        outer_session = getattr(self.policy, "session", None)
        workspace = (
            outer_session.workspace
            if outer_session is not None
            else getattr(self.policy, "workspace", None)
        )
        if workspace is None:
            raise ValueError("EvaluationRunnerTool requires an initialized Policy")
        if isinstance(workspace, GitWorkspace):
            commit = await workspace.resolve_ref(commit)
        evaluate_candidate = getattr(self.policy, "evaluate_candidate", None)
        if not callable(evaluate_candidate):
            raise TypeError("bound policy does not implement evaluate_candidate()")
        record = await evaluate_candidate(commit)
        summary = project_evaluation(record, DisclosureLevel.AGGREGATE)
        return summary.model_dump_json(indent=2)

    @is_tool
    async def evaluation_budget(self) -> str:
        """Return the budget for this Policy's backend and evaluation set."""
        if self.policy is None:
            raise ValueError("EvaluationRunnerTool requires a generic program Policy")
        session = getattr(self.policy, "session", None)
        engine = getattr(self.policy, "engine", None)
        ledger = (
            session.budget_ledger
            if session is not None
            else engine.budget_ledger
            if engine is not None
            else None
        )
        if ledger is None:
            return "No evaluation budget is configured."
        backend_id = (
            session.backend_id
            if session is not None
            else getattr(self.policy, "backend_id", None)
        )
        evaluation_set = (
            session.evaluation_set
            if session is not None
            else getattr(self.policy, "evaluation_set", None)
        )
        if backend_id is None or evaluation_set is None:
            raise ValueError("Policy does not have a configured evaluation target")
        budget = ledger.get(
            backend_id,
            evaluation_set,
        )
        if budget is None:
            return "No limit is configured for this backend and evaluation set."
        return budget.model_dump_json(indent=2)
