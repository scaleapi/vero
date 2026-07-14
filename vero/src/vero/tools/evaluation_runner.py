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
    program_policy: object | None = None

    def bind(self, session) -> None:
        self.program_policy = session.program_policy

    @is_tool
    async def evaluate_candidate(self, commit: str = "HEAD") -> str:
        """Evaluate a candidate commit and return aggregate measurements.

        Args:
            commit: Git commit or ref to evaluate. Defaults to HEAD.
        """
        if self.program_policy is None:
            raise ValueError("EvaluationRunnerTool requires a generic program Policy")
        workspace = self.program_policy.workspace
        if isinstance(workspace, GitWorkspace):
            commit = await workspace.resolve_ref(commit)
        record = await self.program_policy.evaluate_version(commit)
        summary = project_evaluation(record, DisclosureLevel.AGGREGATE)
        return summary.model_dump_json(indent=2)

    @is_tool
    async def evaluation_budget(self) -> str:
        """Return the budget for this Policy's backend and evaluation set."""
        if self.program_policy is None:
            raise ValueError("EvaluationRunnerTool requires a generic program Policy")
        ledger = self.program_policy.engine.budget_ledger
        if ledger is None:
            return "No evaluation budget is configured."
        budget = ledger.get(
            self.program_policy.backend_id,
            self.program_policy.evaluation_set,
        )
        if budget is None:
            return "No limit is configured for this backend and evaluation set."
        return budget.model_dump_json(indent=2)
