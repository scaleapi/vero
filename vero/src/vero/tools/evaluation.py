"""Agent-facing tools for the scoped evaluation capability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vero.evaluation import CaseSelection, EvaluationBudgetExceeded
from vero.optimization import CandidateEvaluationGateway
from vero.tools.utils import is_tool

if TYPE_CHECKING:
    from vero.agents.protocol import AgentContext


@dataclass
class EvaluationTools:
    """Evaluate current or prior programs through the session's named evaluations."""

    exclude_tools: list[str] = field(default_factory=list)
    evaluation: CandidateEvaluationGateway | None = field(default=None, repr=False)

    def bind(self, context: AgentContext) -> None:
        self.evaluation = context.evaluation

    def _gateway(self) -> CandidateEvaluationGateway:
        if self.evaluation is None:
            raise RuntimeError("evaluation tools are not bound to an agent context")
        return self.evaluation

    @is_tool
    async def evaluate(
        self,
        evaluation: str,
        selection: CaseSelection | None = None,
        candidate_id: str | None = None,
        description: str = "Evaluate agent checkpoint",
    ) -> str:
        """Evaluate a program, returning authorized feedback and a filesystem path.

        Args:
            evaluation: Name from ``.vero/evaluations.json``.
            selection: Optional case IDs or range. Omit to use the base selection.
            candidate_id: Existing candidate to re-evaluate. Omit to save and
                evaluate the current workspace.
            description: A short description of the program changes being evaluated.

        Returns:
            A bounded JSON receipt with an authorized summary and the path to the
            complete permitted feedback under ``.vero/evaluations``.
        """

        try:
            result = await self._gateway().evaluate(
                evaluation=evaluation,
                selection=selection,
                candidate_id=candidate_id,
                description=description,
            )
        except EvaluationBudgetExceeded as error:
            # Running out of evaluation budget is expected and benign: hand the
            # agent a clean result rather than raising, so it stops evaluating
            # and proceeds with what it already knows. The final held-out
            # evaluation runs on a separate trusted path and is unaffected.
            return json.dumps(
                {
                    "status": "evaluation_budget_exhausted",
                    "message": str(error),
                    "detail": (
                        "The evaluation budget for this evaluation is spent. "
                        "Further evaluations of this kind are unavailable; "
                        "proceed with the feedback you already have."
                    ),
                },
                indent=2,
            )
        return result.model_dump_json(indent=2)

    @is_tool
    def get_evaluation_budgets(self) -> str:
        """Return remaining agent budgets for every available evaluation."""

        budgets = self._gateway().budgets()
        return json.dumps(
            {
                name: (
                    budget.model_dump(mode="json")
                    if budget is not None
                    else None
                )
                for name, budget in sorted(budgets.items())
            },
            indent=2,
        )
