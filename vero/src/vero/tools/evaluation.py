"""Agent-facing tools for the scoped evaluation capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vero.optimization import CandidateEvaluationGateway
from vero.tools.utils import is_tool

if TYPE_CHECKING:
    from vero.agents.protocol import AgentContext


@dataclass
class EvaluationTools:
    """Evaluate the current program and inspect the remaining evaluation budget."""

    exclude_tools: list[str] = field(default_factory=list)
    evaluation: CandidateEvaluationGateway | None = field(default=None, repr=False)

    def bind(self, context: AgentContext) -> None:
        self.evaluation = context.evaluation

    def _gateway(self) -> CandidateEvaluationGateway:
        if self.evaluation is None:
            raise RuntimeError("evaluation tools are not bound to an agent context")
        return self.evaluation

    @is_tool
    async def evaluate_current(
        self,
        description: str = "Evaluate agent checkpoint",
    ) -> str:
        """Save and evaluate the current program, returning a feedback receipt.

        Args:
            description: A short description of the program changes being evaluated.

        Returns:
            A bounded JSON receipt with an authorized summary and the path to the
            complete permitted feedback under ``.vero/evaluations``.
        """

        result = await self._gateway().evaluate_current(description=description)
        return result.model_dump_json(indent=2)

    @is_tool
    def get_evaluation_budget(self) -> str:
        """Return the remaining evaluation budget as JSON, or an unmetered notice."""

        budget = self._gateway().budget()
        if budget is None:
            return "Evaluation budget is not metered for this run."
        return budget.model_dump_json(indent=2)
