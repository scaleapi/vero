from __future__ import annotations

import json

import pytest

from vero.evaluation import (
    EvaluationAcknowledgement,
    EvaluationBudget,
    EvaluationBudgetExceeded,
    EvaluationSet,
    EvaluationStatus,
)
from vero.tools.evaluation import EvaluationTools
from vero.tools.utils import get_tools_from_class


class StubEvaluationGateway:
    def __init__(self):
        self.descriptions: list[str] = []
        self.remaining_runs = 2

    async def evaluate(
        self,
        *,
        evaluation,
        selection=None,
        candidate_id=None,
        description="Evaluate agent checkpoint",
    ):
        assert evaluation == "validation"
        assert selection is None
        assert candidate_id is None
        self.descriptions.append(description)
        self.remaining_runs -= 1
        return EvaluationAcknowledgement(
            evaluation_id="evaluation-1",
            status=EvaluationStatus.SUCCESS,
        )

    def budgets(self):
        return {
            "validation": EvaluationBudget(
                backend_id="command",
                evaluation_set_key=EvaluationSet(name="validation").budget_key(
                    "command"
                ),
                total_runs=2,
                remaining_runs=self.remaining_runs,
            )
        }


@pytest.mark.asyncio
async def test_evaluation_tools_expose_only_scoped_feedback_and_budget():
    gateway = StubEvaluationGateway()
    tools = EvaluationTools(evaluation=gateway)

    result = json.loads(
        await tools.evaluate(
            evaluation="validation",
            description="Try vectorized implementation",
        )
    )
    budget = json.loads(tools.get_evaluation_budgets())

    assert result == {
        "evaluation_id": "evaluation-1",
        "status": "success",
    }
    assert gateway.descriptions == ["Try vectorized implementation"]
    assert budget["validation"]["remaining_runs"] == 1
    assert {tool.__name__ for tool in get_tools_from_class(tools)} == {
        "evaluate",
        "get_evaluation_budgets",
    }


@pytest.mark.asyncio
async def test_evaluate_returns_clean_result_when_evaluation_budget_is_spent():
    class ExhaustedGateway(StubEvaluationGateway):
        async def evaluate(self, **_kwargs):
            raise EvaluationBudgetExceeded("evaluation run budget exhausted")

    result = json.loads(
        await EvaluationTools(evaluation=ExhaustedGateway()).evaluate(
            evaluation="validation",
        )
    )

    # A benign, non-terminating signal the agent can absorb — not a raised error.
    assert result["status"] == "evaluation_budget_exhausted"
    assert "budget" in result["message"]


def test_evaluation_tools_report_unmetered_and_require_binding():
    class UnmeteredGateway(StubEvaluationGateway):
        def budgets(self):
            return {"validation": None}

    assert json.loads(
        EvaluationTools(evaluation=UnmeteredGateway()).get_evaluation_budgets()
    ) == {"validation": None}
    with pytest.raises(RuntimeError, match="not bound"):
        EvaluationTools().get_evaluation_budgets()
