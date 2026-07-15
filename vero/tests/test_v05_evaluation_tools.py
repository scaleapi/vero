from __future__ import annotations

import json

import pytest

from vero.evaluation import (
    EvaluationAcknowledgement,
    EvaluationBudget,
    EvaluationSet,
    EvaluationStatus,
)
from vero.tools.evaluation import EvaluationTools
from vero.tools.utils import get_tools_from_class


class StubEvaluationGateway:
    def __init__(self):
        self.descriptions: list[str] = []
        self.remaining_runs = 2

    async def evaluate_current(self, *, description="Evaluate agent checkpoint"):
        self.descriptions.append(description)
        self.remaining_runs -= 1
        return EvaluationAcknowledgement(
            evaluation_id="evaluation-1",
            status=EvaluationStatus.SUCCESS,
        )

    def budget(self):
        return EvaluationBudget(
            backend_id="command",
            evaluation_set_key=EvaluationSet(name="test").budget_key("command"),
            total_runs=2,
            remaining_runs=self.remaining_runs,
        )


@pytest.mark.asyncio
async def test_evaluation_tools_expose_only_scoped_feedback_and_budget():
    gateway = StubEvaluationGateway()
    tools = EvaluationTools(evaluation=gateway)

    result = json.loads(
        await tools.evaluate_current(description="Try vectorized implementation")
    )
    budget = json.loads(tools.get_evaluation_budget())

    assert result == {
        "evaluation_id": "evaluation-1",
        "status": "success",
    }
    assert gateway.descriptions == ["Try vectorized implementation"]
    assert budget["remaining_runs"] == 1
    assert {tool.__name__ for tool in get_tools_from_class(tools)} == {
        "evaluate_current",
        "get_evaluation_budget",
    }


def test_evaluation_tools_report_unmetered_and_require_binding():
    class UnmeteredGateway(StubEvaluationGateway):
        def budget(self):
            return None

    assert "not metered" in EvaluationTools(
        evaluation=UnmeteredGateway()
    ).get_evaluation_budget()
    with pytest.raises(RuntimeError, match="not bound"):
        EvaluationTools().get_evaluation_budget()
