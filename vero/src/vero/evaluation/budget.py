"""Durable, backend-qualified evaluation budget reservations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from vero.evaluation.exceptions import EvaluationBudgetExceeded
from vero.evaluation.models import (
    EvaluationBudget,
    EvaluationCost,
    EvaluationPrincipal,
    EvaluationSet,
)
from vero.evaluation.persistence import _atomic_write_json


class BudgetLedger:
    """Atomically meter completed evaluations against configured budgets."""

    schema_version = 1

    def __init__(
        self,
        budgets: list[EvaluationBudget] | None = None,
        *,
        path: Path | None = None,
    ):
        self.path = path
        self._lock = asyncio.Lock()
        self._budgets: dict[tuple[str, str, EvaluationPrincipal], EvaluationBudget] = {}
        for budget in budgets or []:
            self.add(budget)

    @property
    def budgets(self) -> list[EvaluationBudget]:
        return list(self._budgets.values())

    def add(self, budget: EvaluationBudget) -> None:
        key = (budget.backend_id, budget.evaluation_set_key, budget.principal)
        if key in self._budgets:
            raise ValueError(f"duplicate evaluation budget for {key!r}")
        updates: dict[str, int] = {}
        if budget.total_runs is not None and budget.remaining_runs is None:
            updates["remaining_runs"] = budget.total_runs
        if budget.total_cases is not None and budget.remaining_cases is None:
            updates["remaining_cases"] = budget.total_cases
        self._budgets[key] = budget.model_copy(update=updates)

    def get(
        self,
        backend_id: str,
        evaluation_set: EvaluationSet,
        principal: EvaluationPrincipal = EvaluationPrincipal.AGENT,
    ) -> EvaluationBudget | None:
        return self._budgets.get(
            (backend_id, evaluation_set.budget_key(backend_id), principal)
        )

    async def reserve(
        self,
        backend_id: str,
        evaluation_set: EvaluationSet,
        cost: EvaluationCost,
        principal: EvaluationPrincipal = EvaluationPrincipal.AGENT,
    ) -> EvaluationBudget | None:
        key = (backend_id, evaluation_set.budget_key(backend_id), principal)
        async with self._lock:
            budget = self._budgets.get(key)
            if budget is None:
                return None
            if cost.cases is None and budget.remaining_cases is not None:
                raise EvaluationBudgetExceeded(
                    "evaluation case cost is unknown but the budget has a case limit"
                )
            if budget.remaining_runs is not None and cost.runs > budget.remaining_runs:
                raise EvaluationBudgetExceeded("evaluation run budget exhausted")
            if (
                budget.remaining_cases is not None
                and cost.cases is not None
                and cost.cases > budget.remaining_cases
            ):
                raise EvaluationBudgetExceeded("evaluation case budget exhausted")

            updates: dict[str, int] = {}
            if budget.remaining_runs is not None:
                updates["remaining_runs"] = budget.remaining_runs - cost.runs
            if budget.remaining_cases is not None and cost.cases is not None:
                updates["remaining_cases"] = budget.remaining_cases - cost.cases
            updated = budget.model_copy(update=updates)
            snapshot = dict(self._budgets)
            snapshot[key] = updated
            if self.path is not None:
                write = asyncio.create_task(
                    asyncio.to_thread(
                        _atomic_write_json,
                        self.path,
                        self._serialize(snapshot),
                    )
                )
                cancellation: asyncio.CancelledError | None = None
                while not write.done():
                    try:
                        await asyncio.shield(write)
                    except asyncio.CancelledError as error:
                        cancellation = error
                write.result()
                self._budgets = snapshot
                if cancellation is not None:
                    raise cancellation
                return updated
            self._budgets = snapshot
            return updated

    async def refund(
        self,
        backend_id: str,
        evaluation_set: EvaluationSet,
        cost: EvaluationCost,
        principal: EvaluationPrincipal = EvaluationPrincipal.AGENT,
    ) -> EvaluationBudget | None:
        """Undo a reservation when execution produced no usable result."""

        key = (backend_id, evaluation_set.budget_key(backend_id), principal)
        async with self._lock:
            budget = self._budgets.get(key)
            if budget is None:
                return None
            updates: dict[str, int] = {}
            if budget.remaining_runs is not None:
                restored_runs = budget.remaining_runs + cost.runs
                updates["remaining_runs"] = (
                    min(restored_runs, budget.total_runs)
                    if budget.total_runs is not None
                    else restored_runs
                )
            if budget.remaining_cases is not None and cost.cases is not None:
                restored_cases = budget.remaining_cases + cost.cases
                updates["remaining_cases"] = (
                    min(restored_cases, budget.total_cases)
                    if budget.total_cases is not None
                    else restored_cases
                )
            updated = budget.model_copy(update=updates)
            snapshot = dict(self._budgets)
            snapshot[key] = updated
            if self.path is not None:
                write = asyncio.create_task(
                    asyncio.to_thread(
                        _atomic_write_json,
                        self.path,
                        self._serialize(snapshot),
                    )
                )
                cancellation: asyncio.CancelledError | None = None
                while not write.done():
                    try:
                        await asyncio.shield(write)
                    except asyncio.CancelledError as error:
                        cancellation = error
                write.result()
                self._budgets = snapshot
                if cancellation is not None:
                    raise cancellation
                return updated
            self._budgets = snapshot
            return updated

    def _serialize(
        self,
        budgets: dict[tuple[str, str, EvaluationPrincipal], EvaluationBudget]
        | None = None,
    ) -> dict[str, Any]:
        values = budgets if budgets is not None else self._budgets
        return {
            "schema_version": self.schema_version,
            "budgets": [
                budget.model_dump(mode="json") for _, budget in sorted(values.items())
            ],
        }

    def save(self) -> None:
        if self.path is None:
            raise ValueError("budget ledger has no persistence path")
        _atomic_write_json(self.path, self._serialize())

    @classmethod
    def load(cls, path: Path) -> BudgetLedger:
        if not path.exists():
            return cls(path=path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != cls.schema_version:
                raise ValueError("unsupported budget ledger schema")
            budgets = [
                EvaluationBudget.model_validate(value)
                for value in payload.get("budgets", [])
            ]
        except Exception as error:
            raise ValueError(
                f"invalid durable budget ledger {path}: {error}"
            ) from error
        return cls(budgets, path=path)
