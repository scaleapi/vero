"""Canonical evaluation inspection tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from vero.evaluation import DisclosureLevel, project_evaluation
from vero.tools.utils import is_tool


@dataclass
class EvaluationViewer:
    """Inspect canonical evaluation summaries, cases, and artifacts."""

    exclude_tools: list[str] = field(default_factory=list)
    database: object | None = None
    excluded_partitions: set[str] = field(default_factory=set)

    def bind(self, session) -> None:
        self.database = getattr(session, "evaluation_database", None)
        split_accesses = getattr(session, "split_accesses", None)
        if split_accesses:
            from vero.core.dataset import get_non_viewable_splits

            self.excluded_partitions = set(get_non_viewable_splits(split_accesses))

    def _record(self, evaluation_id: str):
        if self.database is None:
            raise ValueError("EvaluationViewer requires a canonical evaluation database")
        record = self.database.get_evaluation(evaluation_id)
        if record is None:
            raise KeyError(f"unknown evaluation ID: {evaluation_id}")
        partition = record.request.evaluation_set.partition
        if partition is not None and partition in self.excluded_partitions:
            raise PermissionError(
                f"evaluation {evaluation_id!r} uses non-viewable partition {partition!r}"
            )
        return record

    @is_tool
    def list_evaluations(
        self,
        backend_id: str | None = None,
        evaluation_set: str | None = None,
        limit: int = 20,
    ) -> str:
        """List aggregate evaluation summaries without case payloads."""
        if self.database is None:
            raise ValueError("EvaluationViewer requires a canonical evaluation database")
        records = [
            record
            for record in self.database.get_evaluations(reverse=True)
            if (backend_id is None or record.backend_id == backend_id)
            and (
                evaluation_set is None
                or record.request.evaluation_set.name == evaluation_set
            )
        ][:limit]
        summaries = [
            project_evaluation(record, DisclosureLevel.AGGREGATE).model_dump(mode="json")
            for record in records
        ]
        return json.dumps(summaries, indent=2)

    @is_tool
    def view_evaluation_report(self, evaluation_id: str) -> str:
        """View report metrics, objective, diagnostics, and top-level artifacts."""
        record = self._record(evaluation_id)
        payload = {
            "evaluation_id": record.id,
            "candidate_commit": record.request.candidate.commit,
            "backend_id": record.backend_id,
            "evaluation_set": record.request.evaluation_set.model_dump(mode="json"),
            "status": record.report.status.value,
            "metrics": record.report.metrics,
            "objective": record.objective.model_dump(mode="json")
            if record.objective is not None
            else None,
            "diagnostics": [
                diagnostic.model_dump(mode="json")
                for diagnostic in record.report.diagnostics
            ],
            "artifacts": [
                artifact.model_dump(mode="json")
                for artifact in record.report.artifacts
            ],
            "error": record.report.error,
        }
        return json.dumps(payload, indent=2)

    @is_tool
    def view_evaluation_cases(self, evaluation_id: str) -> str:
        """View case results for an evaluation when its partition is viewable."""
        record = self._record(evaluation_id)
        return json.dumps(
            [case.model_dump(mode="json") for case in record.report.cases],
            indent=2,
        )

    @is_tool
    def view_evaluation_artifacts(self, evaluation_id: str) -> str:
        """List top-level and per-case evaluation artifacts."""
        record = self._record(evaluation_id)
        payload = {
            "report": [
                artifact.model_dump(mode="json")
                for artifact in record.report.artifacts
            ],
            "cases": {
                case.case_id: [
                    artifact.model_dump(mode="json")
                    for artifact in case.artifacts
                ]
                for case in record.report.cases
                if case.artifacts
            },
        }
        return json.dumps(payload, indent=2)
