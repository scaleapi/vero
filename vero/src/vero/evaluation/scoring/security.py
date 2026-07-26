"""Secret-safe normalization for persisted evaluation content."""

from __future__ import annotations

from typing import Any, Iterable

from vero.evaluation.models import EvaluationReport


def _usable_secrets(secrets: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {secret for secret in secrets if secret and len(secret) >= 4},
            key=len,
            reverse=True,
        )
    )


def sanitize_text(text: str, secrets: Iterable[str | None]) -> str:
    sanitized = text
    for secret in _usable_secrets(secrets):
        sanitized = sanitized.replace(secret, "[REDACTED]")
    return sanitized


def _sanitize_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, secrets)
    if isinstance(value, list):
        return [_sanitize_value(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_value(item, secrets) for key, item in value.items()}
    return value


def sanitize_evaluation_report(
    report: EvaluationReport,
    secrets: Iterable[str | None],
) -> EvaluationReport:
    """Redact free-form report fields while retaining structural identifiers."""
    secret_values = _usable_secrets(secrets)
    if not secret_values:
        return report

    diagnostics = [
        diagnostic.model_copy(
            update={
                "message": sanitize_text(diagnostic.message, secret_values),
                "metadata": _sanitize_value(diagnostic.metadata, secret_values),
            }
        )
        for diagnostic in report.diagnostics
    ]
    cases = []
    for case in report.cases:
        errors = [
            error.model_copy(
                update={
                    "message": sanitize_text(error.message, secret_values),
                    "metadata": _sanitize_value(error.metadata, secret_values),
                }
            )
            for error in case.errors
        ]
        artifacts = [
            artifact.model_copy(
                update={
                    "description": sanitize_text(artifact.description, secret_values)
                    if artifact.description is not None
                    else None
                }
            )
            for artifact in case.artifacts
        ]
        cases.append(
            case.model_copy(
                update={
                    "input": _sanitize_value(case.input, secret_values),
                    "output": _sanitize_value(case.output, secret_values),
                    "feedback": sanitize_text(case.feedback, secret_values)
                    if case.feedback is not None
                    else None,
                    "errors": errors,
                    "execution_trace": _sanitize_value(case.execution_trace, secret_values),
                    "evaluation_trace": _sanitize_value(case.evaluation_trace, secret_values),
                    "metadata": _sanitize_value(case.metadata, secret_values),
                    "artifacts": artifacts,
                }
            )
        )
    artifacts = [
        artifact.model_copy(
            update={
                "description": sanitize_text(artifact.description, secret_values)
                if artifact.description is not None
                else None
            }
        )
        for artifact in report.artifacts
    ]
    return report.model_copy(
        update={
            "diagnostics": diagnostics,
            "cases": cases,
            "artifacts": artifacts,
        }
    )
