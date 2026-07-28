"""Self-contained, read-only reports for durable optimization sessions."""

from __future__ import annotations

import base64
import hashlib
import importlib.resources
import json
import mimetypes
import subprocess
from pathlib import Path
from typing import Any

from vero.candidate import Candidate
from vero.candidate_repository import GitCandidateRepository
from vero.evaluation import EvaluationDatabase
from vero.runtime.events import RuntimeEvent
from vero.runtime.session import SessionManifest
from vero.sidecar.session import HarborSessionManifest
from vero.sidecar.verifier import VerificationResult

_MAX_EMBEDDED_ARTIFACT_BYTES = 5_000_000
_MAX_EMBEDDED_ARTIFACTS_BYTES = 50_000_000
_MAX_DIFF_CHARACTERS = 500_000


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            event = RuntimeEvent.model_validate_json(line)
        except Exception as error:
            raise ValueError(
                f"invalid runtime event on line {line_number}: {error}"
            ) from error
        events.append(event.model_dump(mode="json"))
    return events


def _git_diff(
    repository_path: Path,
    candidate: Candidate,
    parent: Candidate | None,
    *,
    project_subpath: str,
) -> dict[str, Any]:
    if parent is None:
        arguments = [
            "show",
            "--format=",
            "--no-ext-diff",
            "--no-color",
            candidate.version,
        ]
        label = "Initial program"
    else:
        arguments = [
            "diff",
            "--no-ext-diff",
            "--no-color",
            parent.version,
            candidate.version,
        ]
        label = f"Changes from {parent.id}"
    if project_subpath != ".":
        arguments.extend(["--", project_subpath])
    result = subprocess.run(
        ["git", "--git-dir", str(repository_path), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C.UTF-8"},
    )
    if result.returncode != 0:
        return {
            "label": label,
            "text": "",
            "error": result.stderr.strip() or "Git could not render this diff.",
            "truncated": False,
        }
    text = result.stdout
    truncated = len(text) > _MAX_DIFF_CHARACTERS
    if truncated:
        text = text[:_MAX_DIFF_CHARACTERS]
    return {"label": label, "text": text, "error": None, "truncated": truncated}


def _embed_artifact(
    path: Path,
    *,
    media_type: str | None,
    description: str | None,
    relative_path: str,
    remaining_bytes: int,
) -> tuple[dict[str, Any], int]:
    resolved_media_type = media_type or mimetypes.guess_type(path.name)[0]
    artifact: dict[str, Any] = {
        "path": relative_path,
        "media_type": resolved_media_type,
        "description": description,
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "kind": "missing",
        "content": None,
        "omitted_reason": None,
    }
    if not path.is_file():
        artifact["omitted_reason"] = "Artifact file is missing."
        return artifact, 0
    size = path.stat().st_size
    if size > _MAX_EMBEDDED_ARTIFACT_BYTES:
        artifact["kind"] = "omitted"
        artifact["omitted_reason"] = (
            f"Artifact is larger than {_MAX_EMBEDDED_ARTIFACT_BYTES:,} bytes."
        )
        return artifact, 0
    if size > remaining_bytes:
        artifact["kind"] = "omitted"
        artifact["omitted_reason"] = "The report's embedded-artifact limit was reached."
        return artifact, 0

    payload = path.read_bytes()
    if resolved_media_type and (
        resolved_media_type.startswith("image/")
        or resolved_media_type == "application/pdf"
    ):
        artifact["kind"] = (
            "image" if resolved_media_type.startswith("image/") else "binary"
        )
        if artifact["kind"] == "image":
            encoded = base64.b64encode(payload).decode("ascii")
            artifact["content"] = f"data:{resolved_media_type};base64,{encoded}"
        else:
            artifact["omitted_reason"] = "Binary preview is not supported."
    elif (
        resolved_media_type is None
        or resolved_media_type.startswith("text/")
        or resolved_media_type in {"application/json", "application/xml"}
    ):
        artifact["kind"] = "text"
        artifact["content"] = payload.decode("utf-8", errors="replace")
    else:
        artifact["kind"] = "binary"
        artifact["omitted_reason"] = "Binary preview is not supported."
    return artifact, size


def _trace_entries(value: Any) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else [value]
    entries: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            entries.append({"kind": "event", "title": "Event", "body": item})
            continue
        item_type = item.get("type")
        role = item.get("role")
        if item_type == "function_call":
            entries.append(
                {
                    "kind": "tool-call",
                    "title": str(item.get("name") or "Tool call"),
                    "body": item.get("arguments"),
                }
            )
        elif item_type == "function_call_output":
            entries.append(
                {
                    "kind": "tool-result",
                    "title": "Tool result",
                    "body": item.get("output"),
                }
            )
        elif role in {"user", "assistant", "system", "developer"}:
            entries.append(
                {
                    "kind": str(role),
                    "title": str(role).capitalize(),
                    "body": item.get("content"),
                }
            )
        else:
            entries.append(
                {
                    "kind": str(item_type or "event"),
                    "title": str(item_type or "Event").replace("_", " ").title(),
                    "body": item,
                }
            )
    return entries


def _read_traces(session_dir: Path) -> list[dict[str, Any]]:
    root = session_dir / "artifacts" / "agents"
    if not root.is_dir():
        return []
    traces: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_path = directory / "trace.json"
        if directory.name == "producers" or not trace_path.is_file():
            continue
        try:
            raw = json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception as error:
            raw = {"error": f"Could not parse trace: {error}"}
        failure_path = directory / "failure.json"
        failure = None
        if failure_path.is_file():
            try:
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
            except Exception as error:
                failure = {"message": f"Could not parse failure: {error}"}
        traces.append(
            {
                "id": directory.name,
                "entries": _trace_entries(raw),
                "failure": failure,
                "path": str(trace_path.relative_to(session_dir)),
            }
        )
    return traces


def _load_database(session_dir: Path, database_id: str) -> EvaluationDatabase:
    database_path = session_dir / "database.json"
    database = (
        EvaluationDatabase.load_from_file(database_path)
        if database_path.is_file()
        else EvaluationDatabase(id=database_id)
    )
    if database.id != database_id:
        raise ValueError(
            f"evaluation database belongs to {database.id!r}, not {database_id!r}"
        )
    completed = EvaluationDatabase.from_evaluations_dir(
        session_dir / "evaluations", database_id=database_id
    )
    for record in completed.evaluations.values():
        if record.id not in database.evaluations:
            database.add_evaluation(record)
    return database


async def _build_report_data(
    session_dir: Path,
    *,
    manifest: dict[str, Any],
    database: EvaluationDatabase,
    default_trace_id: str | None = None,
) -> dict[str, Any]:
    evaluations = sorted(
        database.evaluations.values(),
        key=lambda record: (record.completed_at, record.id),
    )

    if manifest["candidate_repository_family"] != "git":
        candidates = tuple(
            sorted(
                database.candidates.values(),
                key=lambda value: (value.created_at, value.id),
            )
        )
        repository = None
    else:
        repository = await GitCandidateRepository.open(session_dir / "candidates")
        candidates = repository.list()

    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    traces = _read_traces(session_dir)
    trace_ids = {trace["id"] for trace in traces}
    candidate_data: list[dict[str, Any]] = []
    for candidate in candidates:
        proposal_id = candidate.metadata.get("proposal_id")
        trace_id = (
            hashlib.sha256(str(proposal_id).encode()).hexdigest()[:16]
            if proposal_id is not None
            else default_trace_id
        )
        item = candidate.model_dump(mode="json")
        item["trace_id"] = trace_id if trace_id in trace_ids else None
        if repository is not None:
            item["diff"] = _git_diff(
                repository.repository_path,
                candidate,
                candidate_by_id.get(candidate.parent_id)
                if candidate.parent_id
                else None,
                project_subpath=repository.project_subpath,
            )
        else:
            item["diff"] = {
                "label": "Program changes",
                "text": "",
                "error": "Diffs are unavailable for this candidate repository family.",
                "truncated": False,
            }
        candidate_data.append(item)

    embedded_bytes = 0
    evaluation_data: list[dict[str, Any]] = []
    for step, record in enumerate(evaluations):
        references = list(record.report.artifacts)
        for case in record.report.cases:
            references.extend(case.artifacts)
        seen_paths: set[str] = set()
        artifacts: list[dict[str, Any]] = []
        for reference in references:
            if reference.path in seen_paths:
                continue
            seen_paths.add(reference.path)
            artifact, consumed = _embed_artifact(
                session_dir / "evaluations" / record.id / "artifacts" / reference.path,
                media_type=reference.media_type,
                description=reference.description,
                relative_path=reference.path,
                remaining_bytes=_MAX_EMBEDDED_ARTIFACTS_BYTES - embedded_bytes,
            )
            embedded_bytes += consumed
            artifacts.append(artifact)
        item = record.model_dump(mode="json")
        item["step"] = step
        item["artifacts"] = artifacts
        evaluation_data.append(item)

    events = _read_events(session_dir / "events.jsonl")
    return {
        "schema_version": 1,
        "generated_from": str(session_dir),
        "manifest": manifest,
        "candidates": candidate_data,
        "evaluations": evaluation_data,
        "events": events,
        "traces": traces,
    }


def _harbor_presentation_manifest(
    manifest: HarborSessionManifest,
    database: EvaluationDatabase,
    finalization: VerificationResult | None,
) -> dict[str, Any]:
    selection = manifest.selection
    selected = finalization.candidate if finalization is not None else None
    selection_records = [
        record
        for record in database.evaluations.values()
        if selected is not None
        and selection.backend_id is not None
        and selection.evaluation_set is not None
        and record.request.candidate.id == selected.id
        and record.backend_id == selection.backend_id
        and record.request.evaluation_set == selection.evaluation_set
        and record.objective is not None
        and record.objective.feasible
        and record.objective.value is not None
    ]
    if selection.objective is not None:
        selection_records.sort(key=lambda record: record.id)
        selection_records.sort(
            key=lambda record: record.objective.value,
            reverse=selection.objective.direction == "maximize",
        )
    best_evaluation_id = selection_records[0].id if selection_records else None
    final_evaluation_id = None
    if finalization is not None and finalization.evaluation_ids:
        final_evaluation_id = next(iter(finalization.evaluation_ids.values()))
    completed_at = max(
        (record.completed_at for record in database.evaluations.values()),
        default=manifest.created_at,
    )
    errors = finalization.errors if finalization is not None else {}
    objective = selection.objective or manifest.targets[0].objective
    backend_id = selection.backend_id or manifest.targets[0].backend_id
    selection_set = selection.evaluation_set
    return {
        "schema_version": 1,
        "id": f"{manifest.task_name} · {manifest.id}",
        "status": "failed" if finalization is None or errors else "completed",
        "backend_id": backend_id,
        "candidate_repository_family": manifest.candidate_repository_family,
        "candidate_repository_format_version": (
            manifest.candidate_repository_format_version
        ),
        "evaluation_plan": {
            "selection_evaluation": (
                selection_set.name if selection_set is not None else "selection"
            ),
            "final_evaluation": (
                manifest.targets[0].evaluation_set.name if manifest.targets else None
            ),
        },
        "selection_evaluation_set": (
            selection_set.model_dump(mode="json") if selection_set is not None else None
        ),
        "objective": objective.model_dump(mode="json"),
        "baseline": (
            selection.baseline_candidate.model_dump(mode="json")
            if selection.baseline_candidate is not None
            else None
        ),
        "best_candidate_id": selected.id if selected is not None else None,
        "best_evaluation_id": best_evaluation_id,
        "final_baseline_evaluation_id": None,
        "final_evaluation_id": final_evaluation_id,
        "created_at": manifest.created_at.isoformat(),
        "updated_at": completed_at.isoformat(),
        "failure": (
            {"type": "verification", "message": json.dumps(errors, sort_keys=True)}
            if errors
            else None
        ),
        "metadata": {
            "task_description": manifest.task_description,
            "verification": (
                finalization.model_dump(mode="json")
                if finalization is not None
                else None
            ),
        },
    }


async def build_experiment_report_data(session_dir: Path | str) -> dict[str, Any]:
    """Load a local or Harbor session into the portable report data model."""

    session_dir = Path(session_dir).expanduser().resolve()
    manifest_path = session_dir / "manifest.json"
    if manifest_path.is_file():
        parsed = SessionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest = parsed.model_dump(mode="json")
        manifest["selection_evaluation_set"] = (
            parsed.evaluation_plan.selection.evaluation_set.model_dump(mode="json")
        )
        return await _build_report_data(
            session_dir,
            manifest=manifest,
            database=_load_database(session_dir, parsed.id),
        )

    harbor_path = session_dir / "harbor-session.json"
    if not harbor_path.is_file():
        raise FileNotFoundError(f"session manifest not found: {manifest_path}")
    harbor = HarborSessionManifest.model_validate_json(
        harbor_path.read_text(encoding="utf-8")
    )
    database = _load_database(session_dir, harbor.id)
    finalization_path = session_dir / "harbor-finalization.json"
    finalization = (
        VerificationResult.model_validate_json(
            finalization_path.read_text(encoding="utf-8")
        )
        if finalization_path.is_file()
        else None
    )
    traces = _read_traces(session_dir)
    data = await _build_report_data(
        session_dir,
        manifest=_harbor_presentation_manifest(harbor, database, finalization),
        database=database,
        default_trace_id=traces[0]["id"] if traces else None,
    )
    if not data["events"]:
        data["events"] = [
            {
                "created_at": evaluation["completed_at"],
                "kind": "evaluation.completed",
                "payload": {
                    "evaluation_id": evaluation["id"],
                    "candidate_id": evaluation["request"]["candidate"]["id"],
                    "backend_id": evaluation["backend_id"],
                    "evaluation_set": evaluation["request"]["evaluation_set"],
                    "status": evaluation["report"]["status"],
                    "objective": evaluation["objective"],
                },
            }
            for evaluation in data["evaluations"]
        ]
        if finalization is not None:
            data["events"].append(
                {
                    "created_at": data["manifest"]["updated_at"],
                    "kind": "verification.completed",
                    "payload": finalization.model_dump(mode="json"),
                }
            )
    return data


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


async def generate_experiment_report(
    session_dir: Path | str,
    output: Path | str | None = None,
) -> Path:
    """Generate one portable HTML report without modifying the session."""

    resolved_session = Path(session_dir).expanduser().resolve()
    destination = (
        Path(output).expanduser().resolve()
        if output is not None
        else resolved_session / "experiment.html"
    )
    data = await build_experiment_report_data(resolved_session)
    template = (
        importlib.resources.files("vero")
        .joinpath("templates/report.html")
        .read_text(encoding="utf-8")
    )
    html = template.replace("__VERO_REPORT_DATA__", _safe_json(data))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination

