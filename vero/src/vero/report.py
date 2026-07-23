"""Self-contained, read-only reports for durable optimization sessions."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import subprocess
from pathlib import Path
from typing import Any

from vero.candidate import Candidate
from vero.candidate_repository import GitCandidateRepository
from vero.evaluation import EvaluationDatabase
from vero.harbor.session import HarborSessionManifest
from vero.harbor.verifier import VerificationResult
from vero.runtime.events import RuntimeEvent
from vero.runtime.session import SessionManifest


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
    html = _REPORT_HTML.replace("__VERO_REPORT_DATA__", _safe_json(data))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


_REPORT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
  <title>VeRO experiment report</title>
  <style>
    :root { color-scheme: dark; --bg:#0b0d12; --panel:#131720; --panel2:#191e29; --line:#2a3140; --text:#eef1f7; --muted:#9ca6b8; --accent:#87a6ff; --green:#63d49a; --orange:#f5b75b; --red:#ff7d87; --purple:#c39bff; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 20% -10%,#202942 0,transparent 34%),var(--bg); color:var(--text); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    header,main { width:min(1500px,calc(100% - 40px)); margin:auto; }
    header { padding:42px 0 24px; display:flex; gap:24px; justify-content:space-between; align-items:flex-end; }
    h1 { margin:0; font-size:clamp(30px,4vw,52px); letter-spacing:-.04em; }
    h2 { margin:0 0 14px; font-size:18px; }
    h3 { margin:0 0 8px; font-size:15px; }
    p { margin:0; }
    .eyebrow { color:var(--accent); text-transform:uppercase; letter-spacing:.15em; font-size:11px; font-weight:700; }
    .subtitle,.muted { color:var(--muted); }
    .subtitle { margin-top:8px; font-size:16px; }
    .pill { display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:6px 10px; color:var(--muted); background:#10141c; }
    .stats { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:18px; }
    .stat,.panel { border:1px solid var(--line); background:linear-gradient(180deg,rgba(25,30,41,.96),rgba(17,21,29,.96)); border-radius:14px; box-shadow:0 14px 40px rgba(0,0,0,.16); }
    .stat { padding:15px 17px; }
    .stat span { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.09em; }
    .stat strong { display:block; margin-top:4px; font-size:22px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:18px; }
    .panel { padding:18px; min-width:0; }
    .wide { grid-column:1/-1; }
    svg.chart { display:block; width:100%; height:320px; border-radius:10px; background:#0e1118; }
    .legend { display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; color:var(--muted); font-size:12px; }
    .dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px; }
    .detail-grid { display:grid; grid-template-columns:minmax(260px,.35fr) minmax(0,1fr); gap:18px; }
    .candidate-list { max-height:760px; overflow:auto; padding-right:4px; }
    button { font:inherit; color:inherit; }
    .candidate { width:100%; display:block; text-align:left; border:1px solid var(--line); border-radius:10px; background:#10141b; padding:12px; margin-bottom:8px; cursor:pointer; }
    .candidate:hover,.candidate.active { border-color:var(--accent); background:#171e2d; }
    .candidate .top { display:flex; justify-content:space-between; gap:8px; }
    .candidate .score { color:var(--green); font-variant-numeric:tabular-nums; }
    .candidate small { color:var(--muted); display:block; margin-top:5px; }
    .tags { display:flex; flex-wrap:wrap; gap:5px; margin-top:7px; }
    .tag { font-size:10px; border-radius:999px; padding:2px 7px; background:#242c3b; color:#c9d2e2; }
    .candidate-head { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; margin-bottom:16px; }
    code,.mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
    .identifier { overflow-wrap:anywhere; color:#cdd6e7; }
    .description { font-size:16px; margin:8px 0; }
    .evaluations { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:9px; margin:12px 0 18px; }
    .evaluation { border:1px solid var(--line); background:#10141b; border-radius:9px; padding:11px; cursor:pointer; }
    .evaluation:hover,.evaluation.active { border-color:var(--purple); }
    .evaluation strong { display:block; font-size:17px; }
    .tabs { display:flex; gap:6px; margin:12px 0; border-bottom:1px solid var(--line); padding-bottom:8px; }
    .tab { border:0; background:transparent; color:var(--muted); padding:7px 10px; border-radius:7px; cursor:pointer; }
    .tab.active { color:var(--text); background:#262e3d; }
    pre { margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
    .codebox { max-height:580px; overflow:auto; border:1px solid var(--line); border-radius:9px; background:#090c11; padding:13px; }
    .artifact-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; }
    .artifact { min-width:0; border:1px solid var(--line); border-radius:10px; background:#0e1219; padding:10px; }
    .artifact img { display:block; width:100%; max-height:430px; object-fit:contain; background:white; border-radius:7px; margin-top:8px; }
    .artifact pre { max-height:320px; overflow:auto; margin-top:8px; }
    .trace { max-height:700px; overflow:auto; display:grid; gap:8px; }
    .trace-entry { border-left:3px solid var(--line); background:#0e1219; border-radius:6px; padding:10px 12px; }
    .trace-entry.user { border-color:var(--accent); } .trace-entry.assistant { border-color:var(--green); } .trace-entry.tool-call { border-color:var(--orange); } .trace-entry.tool-result { border-color:var(--purple); }
    .trace-entry h4 { margin:0 0 5px; color:#cbd4e5; }
    .timeline { max-height:480px; overflow:auto; }
    .event { display:grid; grid-template-columns:165px 180px 1fr; gap:12px; padding:9px 2px; border-bottom:1px solid var(--line); }
    .notice { border:1px solid #604d27; background:#241e13; color:#e9ce98; padding:10px 12px; border-radius:9px; margin-bottom:18px; }
    .empty { color:var(--muted); padding:24px 0; text-align:center; }
    footer { color:var(--muted); padding:18px 0 40px; }
    @media(max-width:900px) { header { align-items:flex-start; flex-direction:column; } .stats { grid-template-columns:repeat(2,1fr); } .grid,.detail-grid { grid-template-columns:1fr; } .event { grid-template-columns:1fr; gap:2px; } }
  </style>
</head>
<body>
  <header><div><div class="eyebrow">VeRO experiment report</div><h1 id="title">Optimization session</h1><p class="subtitle" id="subtitle"></p></div><span class="pill" id="status"></span></header>
  <main>
    <div class="notice">This portable report can contain source diffs, evaluation artifacts, prompts, and agent tool output. Treat it as sensitive experiment data.</div>
    <section class="stats" id="stats"></section>
    <section class="grid">
      <div class="panel"><h2>Score trajectories by split</h2><svg class="chart" id="score-chart" role="img" aria-label="Objective scores separated by evaluation split"></svg><div class="legend" id="score-legend"></div></div>
      <div class="panel"><h2>Candidate lineage</h2><svg class="chart" id="lineage" role="img" aria-label="Candidate lineage graph"></svg><div class="legend"><span>Click a node to inspect it</span><span><i class="dot" style="background:var(--accent)"></i>baseline</span><span><i class="dot" style="background:var(--green)"></i>best</span></div></div>
    </section>
    <section class="panel wide"><h2>Candidates</h2><div class="detail-grid"><div class="candidate-list" id="candidate-list"></div><div id="candidate-detail"></div></div></section>
    <section class="grid" style="margin-top:18px">
      <div class="panel"><h2>Producer traces</h2><div id="trace-picker" class="tabs"></div><div id="trace-detail"></div></div>
      <div class="panel"><h2>Event timeline</h2><div class="timeline" id="timeline"></div></div>
    </section>
    <footer id="footer"></footer>
  </main>
  <script id="report-data" type="application/json">__VERO_REPORT_DATA__</script>
  <script>
  (() => {
    'use strict';
    const data = JSON.parse(document.getElementById('report-data').textContent);
    const manifest = data.manifest;
    const candidates = data.candidates;
    const evaluations = data.evaluations;
    const byCandidate = new Map(candidates.map(c => [c.id,c]));
    const evalsByCandidate = new Map();
    evaluations.forEach(e => { const id=e.request.candidate.id; if(!evalsByCandidate.has(id)) evalsByCandidate.set(id,[]); evalsByCandidate.get(id).push(e); });
    let selectedCandidateId = manifest.best_candidate_id || (manifest.baseline && manifest.baseline.id) || (candidates[0] && candidates[0].id);
    let selectedEvaluationId = manifest.best_evaluation_id || null;
    const selectionTarget = manifest.selection_evaluation_set || null;
    const ns='http://www.w3.org/2000/svg';
    const node=(tag,cls,text) => { const value=document.createElement(tag); if(cls)value.className=cls; if(text!==undefined)value.textContent=String(text); return value; };
    const svg=(tag,attrs={}) => { const value=document.createElementNS(ns,tag); Object.entries(attrs).forEach(([k,v])=>value.setAttribute(k,String(v))); return value; };
    const short=value => value ? String(value).slice(0,12) : '—';
    const score=e => e && e.objective ? e.objective.value : null;
    const number=value => typeof value==='number' ? value.toLocaleString(undefined,{maximumFractionDigits:6}) : 'n/a';
    const selectionKey=selection => { if(!selection||selection.kind==='all')return 'all'; if(selection.kind==='range')return `range:${selection.start||0}:${selection.stop}`; if(selection.kind==='ids')return `ids:${JSON.stringify(selection.ids||[])}`; return JSON.stringify(selection); };
    const evaluationSetKey=set => `${set.name}\u0000${set.partition||''}\u0000${selectionKey(set.selection)}`;
    const splitKey=set => `${set.name}\u0000${set.partition||''}`;
    const splitLabel=set => set.partition ? `${set.name} / ${set.partition}` : set.name;
    const subsetLabel=selection => { if(!selection||selection.kind==='all')return 'all cases'; if(selection.kind==='range')return `range [${selection.start||0}, ${selection.stop})`; if(selection.kind==='ids')return `${(selection.ids||[]).length} named cases`; return selection.kind||'custom subset'; };
    const evaluationSetLabel=set => `${splitLabel(set)} · ${subsetLabel(set.selection)}`;
    const sameEvaluationSet=(left,right) => !!left&&!!right&&evaluationSetKey(left)===evaluationSetKey(right);
    const primaryEval=id => { const values=evalsByCandidate.get(id)||[]; const exact=values.find(e=>e.id===manifest.best_evaluation_id); if(exact)return exact; const selection=values.filter(e=>selectionTarget ? e.request.evaluation_set.name===selectionTarget.name&&e.request.evaluation_set.partition===selectionTarget.partition : e.request.evaluation_set.name===manifest.evaluation_plan.selection_evaluation); const feasible=selection.filter(e=>e.objective&&e.objective.feasible&&typeof e.objective.value==='number'); if(feasible.length){ const sign=manifest.objective.direction==='maximize' ? -1 : 1; return feasible.sort((a,b)=>sign*(a.objective.value-b.objective.value))[0]; } return selection.at(-1)||values.find(e=>e.id===manifest.final_evaluation_id)||values.at(-1); };

    document.getElementById('title').textContent = manifest.id;
    const bestEval=evaluations.find(e=>e.id===manifest.best_evaluation_id);
    document.getElementById('subtitle').textContent = `${manifest.objective.direction} ${manifest.objective.selector.metric} · ${bestEval?evaluationSetLabel(bestEval.request.evaluation_set):manifest.backend_id}`;
    document.getElementById('status').textContent = manifest.status;
    const baselineEvals=manifest.baseline?(evalsByCandidate.get(manifest.baseline.id)||[]):[];
    const preferredBaseline=baselineEvals.find(e=>e.id===manifest.final_baseline_evaluation_id);
    const baselineEval=preferredBaseline&&bestEval&&sameEvaluationSet(preferredBaseline.request.evaluation_set,bestEval.request.evaluation_set)?preferredBaseline:(bestEval?baselineEvals.filter(e=>sameEvaluationSet(e.request.evaluation_set,bestEval.request.evaluation_set)).at(-1):null);
    const comparable=score(bestEval)!==null&&score(baselineEval)!==null;
    const statValues=[['Candidates',candidates.length],['Evaluations',evaluations.length],['Comparable baseline',number(score(baselineEval))],['Selected score',number(score(bestEval))],['Comparable delta',comparable?number(score(bestEval)-score(baselineEval)):'n/a']];
    const stats=document.getElementById('stats'); statValues.forEach(([label,value])=>{const box=node('div','stat');box.append(node('span','',label),node('strong','',value));stats.append(box);});

    function drawScoreChart(){const root=document.getElementById('score-chart'),legend=document.getElementById('score-legend');root.replaceChildren();legend.replaceChildren();const points=evaluations.map((evaluation,index)=>({evaluation,index})).filter(point=>typeof score(point.evaluation)==='number');if(!points.length){root.append(svg('text',{x:20,y:40,fill:'#9ca6b8'})).textContent='No objective values';return;}const splits=new Map();points.forEach(point=>{const set=point.evaluation.request.evaluation_set,key=splitKey(set);if(!splits.has(key))splits.set(key,{set,points:[]});splits.get(key).points.push(point);});const palette=['#63d49a','#87a6ff','#f5b75b','#c39bff','#55c7d9','#f08bb4'];const W=760,rowH=132,p={l:118,r:22,t:18,b:38},H=Math.max(320,p.t+p.b+splits.size*rowH);root.setAttribute('viewBox',`0 0 ${W} ${H}`);root.style.height=`${Math.min(640,H)}px`;const vals=points.map(point=>score(point.evaluation)),min=Math.min(...vals),max=Math.max(...vals),pad=(max-min||1)*.12,lo=min-pad,hi=max+pad;const x=index=>p.l+index*Math.max(1,(W-p.l-p.r)/Math.max(1,evaluations.length-1));[...splits.values()].forEach((group,row)=>{const color=palette[row%palette.length],top=p.t+row*rowH,bottom=top+rowH-18,y=value=>top+18+(hi-value)/(hi-lo)*(bottom-top-30);root.append(svg('rect',{x:p.l-8,y:top,width:W-p.l-p.r+16,height:rowH-12,rx:8,fill:row%2?'#0c1017':'#0f131b'}));const label=svg('text',{x:p.l-14,y:top+20,fill:color,'text-anchor':'end','font-size':11,'font-weight':700});label.textContent=splitLabel(group.set);root.append(label);const subsets=new Set(group.points.map(point=>evaluationSetKey(point.evaluation.request.evaluation_set)));const detail=svg('text',{x:p.l-14,y:top+37,fill:'#9ca6b8','text-anchor':'end','font-size':9});detail.textContent=`${group.points.length} eval${group.points.length===1?'':'s'} · ${subsets.size} subset${subsets.size===1?'':'s'}`;root.append(detail);for(let grid=0;grid<3;grid++){const value=hi-grid*(hi-lo)/2,yy=y(value);root.append(svg('line',{x1:p.l,y1:yy,x2:W-p.r,y2:yy,stroke:'#2a3140'}));const tick=svg('text',{x:p.l-10,y:yy+4,fill:'#788397','text-anchor':'end','font-size':9});tick.textContent=number(value);root.append(tick);}const exactSets=new Map();group.points.forEach(point=>{const key=evaluationSetKey(point.evaluation.request.evaluation_set);if(!exactSets.has(key))exactSets.set(key,[]);exactSets.get(key).push(point);});exactSets.forEach(series=>{if(series.length<2)return;root.append(svg('path',{d:series.map((point,i)=>`${i?'L':'M'} ${x(point.index)} ${y(score(point.evaluation))}`).join(' '),fill:'none',stroke:color,'stroke-opacity':.65,'stroke-width':2}));});group.points.forEach(point=>{const e=point.evaluation,final=e.id===manifest.final_evaluation_id||e.id===manifest.final_baseline_evaluation_id,good=e.objective&&e.objective.feasible&&e.report.status==='success',circle=svg('circle',{cx:x(point.index),cy:y(score(e)),r:final?7:5,fill:good?color:'#ff7d87',stroke:final?'#c39bff':'#0b0d12','stroke-width':final?3:2,tabindex:0});circle.style.cursor='pointer';circle.addEventListener('click',()=>selectCandidate(e.request.candidate.id,e.id));const title=svg('title');title.textContent=`${evaluationSetLabel(e.request.evaluation_set)}: ${number(score(e))} · ${e.request.candidate.id}`;circle.append(title);root.append(circle);});const item=node('span');const dot=node('i','dot');dot.style.background=color;item.append(dot,document.createTextNode(splitLabel(group.set)));legend.append(item);});const note=node('span','', 'Lines connect exact case selections only');legend.append(note);const failure=node('span');const failureDot=node('i','dot');failureDot.style.background='var(--red)';failure.append(failureDot,document.createTextNode('failed or infeasible'));legend.append(failure);const final=node('span');const finalDot=node('i','dot');finalDot.style.background='var(--purple)';final.append(finalDot,document.createTextNode('final measurement'));legend.append(final);}

    function depths(){ const memo=new Map(); const visit=(id,seen=new Set())=>{if(memo.has(id))return memo.get(id);if(seen.has(id))return 0;seen.add(id);const c=byCandidate.get(id);const d=c&&c.parent_id&&byCandidate.has(c.parent_id)?visit(c.parent_id,seen)+1:0;memo.set(id,d);return d;};candidates.forEach(c=>visit(c.id));return memo; }
    function drawLineage(){const root=document.getElementById('lineage');root.replaceChildren();if(!candidates.length)return;const ds=depths(),groups=new Map();candidates.forEach(c=>{const d=ds.get(c.id);if(!groups.has(d))groups.set(d,[]);groups.get(d).push(c);});const maxDepth=Math.max(...groups.keys()),maxGroup=Math.max(...[...groups.values()].map(g=>g.length));const W=Math.max(760,(maxDepth+1)*180+90),H=Math.max(320,maxGroup*82+60);root.setAttribute('viewBox',`0 0 ${W} ${H}`);const positions=new Map();[...groups.entries()].forEach(([d,group])=>group.forEach((c,i)=>positions.set(c.id,{x:55+d*180,y:40+(i+1)*H/(group.length+1)})));candidates.forEach(c=>{if(c.parent_id&&positions.has(c.parent_id)){const a=positions.get(c.parent_id),b=positions.get(c.id);root.append(svg('path',{d:`M ${a.x+55} ${a.y} C ${a.x+105} ${a.y}, ${b.x-50} ${b.y}, ${b.x} ${b.y}`,fill:'none',stroke:'#3a4355','stroke-width':2}));}});candidates.forEach(c=>{const p=positions.get(c.id),g=svg('g',{tabindex:0});g.style.cursor='pointer';const isBase=manifest.baseline&&c.id===manifest.baseline.id,isBest=c.id===manifest.best_candidate_id;g.append(svg('rect',{x:p.x,y:p.y-20,width:110,height:40,rx:9,fill:c.id===selectedCandidateId?'#253453':'#151b25',stroke:isBest?'#63d49a':isBase?'#87a6ff':'#394255','stroke-width':isBest||isBase?3:1.5}));const t=svg('text',{x:p.x+55,y:p.y+4,fill:'#eef1f7','text-anchor':'middle','font-size':11});t.textContent=short(c.id);g.append(t);g.addEventListener('click',()=>selectCandidate(c.id));root.append(g);}); }

    function renderCandidateList(){const root=document.getElementById('candidate-list');root.replaceChildren();candidates.forEach((c,index)=>{const e=primaryEval(c.id),button=node('button',`candidate${c.id===selectedCandidateId?' active':''}`);const top=node('div','top');top.append(node('strong','',c.description?`Candidate ${index}`:(index===0?'Baseline':`Candidate ${index}`)),node('span','score',number(score(e))));button.append(top,node('small','mono',short(c.id)));const tags=node('div','tags');if(manifest.baseline&&c.id===manifest.baseline.id)tags.append(node('span','tag','baseline'));if(c.id===manifest.best_candidate_id)tags.append(node('span','tag','best'));if(e)tags.append(node('span','tag',evaluationSetLabel(e.request.evaluation_set)));if(c.trace_id)tags.append(node('span','tag','trace'));button.append(tags);button.addEventListener('click',()=>selectCandidate(c.id));root.append(button);});}

    function renderArtifacts(evaluation){const root=node('div','artifact-grid');if(!evaluation||!evaluation.artifacts.length){root.append(node('div','empty','No evaluation artifacts.'));return root;}evaluation.artifacts.forEach(a=>{const card=node('article','artifact');card.append(node('h3','',a.description||a.path),node('div','muted mono',`${a.path} · ${a.media_type||'unknown'} · ${a.size===null?'missing':a.size.toLocaleString()+' bytes'}`));if(a.kind==='image'&&a.content){const image=node('img');image.src=a.content;image.alt=a.description||a.path;card.append(image);}else if(a.kind==='text'&&a.content!==null){card.append(node('pre','',a.content));}else card.append(node('p','muted',a.omitted_reason||'No preview available.'));root.append(card);});return root;}

    function renderCandidateDetail(){const root=document.getElementById('candidate-detail');root.replaceChildren();const c=byCandidate.get(selectedCandidateId);if(!c){root.append(node('div','empty','No candidate selected.'));return;}const head=node('div','candidate-head'),left=node('div');left.append(node('div','eyebrow',c.id===manifest.best_candidate_id?'Best candidate':manifest.baseline&&c.id===manifest.baseline.id?'Baseline':'Candidate'),node('h2','identifier mono',c.id));if(c.description)left.append(node('p','description',c.description));left.append(node('p','muted',`Version ${short(c.version)} · parent ${short(c.parent_id)} · ${new Date(c.created_at).toLocaleString()}`));head.append(left);root.append(head);const values=evalsByCandidate.get(c.id)||[];const cards=node('div','evaluations');values.forEach(e=>{const card=node('button',`evaluation${e.id===selectedEvaluationId?' active':''}`);card.append(node('span','eyebrow',evaluationSetLabel(e.request.evaluation_set)),node('strong','',number(score(e))),node('small','muted',`${e.report.status} · ${e.principal} · ${new Date(e.completed_at).toLocaleString()}`));card.addEventListener('click',()=>{selectedEvaluationId=e.id;renderCandidateDetail();});cards.append(card);});root.append(cards);let selected=values.find(e=>e.id===selectedEvaluationId)||primaryEval(c.id)||values[0];if(selected)selectedEvaluationId=selected.id;const tabs=node('div','tabs');const content=node('div');const options=[['diff','Program diff'],['artifacts','Artifacts'],['evaluation','Evaluation JSON'],['metadata','Candidate metadata']];let active='diff';const show=kind=>{active=kind;[...tabs.children].forEach(b=>b.classList.toggle('active',b.dataset.kind===kind));content.replaceChildren();if(kind==='diff'){if(c.diff.error)content.append(node('p','muted',c.diff.error));content.append(node('h3','',c.diff.label));content.append(node('pre','codebox',c.diff.text||'No textual changes.'));if(c.diff.truncated)content.append(node('p','muted','Diff was truncated in this report.'));}else if(kind==='artifacts')content.append(renderArtifacts(selected));else if(kind==='evaluation')content.append(node('pre','codebox',selected?JSON.stringify(selected,null,2):'No evaluation.'));else content.append(node('pre','codebox',JSON.stringify(c,null,2)));};options.forEach(([kind,label])=>{const b=node('button',`tab${kind===active?' active':''}`,label);b.dataset.kind=kind;b.addEventListener('click',()=>show(kind));tabs.append(b);});root.append(tabs,content);show(active);if(c.trace_id)selectTrace(c.trace_id,false);}

    function selectCandidate(id,evaluationId=null){selectedCandidateId=id;if(evaluationId)selectedEvaluationId=evaluationId;else{const e=primaryEval(id);selectedEvaluationId=e&&e.id;}renderCandidateList();renderCandidateDetail();drawLineage();}

    function renderTrace(trace){const root=document.getElementById('trace-detail');root.replaceChildren();if(!trace){root.append(node('div','empty','No producer traces were persisted.'));return;}if(trace.failure){const warning=node('div','notice');warning.textContent=`Failed attempt: ${trace.failure.message||JSON.stringify(trace.failure)}`;root.append(warning);}const transcript=node('div','trace');trace.entries.forEach(entry=>{const card=node('article',`trace-entry ${entry.kind}`);card.append(node('h4','',entry.title));const body=typeof entry.body==='string'?entry.body:JSON.stringify(entry.body,null,2);card.append(node('pre','',body===undefined?'':body));transcript.append(card);});root.append(transcript);}
    function selectTrace(id,rerender=true){const trace=data.traces.find(t=>t.id===id);if(!trace)return;if(rerender){[...document.querySelectorAll('#trace-picker .tab')].forEach(b=>b.classList.toggle('active',b.dataset.id===id));}renderTrace(trace);}
    const picker=document.getElementById('trace-picker');data.traces.forEach((trace,i)=>{const b=node('button',`tab${i===0?' active':''}`,trace.failure?`failed ${short(trace.id)}`:`trace ${short(trace.id)}`);b.dataset.id=trace.id;b.addEventListener('click',()=>selectTrace(trace.id));picker.append(b);});renderTrace(data.traces[0]);

    const timeline=document.getElementById('timeline');if(!data.events.length)timeline.append(node('div','empty','No runtime events.'));data.events.forEach(event=>{const row=node('div','event');row.append(node('time','muted',new Date(event.created_at).toLocaleString()),node('strong','',event.kind),node('pre','',JSON.stringify(event.payload,null,2)));timeline.append(row);});
    document.getElementById('footer').textContent=`Generated from ${data.generated_from} · session data remains authoritative`;
    drawScoreChart();drawLineage();renderCandidateList();selectCandidate(selectedCandidateId,selectedEvaluationId);
  })();
  </script>
</body>
</html>
"""
