"""Optional Weights & Biases reporting for canonical runtime events."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from vero.evaluation.models import CaseStatus, EvaluationRecord, EvaluationStatus
from vero.evaluation.scoring.error_taxonomy import TERMINATING_DIAGNOSTIC_CODES
from vero.evaluation.store.budget import BudgetLedger
from vero.runtime.artifacts import ArtifactStore
from vero.runtime.events import RuntimeEvent

logger = logging.getLogger(__name__)


def normalize_wandb_base_url(environment: dict[str, str] | None = None) -> str | None:
    """Give a scheme-less ``WANDB_BASE_URL`` the ``https://`` it needs.

    A self-hosted host is naturally written ``wandb.example.com``, but W&B's
    settings model parses ``base_url`` as a URL and rejects that with
    ``Input should be a valid URL, relative URL without a base``. The error
    surfaces out of ``wandb.init()``, which callers treat as "W&B unavailable",
    so one missing scheme silently costs a run all of its reporting.

    Returns the value in effect, or None when the variable is unset.
    """
    environ = os.environ if environment is None else environment
    value = (environ.get("WANDB_BASE_URL") or "").strip()
    if not value or "://" in value:
        return value or None
    corrected = f"https://{value}"
    environ["WANDB_BASE_URL"] = corrected
    logger.warning(
        "WANDB_BASE_URL %r has no scheme and would be rejected by W&B; using %r",
        value,
        corrected,
    )
    return corrected


def disambiguated_run_name(name: str | None, run_id: str) -> str | None:
    """Append a short digest of ``run_id`` so one cell's attempts stay distinct.

    The benchmark configs set ``group`` to the benchmark and ``name`` to a
    per-launch cell label (``--param wandb_run=officeqa__claude-sonnet-5``), so a
    relaunch of the same cell mints a fresh run id under a byte-identical display
    name. In the officeqa and browsecomp-plus passes on 2026-07-29 that produced
    several same-named rows per cell -- one abandoned, one live -- with no way to
    tell from the name which was which, exactly when a kill/relaunch decision
    needed reading. Worse, killing a run does not stop it: the sandbox is
    orphaned and its sidecar keeps heartbeating, so the abandoned row stays
    ``state=running`` and keeps logging alongside its replacement.

    Keyed to the resolved id rather than to wall-clock time, so the name is
    stable across a sidecar restart that resumes the same id and differs across
    genuinely separate invocations. Returns None unchanged when no name was set,
    where W&B's own generated name is already unique.
    """
    if not name:
        return name
    return f"{name}--{hashlib.sha256(run_id.encode()).hexdigest()[:6]}"


def _open_wandb_run(
    *,
    project: str,
    session_id: str,
    wandb_dir: Path,
    client: Any | None,
    entity: str | None,
    name: str | None,
    group: str | None,
    tags: list[str] | None,
    mode: str | None,
    notes: str | None,
    config: dict[str, Any] | None,
    run_id: str | None,
) -> Any:
    """Open one resumable W&B run keyed stably to the session. W&B is imported
    lazily so nothing here depends on it unless a sink is constructed."""
    if client is None:
        try:
            import wandb as client
        except ImportError as error:
            raise RuntimeError(
                "W&B reporting requires `pip install scale-vero[wandb]`"
            ) from error
    normalize_wandb_base_url()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    stable_id = run_id or ("vero-" + hashlib.sha256(session_id.encode()).hexdigest()[:16])
    init_kwargs: dict[str, Any] = {
        "project": project,
        "id": stable_id,
        "resume": "allow",
        "dir": str(wandb_dir),
        # `vero/run_label` keeps the undecorated cell label groupable and
        # filterable now that the display name carries a per-attempt suffix.
        "config": {
            **(config or {}),
            "vero/session_id": session_id,
            **({"vero/run_label": name} if name else {}),
        },
    }
    for key, value in {
        "entity": entity,
        "name": disambiguated_run_name(name, stable_id),
        "group": group,
        "tags": tags or None,
        "mode": mode,
        "notes": notes,
    }.items():
        if value is not None:
            init_kwargs[key] = value
    return client.init(**init_kwargs)


class WandbEventSink:
    """Log one optimization session as one W&B run.

    W&B is imported only when this sink is constructed, so the core runtime has
    no mandatory tracking dependency.
    """

    def __init__(
        self,
        *,
        project: str,
        session_id: str,
        session_dir: Path,
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
        notes: str | None = None,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
        client: Any | None = None,
    ):
        if client is None:
            try:
                import wandb as client
            except ImportError as error:
                raise RuntimeError(
                    "W&B reporting requires `pip install scale-vero[wandb]`"
                ) from error

        normalize_wandb_base_url()
        wandb_dir = session_dir / "artifacts" / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(session_dir / "artifacts")
        self.state_path = "wandb/state.json"
        if self.artifacts.path(self.state_path).exists():
            state = self.artifacts.read_json(self.state_path)
            self.logged_evaluations = set(state.get("evaluation_ids", []))
            self.next_step = int(state.get("next_step", len(self.logged_evaluations)))
        else:
            self.logged_evaluations: set[str] = set()
            self.next_step = 0
        stable_id = run_id or (
            "vero-" + hashlib.sha256(session_id.encode()).hexdigest()[:16]
        )
        init_kwargs: dict[str, Any] = {
            "project": project,
            "id": stable_id,
            "resume": "allow",
            "dir": str(wandb_dir),
            "config": {
                **(config or {}),
                "vero/session_id": session_id,
                **({"vero/run_label": name} if name else {}),
            },
        }
        for key, value in {
            "entity": entity,
            "name": disambiguated_run_name(name, stable_id),
            "group": group,
            "tags": tags or None,
            "mode": mode,
            "notes": notes,
        }.items():
            if value is not None:
                init_kwargs[key] = value
        self.run = client.init(**init_kwargs)

    def _save_state(self) -> None:
        self.artifacts.write_json(
            self.state_path,
            {
                "evaluation_ids": sorted(self.logged_evaluations),
                "next_step": self.next_step,
            },
        )

    def __call__(self, event: RuntimeEvent) -> None:
        if event.kind == "evaluation_completed":
            payload = dict(event.payload)
            payload.pop("step")
            evaluation_id = str(payload["evaluation_id"])
            if evaluation_id in self.logged_evaluations:
                return
            # Persist the step before spending it. The old order logged at
            # `next_step` and only then wrote state, so a process killed inside
            # that window came back believing the step was still free and handed
            # W&B a step it had already used: the two points collide on one x
            # value and the history is unreadable exactly where the crash was.
            # This is deliberately at-most-once now -- a crash between the write
            # and the log loses this telemetry point instead of duplicating one,
            # and for a live-watch series a missing point is much the cheaper
            # failure than a colliding one.
            step = self.next_step
            self.next_step += 1
            self.logged_evaluations.add(evaluation_id)
            self._save_state()
            self.run.log(payload, step=step)
            return
        if event.kind == "session_completed":
            self.run.summary.update(event.payload)
            self.run.finish()
            return
        if event.kind == "session_failed":
            self.run.summary.update(
                {
                    "status": "failed",
                    "error_type": event.payload.get("error_type"),
                    "error_message": event.payload.get("message"),
                }
            )
            self.run.finish(exit_code=1)


class SidecarWandbSink:
    """Log the trusted eval-sidecar's evaluation stream to one W&B run.

    Subscribed to ``EvaluationEngine.listeners``, so it sees every completed
    evaluation the sidecar produces — the agent's search evals and the trusted
    SYSTEM/ADMIN re-scores — with the real scores, statuses, diagnostics, and
    remaining budget the sidecar holds. This is the harbor-path live-watch
    surface: it does not depend on the untrusted optimizer agent cooperating,
    and the W&B credentials live only in the trusted container.
    """

    def __init__(
        self,
        *,
        project: str,
        session_id: str,
        session_dir: Path,
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
        notes: str | None = None,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
        budget_ledger: BudgetLedger | None = None,
        log_traces: bool = False,
        client: Any | None = None,
    ):
        if client is None:
            try:
                import wandb as client
            except ImportError as error:
                raise RuntimeError(
                    "W&B reporting requires `pip install scale-vero[wandb]`"
                ) from error
        self._wandb = client
        self.session_dir = session_dir
        self.log_traces = log_traces
        self.artifacts = ArtifactStore(session_dir / "artifacts")
        self.state_path = "wandb/state.json"
        # The telemetry poller calls log_inference_usage/ship_request_logs from a
        # worker thread (`asyncio.to_thread`) while the evaluation listener calls
        # __call__ on the loop thread, so the two genuinely interleave rather
        # than merely looking like they might. Every mutate-then-save body below
        # runs under this lock: without it one caller can read `next_step`, be
        # descheduled, and let the other spend the same step, or a save can
        # serialize a half-updated view and drop the other caller's mutation.
        self._lock = threading.Lock()
        stored_run_id = None
        if self.artifacts.path(self.state_path).exists():
            state = self.artifacts.read_json(self.state_path)
            self.logged_evaluations = set(state.get("evaluation_ids", []))
            self.next_step = int(state.get("next_step", len(self.logged_evaluations)))
            stored_run_id = state.get("run_id")
            self._shipped_request_logs = dict(state.get("request_log_files", {}))
            # The gateway usage ledger is the dedupe key for inference telemetry,
            # and holding it only in memory meant a sidecar restart forgot which
            # counters it had already sent: the resumed run re-logged an
            # identical point at a fresh step and could not say what cumulative
            # usage it had reported. It resumes from state.json like the rest.
            self._last_inference_usage = dict(state.get("inference_usage", {}))
        else:
            self.logged_evaluations: set[str] = set()
            self.next_step = 0
            self._shipped_request_logs: dict[str, int] = {}
            self._last_inference_usage: dict[str, Any] = {}
        # Cumulative evaluation outcomes, keyed "<partition>/<principal>".
        # `diagnostics` below is a last-value field, so it can only ever answer
        # "has at least one evaluation been terminated", never "how many" -- and
        # how many, per partition, is what a severity judgement needs. On
        # 2026-07-29 all eight in-flight browsecomp-plus runs showed
        # diagnostics="inference_budget_exhausted", which read as uniform
        # catastrophe; the per-cell counts recovered afterwards from the session
        # database were 0-4 with most of them followed by a successful retry, a
        # completely different picture. There was no way to tell those apart
        # while the runs were live. These counters exist so there is.
        self._evaluations_ok: dict[str, int] = {}
        self._evaluations_terminated: dict[str, int] = {}
        # One W&B run per invocation. A fresh session volume mints a new id; a
        # sidecar restart within the same run reuses the one persisted in state,
        # so it resumes rather than colliding with other invocations.
        self.run_id = run_id or stored_run_id or f"vero-{uuid4().hex[:16]}"
        self._save_state()
        self.budget_ledger = budget_ledger
        # Keep W&B's symlink-laden run dir out of session_dir, which is archived
        # verbatim for export (a symlink there would break the archive).
        wandb_run_dir = Path(tempfile.mkdtemp(prefix="vero-wandb-"))
        self.run = _open_wandb_run(
            project=project,
            session_id=session_id,
            wandb_dir=wandb_run_dir,
            client=client,
            entity=entity,
            name=name,
            group=group,
            tags=tags,
            mode=mode,
            notes=notes,
            config=config,
            run_id=self.run_id,
        )

    def _save_state(self) -> None:
        self.artifacts.write_json(
            self.state_path,
            {
                "evaluation_ids": sorted(self.logged_evaluations),
                "next_step": self.next_step,
                "run_id": self.run_id,
                "request_log_files": self._shipped_request_logs,
                "inference_usage": self._last_inference_usage,
            },
        )

    def _payload(self, record: EvaluationRecord) -> dict[str, Any]:
        report = record.report
        objective = record.objective
        evaluation_set = record.request.evaluation_set
        partition = evaluation_set.partition or "none"
        principal = record.principal.value
        # Scope metrics by partition/principal so unlike evals don't share a series.
        scope = f"{partition}/{principal}"
        counts = {status: 0 for status in CaseStatus}
        for case in report.cases:
            counts[case.status] += 1
        payload: dict[str, Any] = {
            # Identity / context — flat, shared across every evaluation.
            "evaluation_id": record.id,
            "candidate_id": record.request.candidate.id,
            "candidate_version": record.request.candidate.version,
            "evaluation_set": evaluation_set.name,
            "partition": partition,
            "principal": principal,
            "status": report.status.value,
            "latency_seconds": (
                record.completed_at - record.created_at
            ).total_seconds(),
            # Scoped quality metrics. `num_cases` is the evaluation's sample
            # size (number of trials): evaluations cover different case counts,
            # so a score is only interpretable next to how many cases produced
            # it.
            f"{scope}/num_cases": len(report.cases),
            f"{scope}/cases/success": counts[CaseStatus.SUCCESS],
            f"{scope}/cases/error": counts[CaseStatus.ERROR],
            f"{scope}/cases/skipped": counts[CaseStatus.SKIPPED],
            f"{scope}/feasible": (
                bool(objective.feasible) if objective is not None else False
            ),
        }
        if objective is not None and objective.value is not None:
            payload[f"{scope}/score"] = objective.value
        for key, value in report.metrics.items():
            payload[f"{scope}/metric/{key}"] = value
        if report.diagnostics:
            payload["diagnostics"] = ",".join(
                diagnostic.code for diagnostic in report.diagnostics
            )
        # Cumulative outcome counters alongside that last-value field. Emitted for
        # every evaluation, including zeros, so a flat line is distinguishable
        # from a missing series.
        codes = {diagnostic.code for diagnostic in report.diagnostics}
        terminated = codes & TERMINATING_DIAGNOSTIC_CODES
        if terminated:
            self._evaluations_terminated[scope] = (
                self._evaluations_terminated.get(scope, 0) + 1
            )
        elif report.status is EvaluationStatus.SUCCESS:
            self._evaluations_ok[scope] = self._evaluations_ok.get(scope, 0) + 1
        payload[f"{scope}/evaluations/success_total"] = self._evaluations_ok.get(scope, 0)
        payload[f"{scope}/evaluations/terminated_total"] = (
            self._evaluations_terminated.get(scope, 0)
        )
        # The successful count on the SELECTION partition is the evidence the
        # optimizer had to choose its candidate on. Thin selection evidence -- not
        # corrupted scores -- was the real damage the 2026-07-29 misclassification
        # did, and officeqa cells ranged from 1 to 5. Recovery is deliberately not
        # computed here: whether a terminated evaluation was followed by a
        # successful one is retrospective, and these two series make it derivable
        # downstream without guessing at log time.
        if self.budget_ledger is not None:
            for budget in self.budget_ledger.budgets:
                scope = f"{budget.backend_id}/{budget.principal.value}"
                if budget.remaining_runs is not None:
                    payload[f"budget/{scope}/remaining_runs"] = budget.remaining_runs
                if budget.remaining_cases is not None:
                    payload[f"budget/{scope}/remaining_cases"] = budget.remaining_cases
        return payload

    def _log_trace(self, record: EvaluationRecord) -> None:
        """Upload an evaluation's trace artifacts as one W&B artifact.

        Full job data is on the per-case artifacts, not just the report-level
        harbor logs, so walk both. Best-effort."""
        entries = list(record.report.artifacts)
        for case in record.report.cases:
            entries.extend(case.artifacts)
        if not entries:
            return
        eval_dir = self.session_dir / "evaluations" / record.id / "artifacts"
        artifact = self._wandb.Artifact(
            name=f"trace-{record.id}", type="evaluation_trace"
        )
        added = 0
        seen: set[str] = set()
        for entry in entries:
            if entry.path in seen:
                continue
            seen.add(entry.path)
            path = eval_dir / entry.path
            if path.is_file():
                artifact.add_file(str(path), name=entry.path)
                added += 1
        if added:
            self.run.log_artifact(artifact)

    def log_inference_usage(self, scopes: dict[str, Any]) -> None:
        """Log the gateway's per-scope usage counters as W&B series."""
        payload: dict[str, Any] = {}
        for name, usage in sorted(scopes.items()):
            if not isinstance(usage, dict):
                continue
            for key in (
                "requests",
                "upstream_errors",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "total_tokens",
                "active_requests",
            ):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    payload[f"inference/{name}/{key}"] = value
        # The poller thread shares `next_step` and state.json with the evaluation
        # listener on the loop thread, so the mutate-then-save body is serialized.
        with self._lock:
            if not payload or payload == self._last_inference_usage:
                return
            # Persist the step before spending it, as in __call__: at-most-once,
            # so a crash between the write and the log drops this usage point
            # rather than leaving a resumed process to reuse a step W&B has
            # already been given.
            step = self.next_step
            self.next_step += 1
            self._last_inference_usage = payload
            self._save_state()
            self.run.log(payload, step=step)

    def ship_request_logs(self, directory: Path, *, final: bool = False) -> None:
        """Upload the gateway's rotated request-log files as one W&B artifact.

        The highest-numbered file is still being appended to, so it is only
        included on the final call. Unchanged files dedupe by digest on the
        W&B side, so re-adding them per version is cheap.
        """
        files = sorted(directory.glob("requests-*.jsonl"))
        if not final:
            files = files[:-1]
        if not files:
            return
        snapshot = {path.name: path.stat().st_size for path in files}
        # Same shared-state lock as the other mutate-then-save bodies, but held
        # only around the bookkeeping and never across the upload. log_artifact
        # is a file transfer, and __call__ waits on this same lock from the
        # sidecar's event loop, so spanning the upload would let a slow W&B
        # transfer stall the thing that answers the agent's requests. Leaving the
        # upload outside keeps the order at-least-once: a crash between shipping
        # and saving re-ships next poll, and W&B folds byte-identical content
        # into one artifact version, so a duplicate costs nothing while a lost
        # request log cannot be recovered.
        with self._lock:
            if snapshot == self._shipped_request_logs:
                return
        artifact = self._wandb.Artifact(
            name="inference-requests", type="inference_request_log"
        )
        for path in files:
            artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact)
        with self._lock:
            self._shipped_request_logs = snapshot
            self._save_state()

    def __call__(self, record: EvaluationRecord) -> None:
        # Runs on the loop thread while the telemetry poller mutates the same
        # counters from its worker thread, so hold the lock across the whole body.
        with self._lock:
            if record.id in self.logged_evaluations:
                return
            payload = self._payload(record)
            # Persist the step before spending it. Logging first and saving
            # afterwards meant a kill inside that window brought the sidecar back
            # thinking the step was unused, and it then re-sent this evaluation on
            # a step W&B already holds a point for, so the two collide. Written
            # this way the window is deliberately at-most-once: a crash here
            # loses this evaluation's point rather than duplicating one, and a
            # gap in the series is the cheaper of the two failures.
            step = self.next_step
            self.next_step += 1
            self.logged_evaluations.add(record.id)
            self._save_state()
            self.run.log(payload, step=step)
            if self.log_traces:
                try:
                    self._log_trace(record)
                except Exception:  # tracing must never drop the metric log
                    pass

    def finish(
        self,
        summary: dict[str, Any] | None = None,
        *,
        failed: bool = False,
    ) -> None:
        """Update the run summary (e.g. shipped/rewards at finalize) and close."""
        if summary:
            self.run.summary.update(summary)
        self.run.finish(exit_code=1 if failed else 0)


class InferenceTelemetryPoller:
    """Mirror the gateway's durable state into the sidecar's W&B run.

    The gateway state volume is mounted read-only in the trusted sidecar, so
    this gives live inference telemetry (including the untrusted optimizer's
    producer-scope burn) without W&B credentials leaving the sidecar.
    Best-effort throughout: telemetry must never affect the eval path.
    """

    def __init__(
        self,
        *,
        sink: SidecarWandbSink,
        usage_path: Path | None = None,
        request_log_dir: Path | None = None,
        interval_seconds: float = 30.0,
    ):
        if usage_path is None and request_log_dir is None:
            raise ValueError("telemetry poller requires at least one source")
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.sink = sink
        self.usage_path = usage_path
        self.request_log_dir = request_log_dir
        self.interval_seconds = interval_seconds

    def poll_once(self, *, final: bool = False) -> None:
        if self.usage_path is not None:
            try:
                if self.usage_path.exists():
                    value = json.loads(self.usage_path.read_text(encoding="utf-8"))
                    scopes = value.get("scopes") if isinstance(value, dict) else None
                    if isinstance(scopes, dict):
                        self.sink.log_inference_usage(scopes)
            except Exception:
                logger.warning("inference usage telemetry failed", exc_info=True)
        if self.request_log_dir is not None:
            try:
                if self.request_log_dir.is_dir():
                    self.sink.ship_request_logs(self.request_log_dir, final=final)
            except Exception:
                logger.warning("inference request log shipping failed", exc_info=True)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            await asyncio.to_thread(self.poll_once)
