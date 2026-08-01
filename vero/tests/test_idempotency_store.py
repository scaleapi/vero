"""Durability regressions for the evaluation store's one atomic writer and index."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import vero.evaluation.store.persistence as persistence
from vero.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    CaseResult,
    CaseStatus,
    EvaluationDatabase,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationStore,
)


def record(candidate_id: str = "candidate") -> EvaluationRecord:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    return EvaluationRecord(
        id=f"evaluation:{candidate_id}",
        request=EvaluationRequest(
            candidate=Candidate(
                id=candidate_id,
                version=f"version:{candidate_id}",
                created_at=created_at,
            )
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 1.0},
            cases=[
                CaseResult(
                    case_id="case-one",
                    status=CaseStatus.SUCCESS,
                    metrics={"score": 1.0},
                )
            ],
        ),
        backend_id="default",
        backend=BackendProvenance(name="stub", version="1", config_digest="0" * 64),
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=1),
    )


def test_atomic_write_fsyncs_the_parent_directory_after_the_rename(
    tmp_path: Path,
    monkeypatch,
):
    """The rename itself has to be flushed, not just the bytes it publishes."""

    real_fsync = os.fsync
    fsynced_directories: list[str] = []

    def tracking_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            fsynced_directories.append(str(descriptor))
        real_fsync(descriptor)

    monkeypatch.setattr(persistence.os, "fsync", tracking_fsync)

    target = tmp_path / "session" / "database.json"
    persistence._atomic_write_json(target, {"schema_version": 1})
    monkeypatch.undo()

    assert json.loads(target.read_text(encoding="utf-8")) == {"schema_version": 1}
    assert fsynced_directories, "the parent directory was never fsynced"


def test_atomic_write_does_not_close_a_descriptor_it_handed_to_the_file_object(
    tmp_path: Path,
    monkeypatch,
):
    """A failure inside the write must not close the descriptor a second time.

    The file object closes the descriptor on its way out of the block, so the
    old shared failure path closed it again; three writers reach this helper
    concurrently through asyncio.to_thread, where that second close can land on
    an unrelated descriptor that has since been handed the same number.
    """

    real_mkstemp = tempfile.mkstemp
    handed_out: list[int] = []

    def tracking_mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        handed_out.append(descriptor)
        return descriptor, name

    real_close = os.close
    closed: list[int] = []

    def tracking_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(persistence.tempfile, "mkstemp", tracking_mkstemp)
    monkeypatch.setattr(persistence.os, "close", tracking_close)

    target = tmp_path / "evaluation.json"
    with pytest.raises(TypeError):
        # json.dump raises from inside the block, once the file object owns the
        # descriptor.
        persistence._atomic_write_json(target, {"not_serializable": object()})
    monkeypatch.undo()

    assert len(handed_out) == 1
    assert handed_out[0] not in closed
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.asyncio
async def test_reconciled_load_rebuilds_a_torn_index_from_the_manifests(
    tmp_path: Path,
    caplog,
):
    """A truncated database.json must not brick every later run of the session."""

    value = record()
    await EvaluationStore(tmp_path / "evaluations" / value.id).save(value)
    database_path = tmp_path / "database.json"
    EvaluationDatabase(id="session").save_to_file(database_path)
    intact = database_path.read_text(encoding="utf-8")
    database_path.write_text(intact[: len(intact) // 2], encoding="utf-8")

    with caplog.at_level("WARNING"):
        restored = EvaluationDatabase.load_reconciled(
            database_path=database_path,
            evaluations_dir=tmp_path / "evaluations",
            database_id="session",
        )

    assert restored.id == "session"
    assert restored.get_evaluation(value.id) == value
    assert "Rebuilding unreadable evaluation database" in caplog.text
    # The repaired index is written back, so the next run reads a whole file.
    reloaded = EvaluationDatabase.load_from_file(database_path)
    assert reloaded.get_evaluation(value.id) == value


def test_reconciled_load_still_rejects_an_index_from_another_session(tmp_path: Path):
    """A readable index naming another session stays a hard error, not a rebuild."""

    database_path = tmp_path / "database.json"
    EvaluationDatabase(id="other-session").save_to_file(database_path)

    with pytest.raises(ValueError, match="belongs to 'other-session'"):
        EvaluationDatabase.load_reconciled(
            database_path=database_path,
            evaluations_dir=tmp_path / "evaluations",
            database_id="session",
        )
