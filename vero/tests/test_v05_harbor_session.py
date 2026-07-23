from __future__ import annotations

import io
import json
import tarfile
from datetime import UTC, datetime

import pytest

from vero.evaluation import (
    BackendProvenance,
    EvaluationSet,
    MetricSelector,
    ObjectiveSpec,
)
from vero.harbor.session import (
    HarborSessionManifest,
    create_harbor_session_archive,
    extract_harbor_session_archive,
    file_sha256,
)
from vero.harbor.verifier import VerificationSelection, VerificationTarget


def _manifest() -> HarborSessionManifest:
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )
    evaluation_set = EvaluationSet(name="benchmark", partition="validation")
    return HarborSessionManifest(
        id="trial",
        task_name="org/optimize",
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
        backends={
            "validation": BackendProvenance(
                name="harbor",
                version="2",
                config_digest="a" * 64,
            )
        },
        selection=VerificationSelection(mode="submit"),
        targets=[
            VerificationTarget(
                reward_key="reward",
                backend_id="validation",
                evaluation_set=evaluation_set,
                objective=objective,
            )
        ],
    )


def test_harbor_session_archive_round_trip_and_checksum(tmp_path):
    session = tmp_path / "source"
    session.mkdir()
    (session / "harbor-session.json").write_text(
        _manifest().model_dump_json(indent=2) + "\n"
    )
    (session / "database.json").write_text('{"id":"trial"}\n')
    archive = create_harbor_session_archive(session, tmp_path / "session.tar.gz")

    extracted = extract_harbor_session_archive(archive, tmp_path / "extracted")

    assert (
        HarborSessionManifest.model_validate_json(
            (extracted / "harbor-session.json").read_text()
        ).id
        == "trial"
    )
    assert len(file_sha256(archive)) == 64


def test_harbor_session_archive_records_symlinks_as_metadata_without_failing(tmp_path):
    # Regression: a single symlink anywhere in the tree used to raise and 500 the
    # whole export, discarding an entire run's data. It must instead succeed,
    # preserve each link's target as inert metadata, and record the omission.
    session = tmp_path / "source"
    session.mkdir()
    (session / "harbor-session.json").write_text(
        _manifest().model_dump_json(indent=2) + "\n"
    )
    (session / "database.json").write_text('{"id":"trial"}\n')
    (session / "link_in").symlink_to("database.json")  # relative, in-tree
    (session / "link_abs").symlink_to("/etc/hostname")  # absolute, out-of-tree

    archive = create_harbor_session_archive(session, tmp_path / "session.tar.gz")

    # No link members survive (keeps the archive extractable + traversal-safe).
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert not any(m.issym() or m.islnk() for m in members)
        names = {m.name for m in members}
    assert "session/link_in.symlink" in names
    assert "session/link_abs.symlink" in names
    assert "session/vero-export-skipped.json" in names

    extracted = extract_harbor_session_archive(archive, tmp_path / "extracted")
    assert (extracted / "database.json").read_text() == '{"id":"trial"}\n'
    assert "database.json" in (extracted / "link_in.symlink").read_text()
    assert "/etc/hostname" in (extracted / "link_abs.symlink").read_text()
    skipped = json.loads((extracted / "vero-export-skipped.json").read_text())
    reasons = {entry["path"]: entry["reason"] for entry in skipped["skipped"]}
    assert reasons["session/link_in"] == "symlink"
    assert reasons["session/link_abs"] == "symlink"


def test_harbor_session_archive_rejects_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("session/../../escape")
        payload = b"bad"
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe Harbor session archive member"):
        extract_harbor_session_archive(archive, tmp_path / "extracted")
