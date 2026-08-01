"""Durability regressions for the Harbor CLI's terminal writes.

Every file this module is about is written once, at the end of a run that took
hours, and is the only copy of what the run produced: reward.json, the session
archive, and the archive's checksum. The tests below pin the ordering and the
durability properties that make those files survive a crash or a partial
failure, not the numbers inside them.
"""

from __future__ import annotations

import io
import json
import os
import stat
import urllib.request
from pathlib import Path

import pytest
from click.testing import CliRunner

import vero.harbor.cli as harbor_cli
import vero.report as report_module
from vero.cli import main
from vero.layout import LAYOUT
from vero.sidecar.auth import write_admin_token
from vero.sidecar.session import file_sha256


def _record_fsynced_inodes(monkeypatch) -> list[int]:
    """Collect the inode of every directory handed to ``os.fsync``."""
    inodes: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor):
        info = os.fstat(descriptor)
        if stat.S_ISDIR(info.st_mode):
            inodes.append(info.st_ino)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    return inodes


def test_atomic_write_bytes_fsyncs_the_parent_directory(tmp_path, monkeypatch):
    inodes = _record_fsynced_inodes(monkeypatch)
    destination = tmp_path / "verifier" / "reward.json"

    harbor_cli._atomic_write_bytes(destination, b'{"reward": 0.0}\n')

    assert destination.read_bytes() == b'{"reward": 0.0}\n'
    assert destination.parent.stat().st_ino in inodes


def test_download_fsyncs_the_parent_directory(tmp_path, monkeypatch):
    monkeypatch.setenv(LAYOUT.eval_url_env, "http://sidecar")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request: io.BytesIO(b"sidecar archive"),
    )
    inodes = _record_fsynced_inodes(monkeypatch)
    destination = tmp_path / "verifier" / "session.tar.gz"

    harbor_cli._download("/session/export", destination)

    assert destination.read_bytes() == b"sidecar archive"
    assert destination.parent.stat().st_ino in inodes


def test_finalize_routes_both_records_through_the_atomic_writer(tmp_path, monkeypatch):
    token_file = write_admin_token(tmp_path / "token", "admin-secret")
    output = tmp_path / "verifier" / "reward.json"
    written: list[Path] = []
    real_atomic = harbor_cli._atomic_write_bytes

    def recording_atomic(path, payload):
        written.append(Path(path))
        return real_atomic(path, payload)

    monkeypatch.setattr(harbor_cli, "_atomic_write_bytes", recording_atomic)
    monkeypatch.setattr(
        harbor_cli,
        "_request",
        lambda method, path, *, payload=None, headers=None: {
            "rewards": {"reward": 0.25},
            "baseline_rewards": {"reward": 0.1},
            "errors": {},
        },
    )

    result = CliRunner().invoke(
        main,
        [
            "harbor",
            "finalize",
            "--token-file",
            str(token_file),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert written == [output, output.parent / "finalization.json"]
    assert json.loads(output.read_text()) == {"reward": 0.25}
    finalization = json.loads((output.parent / "finalization.json").read_text())
    assert finalization["baseline_rewards"] == {"reward": 0.1}
    # A rename-published write leaves no temporary behind for Harbor to trip on.
    assert sorted(path.name for path in output.parent.iterdir()) == [
        "finalization.json",
        "reward.json",
    ]


class _ExportHarness:
    """Mocked sidecar for ``export-session``: no HTTP, no real archive."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.requests: list[tuple[str, str]] = []
        self.archived_finalizations: list[object] = []
        self.token_file = write_admin_token(tmp_path / "token", "admin-secret")
        self.output = tmp_path / "logs" / "session.tar.gz"
        self.report = tmp_path / "logs" / "experiment.html"
        self.status_output = tmp_path / "logs" / "status.json"
        self.finalization_output = tmp_path / "logs" / "finalization.json"
        self.trace = tmp_path / "trajectory.json"
        self.trace.write_text("[]\n")
        self.extract_error: Exception | None = None

        def fake_request(method, path, *, payload=None, headers=None):
            self.requests.append((method, path))
            if path == "/finalize":
                return {"candidate": "posted", "rewards": {"reward": 0.0}}
            return {"submit_enabled": False, "evaluation_access": []}

        def fake_download(path, destination, *, headers=None):
            destination.write_bytes(b"sidecar archive")

        def fake_extract(_archive, destination):
            if self.extract_error is not None:
                raise self.extract_error
            session = destination / "session"
            session.mkdir(parents=True)
            (session / "harbor-session.json").write_text("{}\n")
            return session

        async def fake_report(session, destination):
            # The archived finalization is what the report is built from, so
            # read it here to prove which copy the export actually used.
            self.archived_finalizations.append(
                json.loads((session / "harbor-finalization.json").read_text())
            )
            destination.write_text("<html>experiment</html>\n")
            return destination

        monkeypatch.setattr(harbor_cli, "_request", fake_request)
        monkeypatch.setattr(harbor_cli, "_download", fake_download)
        monkeypatch.setattr(harbor_cli, "extract_harbor_session_archive", fake_extract)
        monkeypatch.setattr(report_module, "generate_experiment_report", fake_report)

    def invoke(self):
        return CliRunner().invoke(
            main,
            [
                "harbor",
                "export-session",
                "--token-file",
                str(self.token_file),
                "--output",
                str(self.output),
                "--report-output",
                str(self.report),
                "--status-output",
                str(self.status_output),
                "--finalization-output",
                str(self.finalization_output),
                "--agent-trace",
                str(self.trace),
            ],
        )


def test_export_session_keeps_the_raw_archive_when_augmentation_fails(
    tmp_path, monkeypatch
):
    harness = _ExportHarness(tmp_path, monkeypatch)
    harness.extract_error = ValueError("unsafe Harbor session archive member: session")

    result = harness.invoke()

    assert result.exit_code != 0
    # The download landed at --output before anything could fail, so the
    # operator still holds the un-augmented archive rather than nothing.
    assert harness.output.read_bytes() == b"sidecar archive"


def test_export_session_writes_the_checksum_before_the_report(tmp_path, monkeypatch):
    harness = _ExportHarness(tmp_path, monkeypatch)
    real_atomic = harbor_cli._atomic_write_bytes

    def failing_atomic(path, payload):
        if Path(path) == harness.report:
            raise OSError("simulated /logs exhaustion")
        return real_atomic(path, payload)

    monkeypatch.setattr(harbor_cli, "_atomic_write_bytes", failing_atomic)

    result = harness.invoke()

    assert result.exit_code != 0
    assert not harness.report.exists()
    checksum = harness.output.with_name(f"{harness.output.name}.sha256")
    assert checksum.read_text() == f"{file_sha256(harness.output)}  session.tar.gz\n"


@pytest.mark.parametrize(
    "payload",
    ['{"candidate": "planted", "rewards": {"reward": 1.0}}', "{ truncated", ""],
)
def test_export_session_never_trusts_a_finalization_it_finds_on_disk(
    tmp_path, monkeypatch, payload
):
    """The archived held-out result must come from the sidecar, always.

    Skipping the POST when --finalization-output already exists would save the
    run's second finalize, and it is tempting because the generated verifier
    script writes that file moments earlier. It is also a forgery vector: the
    default path is under /logs, which candidate harness code can write in the
    topology these benchmarks run in, so a planted record would reach the
    archive, the status file and the report. A trusted reusable copy has to live
    on the admin volume instead.
    """

    harness = _ExportHarness(tmp_path, monkeypatch)
    harness.finalization_output.parent.mkdir(parents=True, exist_ok=True)
    harness.finalization_output.write_text(payload)

    result = harness.invoke()

    assert result.exit_code == 0, result.output
    assert ("POST", "/finalize") in harness.requests
    assert harness.archived_finalizations[0]["candidate"] == "posted"
    assert json.loads(harness.finalization_output.read_text())["candidate"] == "posted"
