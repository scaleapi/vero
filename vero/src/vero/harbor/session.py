"""Portable, reportable session snapshots for ephemeral Harbor environments."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tarfile
import tempfile
from collections import deque
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import field_validator

from vero.evaluation import BackendProvenance, EvaluationModel
from vero.evaluation.persistence import _atomic_write_json
from vero.harbor.verifier import VerificationSelection, VerificationTarget

logger = logging.getLogger(__name__)


class HarborSessionManifest(EvaluationModel):
    """Trusted metadata needed to interpret an exported Harbor session."""

    schema_version: Literal[1] = 1
    id: str
    task_name: str
    task_description: str = ""
    created_at: datetime
    candidate_repository_family: Literal["git"] = "git"
    candidate_repository_format_version: Literal[1] = 1
    backends: dict[str, BackendProvenance]
    selection: VerificationSelection
    targets: list[VerificationTarget]

    @field_validator("id", "task_name")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Harbor session identity must not be empty")
        return value

    @field_validator("created_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Harbor session timestamp must be timezone-aware")
        return value.astimezone(UTC)


def initialize_harbor_session_manifest(
    session_dir: Path,
    *,
    session_id: str,
    task_name: str,
    task_description: str,
    backends: dict[str, BackendProvenance],
    selection: VerificationSelection,
    targets: list[VerificationTarget],
) -> HarborSessionManifest:
    """Create the immutable session identity or validate it on restart."""

    path = session_dir / "harbor-session.json"
    if path.is_file():
        stored = HarborSessionManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        expected = HarborSessionManifest(
            id=session_id,
            task_name=task_name,
            task_description=task_description,
            created_at=stored.created_at,
            backends=backends,
            selection=selection,
            targets=targets,
        )
        if stored != expected:
            raise ValueError("Harbor session manifest is incompatible with deployment")
        return stored

    manifest = HarborSessionManifest(
        id=session_id,
        task_name=task_name,
        task_description=task_description,
        created_at=datetime.now(UTC),
        backends=backends,
        selection=selection,
        targets=targets,
    )
    _atomic_write_json(path, manifest.model_dump(mode="json"))
    return manifest


def _walk_without_following_links(root: Path):
    """Yield every entry beneath ``root`` without ever descending through a
    symbolic link (so the walk cannot escape the tree or loop)."""
    queue: deque[Path] = deque([root])
    while queue:
        current = queue.popleft()
        try:
            children = sorted(current.iterdir())
        except OSError:
            continue
        for child in children:
            yield child
            if child.is_dir() and not child.is_symlink():
                queue.append(child)


def create_harbor_session_archive(
    session_dir: Path | str,
    destination: Path | str,
) -> Path:
    """Atomically archive a session beneath a stable ``session/`` root.

    Resilient: no single entry can abort the export. Symlinks are recorded as
    inert ``<name>.symlink`` metadata (not link members, which the extractor
    refuses), unreadable/special files are skipped, and every omission is listed
    in ``vero-export-skipped.json`` inside the archive.
    """

    source = Path(session_dir).expanduser().resolve()
    if not (source / "harbor-session.json").is_file():
        raise FileNotFoundError("Harbor session manifest not found")

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    skipped: list[dict[str, str]] = []

    def _text_member(archive: tarfile.TarFile, arcname: str, payload: bytes) -> None:
        info = tarfile.TarInfo(arcname)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    try:
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(source, arcname="session", recursive=False)
            for path in _walk_without_following_links(source):
                arcname = "session/" + path.relative_to(source).as_posix()
                if path.is_symlink():
                    try:
                        target = os.readlink(path)
                    except OSError as error:
                        target = f"<unreadable link: {error.__class__.__name__}>"
                    _text_member(
                        archive,
                        arcname + ".symlink",
                        (f"symlink -> {target}\n").encode("utf-8"),
                    )
                    skipped.append(
                        {"path": arcname, "reason": "symlink", "target": target}
                    )
                elif path.is_dir():
                    archive.add(path, arcname=arcname, recursive=False)
                elif path.is_file():
                    try:
                        archive.add(path, arcname=arcname, recursive=False)
                    except OSError as error:
                        skipped.append(
                            {
                                "path": arcname,
                                "reason": f"unreadable: {error.__class__.__name__}",
                            }
                        )
                else:
                    skipped.append({"path": arcname, "reason": "special-file"})
            if skipped:
                _text_member(
                    archive,
                    "session/vero-export-skipped.json",
                    (json.dumps({"skipped": skipped}, indent=2) + "\n").encode("utf-8"),
                )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if skipped:
        logger.warning(
            "session export: %d entr%s not archived verbatim "
            "(symlinks recorded as metadata; special/unreadable files skipped); "
            "see session/vero-export-skipped.json",
            len(skipped),
            "y" if len(skipped) == 1 else "ies",
        )
    return output


def extract_harbor_session_archive(
    archive_path: Path | str,
    destination: Path | str,
) -> Path:
    """Extract a trusted sidecar export without permitting link or path traversal."""

    archive_path = Path(archive_path).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or not path.parts
                or path.parts[0] != "session"
                or ".." in path.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
            ):
                raise ValueError(f"unsafe Harbor session archive member: {member.name}")
        archive.extractall(destination, members=members, filter="data")
    session = destination / "session"
    HarborSessionManifest.model_validate_json(
        (session / "harbor-session.json").read_text(encoding="utf-8")
    )
    return session


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
