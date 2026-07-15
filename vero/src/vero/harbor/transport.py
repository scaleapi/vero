"""Candidate transfer across the Harbor sidecar trust boundary."""

from __future__ import annotations

import re
import os
from datetime import datetime
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable
from urllib.parse import quote
from uuid import uuid4

from vero.candidate import Candidate
from vero.workspace import Workspace


class CandidateTransferError(RuntimeError):
    """Raised when an untrusted candidate cannot be imported safely."""


@runtime_checkable
class CandidateTransport(Protocol):
    """Import an external program version into the evaluator's workspace."""

    async def import_candidate(self, version: str | None = None) -> Candidate: ...


class GitCandidateTransport:
    """Copy a commit from an agent repository into a trusted Git workspace.

    The source ref is resolved before fetching, the object is fetched through a
    unique temporary ref, and the imported commit is retained under a stable
    ``refs/vero/candidates`` ref. This avoids the process-global ``FETCH_HEAD``
    race and ensures later verifier evaluations do not depend on the agent
    repository still retaining the object.
    """

    _OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

    def __init__(
        self,
        *,
        workspace: Workspace,
        agent_repo_path: str,
        fetch_timeout_seconds: int = 120,
    ):
        if not agent_repo_path.startswith("/"):
            raise ValueError("agent_repo_path must be an absolute sandbox path")
        if fetch_timeout_seconds <= 0:
            raise ValueError("fetch_timeout_seconds must be positive")
        if PurePosixPath(agent_repo_path) == PurePosixPath(workspace.root):
            raise ValueError("agent and trusted repositories must be distinct")
        self.workspace = workspace
        self.agent_repo_path = agent_repo_path.rstrip("/") or "/"
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self._candidates: dict[str, Candidate] = {}

    async def _run(
        self,
        command: list[str],
        *,
        cwd: str,
        timeout: int,
    ) -> str:
        if command[0] != "git":
            raise ValueError("candidate transport only permits Git commands")
        command = [
            "git",
            "-c",
            f"safe.directory={cwd}",
            *command[1:],
        ]
        result = await self.workspace.sandbox.run(
            command,
            cwd=cwd,
            timeout=timeout,
            env={
                "PATH": os.defpath,
                "LANG": "C.UTF-8",
                "GIT_CONFIG_GLOBAL": "/dev/null",
            },
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "git failed"
            raise CandidateTransferError(message)
        return result.stdout.strip()

    async def _resolve_ref(self, version: str, *, repository: str) -> str:
        value = await self._run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{version}^{{commit}}",
            ],
            cwd=repository,
            timeout=30,
        )
        if self._OBJECT_ID.fullmatch(value) is None:
            raise CandidateTransferError(
                "source ref did not resolve to a Git object ID"
            )
        return value

    async def trusted_candidate(self, version: str | None = None) -> Candidate:
        """Resolve an existing commit already owned by the trusted workspace."""
        source_version = version or "HEAD"
        if not source_version.strip() or "\x00" in source_version:
            raise CandidateTransferError("candidate version must not be empty")
        object_id = await self._resolve_ref(
            source_version,
            repository=self.workspace.root,
        )
        cached = self._candidates.get(object_id)
        if cached is None:
            cached = await self._candidate_metadata(object_id)
            self._candidates[object_id] = cached
        return cached

    async def _candidate_metadata(self, object_id: str) -> Candidate:
        value = await self._run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "show",
                "-s",
                "--format=%cI%x00%P%x00%T%x00%s",
                "--end-of-options",
                object_id,
            ],
            cwd=self.workspace.root,
            timeout=30,
        )
        parts = value.split("\x00", 3)
        if len(parts) != 4:
            raise CandidateTransferError("could not read imported commit metadata")
        timestamp, parents, tree, subject = parts
        if self._OBJECT_ID.fullmatch(tree) is None:
            raise CandidateTransferError("imported commit has an invalid tree identity")
        try:
            created_at = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise CandidateTransferError(
                "imported commit has an invalid timestamp"
            ) from error
        parent_id = parents.split()[0] if parents.strip() else None
        return Candidate(
            id=object_id,
            version=object_id,
            parent_id=parent_id,
            created_at=created_at,
            description=subject.strip() or None,
            metadata={"transport": "git", "content_digest": tree},
        )

    async def import_candidate(self, version: str | None = None) -> Candidate:
        source_version = version or "HEAD"
        if not source_version.strip() or "\x00" in source_version:
            raise CandidateTransferError("candidate version must not be empty")
        object_id = await self._resolve_ref(
            source_version,
            repository=self.agent_repo_path,
        )
        cached = self._candidates.get(object_id)
        if cached is not None:
            return cached

        nonce = uuid4().hex
        temporary_ref = f"refs/vero/incoming/{nonce}"
        retained_ref = f"refs/vero/candidates/{object_id}"
        source_url = f"file://{quote(self.agent_repo_path, safe='/')}"
        try:
            await self._run(
                [
                    "git",
                    "-c",
                    f"safe.directory={self.agent_repo_path}",
                    "-c",
                    f"safe.directory={self.agent_repo_path}/.git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "protocol.file.allow=always",
                    "-C",
                    self.workspace.root,
                    "fetch",
                    "--force",
                    "--no-tags",
                    "--no-recurse-submodules",
                    "--",
                    source_url,
                    f"+{object_id}:{temporary_ref}",
                ],
                cwd=self.workspace.root,
                timeout=self.fetch_timeout_seconds,
            )
            await self._run(
                [
                    "git",
                    "-C",
                    self.workspace.root,
                    "update-ref",
                    retained_ref,
                    object_id,
                ],
                cwd=self.workspace.root,
                timeout=30,
            )
            candidate = await self._candidate_metadata(object_id)
            self._candidates[object_id] = candidate
            return candidate
        finally:
            try:
                await self.workspace.sandbox.run(
                    [
                        "git",
                        "-c",
                        f"safe.directory={self.workspace.root}",
                        "-C",
                        self.workspace.root,
                        "update-ref",
                        "-d",
                        temporary_ref,
                    ],
                    cwd=self.workspace.root,
                    timeout=30,
                    env={
                        "PATH": os.defpath,
                        "LANG": "C.UTF-8",
                        "GIT_CONFIG_GLOBAL": "/dev/null",
                    },
                )
            except Exception:
                # Cleanup is best-effort. A unique leftover ref is safe and
                # auditable; it must not mask the original transfer failure.
                pass
