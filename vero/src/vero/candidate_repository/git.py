"""Git-backed durable candidate repository."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Literal, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from vero.candidate import Candidate
from vero.candidate_repository.base import CandidateRepository, CandidateRepositoryError
from vero.evaluation.persistence import _atomic_write_json
from vero.sandbox import LocalSandbox, Sandbox
from vero.workspace import GitWorkspace, Workspace

_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_AGENT_CONTEXT_DIRECTORY = ".evals"


class _GitRepositoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    family: Literal["git"] = "git"
    project_subpath: str = "."

    @field_validator("project_subpath")
    @classmethod
    def validate_project_subpath(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("project_subpath must stay within the Git repository")
        return value


class _CandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    candidate: Candidate


def _candidate_digest(candidate_id: str) -> str:
    return hashlib.sha256(candidate_id.encode()).hexdigest()


def _safe_prefix(name: str | None) -> str:
    if name is None:
        return "vero-candidate-"
    value = "".join(
        character for character in name if character.isalnum() or character in "-_"
    )
    return f"{value[:48] or 'vero-candidate'}-"


class GitCandidateRepository(CandidateRepository[GitWorkspace]):
    """A session-owned bare Git repository plus durable candidate records."""

    def __init__(
        self,
        *,
        root: Path,
        host: LocalSandbox,
        config: _GitRepositoryConfig,
        candidates: dict[str, Candidate],
    ) -> None:
        self.root = root
        self.repository_path = root / "repository.git"
        self.records_path = root / "records"
        self.config_path = root / "repository.json"
        self._host = host
        self._config = config
        self._candidates = candidates
        self._lock = asyncio.Lock()

    @property
    def family(self) -> str:
        return "git"

    @property
    def format_version(self) -> int:
        return self._config.schema_version

    @property
    def project_subpath(self) -> str:
        return self._config.project_subpath

    @classmethod
    async def create(
        cls,
        root: Path | str,
        *,
        workspace: GitWorkspace,
    ) -> GitCandidateRepository:
        """Create or open the Git repository paired with ``workspace``."""

        root = Path(root).expanduser().resolve()
        try:
            relative = PurePosixPath(workspace.project_path).relative_to(workspace.root)
        except ValueError as error:
            raise ValueError(
                "workspace project path must be within its Git root"
            ) from error
        project_subpath = str(relative)
        config = _GitRepositoryConfig(project_subpath=project_subpath)
        root.mkdir(parents=True, exist_ok=True)
        host = await LocalSandbox.create(root=root.parent)
        config_path = root / "repository.json"
        repository_path = root / "repository.git"
        records_path = root / "records"

        if config_path.exists():
            stored = _GitRepositoryConfig.model_validate_json(
                config_path.read_text(encoding="utf-8")
            )
            if stored != config:
                raise ValueError(
                    "Git candidate repository does not match workspace project path"
                )
            config = stored
        else:
            await asyncio.to_thread(
                _atomic_write_json,
                config_path,
                config.model_dump(mode="json"),
            )

        if not repository_path.exists():
            result = await host.run(["git", "init", "--bare", str(repository_path)])
            if result.returncode != 0:
                raise CandidateRepositoryError(
                    result.stderr or "failed to initialize candidate repository"
                )
        records_path.mkdir(parents=True, exist_ok=True)
        instance = cls(
            root=root,
            host=host,
            config=config,
            candidates={},
        )
        await instance._load_records()
        return instance

    @classmethod
    async def open(cls, root: Path | str) -> GitCandidateRepository:
        """Open an existing Git candidate repository without a source workspace."""

        root = Path(root).expanduser().resolve()
        config_path = root / "repository.json"
        repository_path = root / "repository.git"
        if not config_path.is_file() or not repository_path.is_dir():
            raise FileNotFoundError(f"candidate repository does not exist: {root}")
        config = _GitRepositoryConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
        host = await LocalSandbox.create(root=root.parent)
        instance = cls(root=root, host=host, config=config, candidates={})
        await instance._load_records()
        return instance

    def supports(self, workspace: Workspace) -> bool:
        if not isinstance(workspace, GitWorkspace):
            return False
        try:
            relative = PurePosixPath(workspace.project_path).relative_to(workspace.root)
        except ValueError:
            return False
        return str(relative) == self.project_subpath

    def _record_path(self, candidate_id: str) -> Path:
        return self.records_path / f"{_candidate_digest(candidate_id)}.json"

    def _candidate_ref(self, candidate_id: str) -> str:
        return f"refs/vero/candidates/{_candidate_digest(candidate_id)}"

    def _context_ref(self, candidate_id: str) -> str:
        return f"refs/vero/context/{_candidate_digest(candidate_id)}"

    @property
    def _reserved_context_path(self) -> str:
        if self.project_subpath == ".":
            return _AGENT_CONTEXT_DIRECTORY
        return str(PurePosixPath(self.project_subpath) / _AGENT_CONTEXT_DIRECTORY)

    async def _host_git(self, *arguments: str, timeout: int = 120) -> str:
        result = await self._host.run(
            ["git", "--git-dir", str(self.repository_path), *arguments],
            timeout=timeout,
            env={
                "PATH": os.defpath,
                "LANG": "C.UTF-8",
                "GIT_CONFIG_GLOBAL": "/dev/null",
            },
        )
        if result.returncode != 0:
            raise CandidateRepositoryError(
                result.stderr.strip() or result.stdout.strip() or "git command failed"
            )
        return result.stdout.strip()

    async def _load_records(self) -> None:
        loaded: dict[str, Candidate] = {}
        for path in sorted(self.records_path.glob("*.json")):
            try:
                record = _CandidateRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                raise CandidateRepositoryError(
                    f"invalid candidate record {path.name}: {error}"
                ) from error
            candidate = record.candidate
            if path != self._record_path(candidate.id):
                raise CandidateRepositoryError(
                    f"candidate record path does not match identity {candidate.id!r}"
                )
            if candidate.id in loaded:
                raise CandidateRepositoryError(
                    f"duplicate candidate record: {candidate.id!r}"
                )
            ref = self._candidate_ref(candidate.id)
            try:
                resolved = await self._host_git(
                    "rev-parse", "--verify", f"{ref}^{{commit}}", timeout=30
                )
            except CandidateRepositoryError as error:
                raise CandidateRepositoryError(
                    f"candidate {candidate.id!r} references a missing Git object"
                ) from error
            if resolved != candidate.version:
                raise CandidateRepositoryError(
                    f"candidate {candidate.id!r} ref does not match its version"
                )
            loaded[candidate.id] = candidate
        self._candidates = loaded

    async def _source_git(
        self,
        workspace: GitWorkspace,
        *arguments: str,
        timeout: int = 120,
    ) -> str:
        result = await workspace.sandbox.run(
            [
                "git",
                "-c",
                f"safe.directory={workspace.root}",
                "-c",
                "core.hooksPath=/dev/null",
                *arguments,
            ],
            cwd=workspace.root,
            timeout=timeout,
            env={
                "PATH": os.defpath,
                "LANG": "C.UTF-8",
                "GIT_CONFIG_GLOBAL": "/dev/null",
            },
        )
        if result.returncode != 0:
            raise CandidateRepositoryError(
                result.stderr.strip()
                or result.stdout.strip()
                or "source git command failed"
            )
        return result.stdout.strip()

    async def _repository_git(
        self,
        sandbox: Sandbox,
        repository_path: str,
        *arguments: str,
        timeout: int = 120,
    ) -> str:
        result = await sandbox.run(
            [
                "git",
                "-c",
                f"safe.directory={repository_path}",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                repository_path,
                *arguments,
            ],
            cwd=repository_path,
            timeout=timeout,
            env={
                "PATH": os.defpath,
                "LANG": "C.UTF-8",
                "GIT_CONFIG_GLOBAL": "/dev/null",
            },
        )
        if result.returncode != 0:
            raise CandidateRepositoryError(
                result.stderr.strip()
                or result.stdout.strip()
                or "source git command failed"
            )
        return result.stdout.strip()

    async def _fetch_from_workspace(
        self,
        candidate: Candidate,
        workspace: GitWorkspace,
        temporary_ref: str,
    ) -> None:
        source_path = workspace.sandbox.host_path(workspace.root)
        if source_path is not None:
            await self._host_git(
                "-c",
                "protocol.file.allow=always",
                "fetch",
                "--force",
                "--no-tags",
                "--no-recurse-submodules",
                str(source_path),
                f"+{candidate.version}:{temporary_ref}",
            )
            return

        export_ref = f"refs/vero/export/{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="vero-candidate-bundle-") as directory:
            local_bundle = Path(directory) / "candidate.bundle"
            async with workspace.sandbox.temporary_directory(
                prefix="vero-candidate-export-"
            ) as remote_directory:
                remote_bundle = str(
                    PurePosixPath(remote_directory) / "candidate.bundle"
                )
                try:
                    await self._source_git(
                        workspace,
                        "update-ref",
                        export_ref,
                        candidate.version,
                        timeout=30,
                    )
                    await self._source_git(
                        workspace,
                        "bundle",
                        "create",
                        remote_bundle,
                        export_ref,
                    )
                    await workspace.sandbox.download(remote_bundle, str(local_bundle))
                    await self._host_git(
                        "fetch",
                        "--force",
                        "--no-tags",
                        str(local_bundle),
                        f"+{export_ref}:{temporary_ref}",
                    )
                finally:
                    try:
                        await asyncio.shield(
                            self._source_git(
                                workspace,
                                "update-ref",
                                "-d",
                                export_ref,
                                timeout=30,
                            )
                        )
                    except Exception:
                        pass

    async def _fetch_from_repository(
        self,
        candidate: Candidate,
        *,
        sandbox: Sandbox,
        repository_path: str,
        temporary_ref: str,
    ) -> None:
        source_path = sandbox.host_path(repository_path)
        if source_path is not None:
            await self._host_git(
                "-c",
                "protocol.file.allow=always",
                "fetch",
                "--force",
                "--no-tags",
                "--no-recurse-submodules",
                str(source_path),
                f"+{candidate.version}:{temporary_ref}",
            )
            return

        export_ref = f"refs/vero/export/{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="vero-candidate-bundle-") as directory:
            local_bundle = Path(directory) / "candidate.bundle"
            async with sandbox.temporary_directory(
                prefix="vero-candidate-export-"
            ) as remote_directory:
                remote_bundle = str(
                    PurePosixPath(remote_directory) / "candidate.bundle"
                )
                try:
                    await self._repository_git(
                        sandbox,
                        repository_path,
                        "update-ref",
                        export_ref,
                        candidate.version,
                        timeout=30,
                    )
                    await self._repository_git(
                        sandbox,
                        repository_path,
                        "bundle",
                        "create",
                        remote_bundle,
                        export_ref,
                    )
                    await sandbox.download(remote_bundle, str(local_bundle))
                    await self._host_git(
                        "fetch",
                        "--force",
                        "--no-tags",
                        str(local_bundle),
                        f"+{export_ref}:{temporary_ref}",
                    )
                finally:
                    try:
                        await asyncio.shield(
                            self._repository_git(
                                sandbox,
                                repository_path,
                                "update-ref",
                                "-d",
                                export_ref,
                                timeout=30,
                            )
                        )
                    except Exception:
                        pass

    async def _persist_import(
        self,
        candidate: Candidate,
        fetch: Callable[[str], Awaitable[None]],
    ) -> Candidate:
        if _OBJECT_ID.fullmatch(candidate.version) is None:
            raise ValueError("Git candidate version must be a full object ID")
        async with self._lock:
            existing = self._candidates.get(candidate.id)
            if existing is not None:
                if existing != candidate:
                    raise ValueError(
                        f"candidate ID {candidate.id!r} is already stored "
                        "with different data"
                    )
                return existing

            temporary_ref = f"refs/vero/incoming/{uuid.uuid4().hex}"
            stable_ref = self._candidate_ref(candidate.id)
            try:
                await fetch(temporary_ref)
                imported = await self._host_git(
                    "rev-parse", "--verify", f"{temporary_ref}^{{commit}}", timeout=30
                )
                if imported != candidate.version:
                    raise CandidateRepositoryError(
                        "imported Git object does not match candidate version"
                    )
                tracked_context = await self._host_git(
                    "ls-tree",
                    "-r",
                    "--name-only",
                    temporary_ref,
                    "--",
                    self._reserved_context_path,
                    timeout=30,
                )
                if tracked_context:
                    raise CandidateRepositoryError(
                        f"candidate tracks reserved agent context path "
                        f"{self._reserved_context_path!r}"
                    )
                await self._host_git(
                    "update-ref", stable_ref, candidate.version, timeout=30
                )
                record = _CandidateRecord(candidate=candidate)
                await asyncio.to_thread(
                    _atomic_write_json,
                    self._record_path(candidate.id),
                    record.model_dump(mode="json"),
                )
                self._candidates[candidate.id] = candidate
                return candidate
            finally:
                try:
                    await asyncio.shield(
                        self._host_git("update-ref", "-d", temporary_ref, timeout=30)
                    )
                except Exception:
                    pass

    async def capture(
        self,
        candidate: Candidate,
        workspace: GitWorkspace,
    ) -> Candidate:
        if not self.supports(workspace):
            raise TypeError(
                "Git candidate repository requires a compatible GitWorkspace"
            )
        if await workspace.is_dirty():
            raise ValueError("candidate workspace must be clean before capture")
        actual = await workspace.current_version()
        if actual != candidate.version:
            raise ValueError(
                f"candidate workspace is at {actual!r}, expected {candidate.version!r}"
            )

        async def fetch(temporary_ref: str) -> None:
            await self._fetch_from_workspace(candidate, workspace, temporary_ref)

        return await self._persist_import(candidate, fetch)

    async def _exclude_agent_context(self, workspace: GitWorkspace) -> None:
        git_directory = await self._source_git(
            workspace,
            "rev-parse",
            "--absolute-git-dir",
            timeout=30,
        )
        exclude_path = str(PurePosixPath(git_directory) / "info" / "exclude")
        pattern = f"/{self._reserved_context_path}/"
        existing = (
            await workspace.sandbox.read_file(exclude_path)
            if await workspace.sandbox.exists(exclude_path)
            else ""
        )
        lines = existing.splitlines()
        if pattern not in lines:
            value = existing
            if value and not value.endswith("\n"):
                value += "\n"
            value += pattern + "\n"
            await workspace.sandbox.write_file(exclude_path, value)

    async def _bundle_candidates(
        self,
        candidates: Sequence[Candidate],
        destination: Path,
    ) -> None:
        refs = [self._candidate_ref(candidate.id) for candidate in candidates]
        await self._host_git("bundle", "create", str(destination), *refs)

    async def materialize_agent_history(
        self,
        candidates: Sequence[Candidate],
        *,
        workspace: GitWorkspace,
        destination: str,
    ) -> None:
        if not self.supports(workspace):
            raise TypeError(
                "Git candidate repository requires a compatible GitWorkspace"
            )
        visible = sorted(candidates, key=lambda item: (item.created_at, item.id))
        for candidate in visible:
            stored = self.get(candidate.id)
            if stored != candidate:
                raise ValueError(
                    f"candidate {candidate.id!r} is not present in durable storage"
                )

        await self._exclude_agent_context(workspace)
        existing_refs = await self._source_git(
            workspace,
            "for-each-ref",
            "--format=%(refname)",
            "refs/vero/context",
            timeout=30,
        )
        for reference in existing_refs.splitlines():
            if reference:
                await self._source_git(
                    workspace,
                    "update-ref",
                    "-d",
                    reference,
                    timeout=30,
                )

        if visible:
            refspecs = [
                f"+{self._candidate_ref(candidate.id)}:{self._context_ref(candidate.id)}"
                for candidate in visible
            ]
            if workspace.sandbox.host_path(workspace.root) is not None:
                await self._source_git(
                    workspace,
                    "-c",
                    "protocol.file.allow=always",
                    "fetch",
                    "--force",
                    "--no-tags",
                    "--no-recurse-submodules",
                    str(self.repository_path),
                    *refspecs,
                )
            else:
                with tempfile.TemporaryDirectory(
                    prefix="vero-context-bundle-"
                ) as directory:
                    local_bundle = Path(directory) / "candidates.bundle"
                    await self._bundle_candidates(visible, local_bundle)
                    async with workspace.sandbox.temporary_directory(
                        prefix="vero-context-import-"
                    ) as remote_directory:
                        remote_bundle = str(
                            PurePosixPath(remote_directory) / "candidates.bundle"
                        )
                        await workspace.sandbox.upload(
                            str(local_bundle),
                            remote_bundle,
                        )
                        await self._source_git(
                            workspace,
                            "fetch",
                            "--force",
                            "--no-tags",
                            "--no-recurse-submodules",
                            remote_bundle,
                            *refspecs,
                        )

        if await workspace.sandbox.exists(destination):
            await workspace.sandbox.remove(destination, recursive=True)
        await workspace.sandbox.mkdir(destination)
        by_id = {candidate.id: candidate for candidate in visible}
        index = []
        for candidate in visible:
            digest = _candidate_digest(candidate.id)
            candidate_dir = str(PurePosixPath(destination) / digest)
            await workspace.sandbox.mkdir(candidate_dir)
            native_ref = self._context_ref(candidate.id)
            await workspace.sandbox.write_file(
                str(PurePosixPath(candidate_dir) / "candidate.json"),
                candidate.model_dump_json(indent=2) + "\n",
            )
            patch_path = None
            parent = by_id.get(candidate.parent_id) if candidate.parent_id else None
            if parent is not None:
                arguments = [
                    "diff",
                    "--binary",
                    parent.version,
                    candidate.version,
                ]
                if self.project_subpath != ".":
                    arguments.extend(["--", self.project_subpath])
                patch = await self._source_git(workspace, *arguments)
                patch_path = f"{digest}/parent.patch"
                await workspace.sandbox.write_file(
                    str(PurePosixPath(destination) / patch_path),
                    patch + ("\n" if patch and not patch.endswith("\n") else ""),
                )
            index.append(
                {
                    "candidate_id": candidate.id,
                    "version": candidate.version,
                    "parent_id": candidate.parent_id,
                    "native_ref": native_ref,
                    "metadata_path": f"{digest}/candidate.json",
                    "parent_patch_path": patch_path,
                }
            )
        await workspace.sandbox.write_file(
            str(PurePosixPath(destination) / "index.json"),
            json.dumps(
                {"schema_version": 1, "candidates": index},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    async def import_candidate(
        self,
        candidate: Candidate,
        *,
        sandbox: Sandbox,
        repository_path: str,
    ) -> Candidate:
        """Import a validated commit without checking out the source repository."""

        async def fetch(temporary_ref: str) -> None:
            await self._fetch_from_repository(
                candidate,
                sandbox=sandbox,
                repository_path=repository_path,
                temporary_ref=temporary_ref,
            )

        return await self._persist_import(candidate, fetch)

    def get(self, candidate_id: str) -> Candidate | None:
        return self._candidates.get(candidate_id)

    def list(self) -> tuple[Candidate, ...]:
        return tuple(
            sorted(
                self._candidates.values(),
                key=lambda candidate: (candidate.created_at, candidate.id),
            )
        )

    async def _bundle_candidate(self, candidate: Candidate, destination: Path) -> None:
        await self._host_git(
            "bundle",
            "create",
            str(destination),
            self._candidate_ref(candidate.id),
        )

    @asynccontextmanager
    async def checkout(
        self,
        candidate: Candidate,
        *,
        sandbox: Sandbox,
        name: str | None = None,
    ) -> AsyncIterator[GitWorkspace]:
        stored = self.get(candidate.id)
        if stored is None:
            raise KeyError(f"unknown candidate: {candidate.id!r}")
        if stored != candidate:
            raise ValueError(
                f"candidate {candidate.id!r} does not match its durable record"
            )

        async with sandbox.temporary_directory(prefix=_safe_prefix(name)) as directory:
            checkout_root = str(PurePosixPath(directory) / "repository")
            result = await sandbox.run(["git", "init", checkout_root], timeout=30)
            if result.returncode != 0:
                raise CandidateRepositoryError(
                    result.stderr or "failed to initialize candidate checkout"
                )

            host_directory = sandbox.host_path(directory)
            if host_directory is not None:
                source = str(self.repository_path)
                ref = self._candidate_ref(candidate.id)
            else:
                with tempfile.TemporaryDirectory(
                    prefix="vero-candidate-checkout-"
                ) as host_temporary:
                    local_bundle = Path(host_temporary) / "candidate.bundle"
                    await self._bundle_candidate(candidate, local_bundle)
                    remote_bundle = str(PurePosixPath(directory) / "candidate.bundle")
                    await sandbox.upload(str(local_bundle), remote_bundle)
                    source = remote_bundle
                    ref = self._candidate_ref(candidate.id)
                    result = await sandbox.run(
                        [
                            "git",
                            "-c",
                            f"safe.directory={checkout_root}",
                            "-C",
                            checkout_root,
                            "fetch",
                            "--force",
                            "--no-tags",
                            source,
                            f"+{ref}:refs/vero/checkout",
                        ],
                        timeout=120,
                    )
                if result.returncode != 0:
                    raise CandidateRepositoryError(
                        result.stderr or "failed to fetch remote candidate checkout"
                    )

            if host_directory is not None:
                result = await sandbox.run(
                    [
                        "git",
                        "-c",
                        "protocol.file.allow=always",
                        "-c",
                        f"safe.directory={checkout_root}",
                        "-C",
                        checkout_root,
                        "fetch",
                        "--force",
                        "--no-tags",
                        source,
                        f"+{ref}:refs/vero/checkout",
                    ],
                    timeout=120,
                )
                if result.returncode != 0:
                    raise CandidateRepositoryError(
                        result.stderr or "failed to fetch candidate checkout"
                    )

            result = await sandbox.run(
                [
                    "git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    f"safe.directory={checkout_root}",
                    "-C",
                    checkout_root,
                    "checkout",
                    "--detach",
                    candidate.version,
                ],
                timeout=60,
            )
            if result.returncode != 0:
                raise CandidateRepositoryError(
                    result.stderr or "failed to check out candidate version"
                )
            project_path = (
                checkout_root
                if self.project_subpath == "."
                else str(PurePosixPath(checkout_root) / self.project_subpath)
            )
            workspace = GitWorkspace(
                sandbox=sandbox,
                root=checkout_root,
                project_path=project_path,
            )
            if await workspace.current_version() != candidate.version:
                raise CandidateRepositoryError(
                    "candidate checkout resolved incorrectly"
                )
            if await workspace.is_dirty():
                raise CandidateRepositoryError(
                    "candidate checkout is unexpectedly dirty"
                )
            try:
                yield workspace
            finally:
                context_path = str(
                    PurePosixPath(workspace.project_path) / _AGENT_CONTEXT_DIRECTORY
                )
                if await workspace.sandbox.exists(context_path):
                    result = await asyncio.shield(
                        workspace.sandbox.run(
                            ["chmod", "-R", "u+w", context_path],
                            timeout=30,
                        )
                    )
                    if result.returncode != 0:
                        raise CandidateRepositoryError(
                            result.stderr
                            or f"failed to unseal agent context {context_path}"
                        )
