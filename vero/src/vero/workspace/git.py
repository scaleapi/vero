from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import AsyncGenerator

from vero.sandbox import Sandbox
from vero.workspace.base import Workspace

logger = logging.getLogger(__name__)

# Commit dates for the automated saves below are pinned to the epoch so that
# committing the same staged tree on the same parent with the same message
# always yields the same sha. Left on the wall clock, a resumed run recommitted
# byte-identical content and got a brand new sha, so it could not recognise the
# candidate its own previous attempt had already built and paid to evaluate.
_PINNED_COMMIT_DATE = "1970-01-01T00:00:00+00:00"


def _basename(path: str) -> str:
    """Extract the last component of a path."""
    return PurePosixPath(path).name


def _parent(path: str) -> str:
    """Extract the parent directory of a path."""
    return str(PurePosixPath(path).parent)


def _join(parent: str, child: str) -> str:
    """Join two path components."""
    return str(PurePosixPath(parent) / child)


class GitWorkspace(Workspace):
    """Workspace backed by git, using sandbox.run() for all git operations.

    Works with any Sandbox implementation — local, Docker, remote.
    All path manipulation uses string operations (no pathlib/os.path)
    so it remains correct for non-local sandboxes.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        root: str,
        project_path: str | None = None,
        name: str | None = None,
        worktree_owner_root: str | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._root = root
        self._project_path = project_path or root
        self._name = name or _basename(root)
        self._worktree_owner_root = worktree_owner_root
        self._lock = asyncio.Lock()

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def root(self) -> str:
        return self._root

    @property
    def project_path(self) -> str:
        return self._project_path

    @property
    def name(self) -> str:
        return self._name

    # ── Git helper ──────────────────────────────────────────────────

    async def _git(self, *args: str, env: dict[str, str] | None = None) -> str:
        """Run a git command via sandbox.run(), returning stdout. Raises on non-zero exit."""
        command = ["git", "-c", f"safe.directory={self._root}", *args]
        if env:
            # Extra variables go through the `env` binary rather than
            # sandbox.run(env=...) because the two Sandbox implementations
            # disagree about what that parameter means: LocalSandbox hands the
            # dict straight to the subprocess and therefore *replaces* the whole
            # environment (losing PATH, HOME and the ambient git configuration),
            # while DockerSandbox adds the entries on top of the container's.
            # Prefixing the command keeps "add these variables" meaning the same
            # thing in every sandbox.
            command = [
                "env",
                *(f"{name}={value}" for name, value in env.items()),
                *command,
            ]
        result = await self._sandbox.run(
            command,
            cwd=self._root,
            # The sandbox's 30 second default is far too short for git over a
            # large candidate tree: a `git add` of a big working copy while the
            # machine was loaded with concurrent trials ran past it, the timeout
            # killed git mid-write, and the .git/index.lock it left behind
            # destroyed that round's candidate with an error that pointed
            # nowhere near the real cause. The tradeoff of 120 seconds is that a
            # git process which is genuinely hung now takes four times as long
            # to fail, which we accept because losing a candidate is worse than
            # waiting.
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
        return result.stdout

    # ── Construction helpers ────────────────────────────────────────

    @classmethod
    async def from_path(
        cls,
        sandbox: Sandbox,
        project_path: str,
    ) -> GitWorkspace:
        """Create a GitWorkspace from a project path.

        Finds the git repo root and determines the project-relative path.
        All resolution happens via sandbox commands, not local path operations.
        """
        project_path = await sandbox.canonicalize(str(project_path))

        result = await sandbox.run(
            [
                "git",
                "-c",
                "safe.directory=*",
                "rev-parse",
                "--show-toplevel",
            ],
            cwd=project_path,
        )
        if result.returncode != 0:
            detail = f": {result.stderr}" if result.stderr else ""
            raise RuntimeError(f"Not a git repository: {project_path}{detail}")
        repo_root = await sandbox.canonicalize(result.stdout.strip())

        # Find the main repo name (handles worktrees whose common dir differs)
        repo_name = _basename(repo_root)
        try:
            result = await sandbox.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={repo_root}",
                    "rev-parse",
                    "--git-common-dir",
                ],
                cwd=project_path,
            )
            if result.returncode == 0:
                git_common_dir = result.stdout.strip()
                # Only use common-dir if it's an absolute path (worktree case).
                # Regular repos return ".git" (relative), which is not useful.
                if git_common_dir.startswith("/"):
                    main_root = _parent(git_common_dir)
                    repo_name = _basename(main_root)
        except RuntimeError:
            pass

        return cls(
            sandbox=sandbox,
            root=repo_root,
            project_path=project_path,
            name=repo_name,
        )

    @classmethod
    async def create(cls, project_path: str) -> GitWorkspace:
        """Create a GitWorkspace with a default local sandbox.

        Convenience factory for non-Policy callers (evaluator, trace analysis,
        etc.) that just need a workspace for git operations without agent
        access control.
        """
        from vero.sandbox import LocalSandbox

        sandbox = await LocalSandbox.create()
        return await cls.from_path(sandbox, str(project_path))

    # ── History ─────────────────────────────────────────────────────

    async def current_version(self) -> str:
        return await self._git("rev-parse", "HEAD")

    async def save(self, message: str = "Save") -> str:
        async with self._lock:
            if not await self.is_dirty():
                return await self.current_version()

            # Stage changes scoped to the project path
            if self._project_path == self._root:
                await self._git("add", "--all")
            else:
                await self._git("add", self._project_path)

            # Commit (skip hooks for automated commits). Both dates have to be
            # pinned, not just the author date: the committer date is part of
            # the commit object too, so leaving it on the wall clock keeps the
            # sha unstable. Note this invalidates existing session directories
            # exactly once, because candidates saved before this change carry
            # wall-clock shas that will never be reproduced again. Candidate
            # ordering does not move: the optimizer stamps Candidate.created_at
            # from datetime.now(UTC) separately, and both the repository listing
            # and the objective tie-break sort on that field, never on a commit
            # date.
            await self._git(
                "-c",
                "user.name=vero",
                "-c",
                "user.email=vero@localhost",
                "commit",
                "-m",
                message,
                "--no-verify",
                env={
                    "GIT_AUTHOR_DATE": _PINNED_COMMIT_DATE,
                    "GIT_COMMITTER_DATE": _PINNED_COMMIT_DATE,
                },
            )

            return await self.current_version()

    async def restore(self, version_id: str, message: str | None = None) -> str:
        async with self._lock:
            message = message or f"Restore to {version_id[:8]}"

            # Checkout files from the target version
            await self._git("checkout", version_id, "--", ".")

            # Stage everything
            await self._git("add", "--all")

            # Only commit if there are actual changes
            try:
                await self._git("diff", "--cached", "--quiet", "--exit-code")
                # No changes — already at that state
            except RuntimeError:
                # There are staged changes — commit them
                await self._git(
                    "-c",
                    "user.name=vero",
                    "-c",
                    "user.email=vero@localhost",
                    "commit",
                    "-m",
                    message,
                    "--no-verify",
                )

            return await self.current_version()

    # ── History inspection ──────────────────────────────────────────

    async def diff(
        self, from_version: str | None = None, to_version: str | None = None
    ) -> str:
        args = ["diff"]
        if from_version:
            args.append(from_version)
        if to_version:
            args.append(to_version)
        return await self._git(*args)

    async def log(self, max_count: int = 10, since_version: str | None = None) -> str:
        args = ["log", f"-n{max_count}", "--pretty=format:%h - %s (%cr) <%an>"]
        if since_version:
            args.append(f"{since_version}..HEAD")
        return await self._git(*args)

    async def is_ancestor(self, version_a: str, version_b: str) -> bool:
        try:
            await self._git("merge-base", "--is-ancestor", version_a, version_b)
            return True
        except RuntimeError:
            return False

    def _copied_project_path(self, copied_root: str) -> str:
        relative = PurePosixPath(self._project_path).relative_to(self._root)
        if str(relative) == ".":
            return copied_root
        return _join(copied_root, str(relative))

    # ── Copies ──────────────────────────────────────────────────────

    async def _remove_worktree(self, target_path: str) -> None:
        result = await self._sandbox.run(
            ["git", "worktree", "remove", "--force", target_path],
            cwd=self._root,
        )
        if result.returncode != 0 and await self._sandbox.exists(target_path):
            await self._sandbox.remove(target_path, recursive=True)
        await self._sandbox.run(
            ["git", "worktree", "prune"],
            cwd=self._root,
        )

    async def _add_worktree(self, target_path: str, from_version: str | None) -> None:
        if await self._sandbox.exists(target_path):
            raise FileExistsError(target_path)
        arguments = ["worktree", "add", "--detach", target_path]
        if from_version is not None:
            arguments.append(from_version)
        try:
            await self._git(*arguments)
        except BaseException:
            await asyncio.shield(self._remove_worktree(target_path))
            raise

    async def copy(
        self, name: str | None = None, from_version: str | None = None
    ) -> GitWorkspace:
        """Create a new git worktree as an isolated copy."""
        async with self._lock:
            if name is None:
                name = f"worktree-{uuid.uuid4().hex[:8]}"

            target_path = _join(_parent(self._root), name)
            await self._add_worktree(target_path, from_version)

            return GitWorkspace(
                sandbox=self._sandbox,
                root=target_path,
                project_path=self._copied_project_path(target_path),
                name=self._name,
                worktree_owner_root=self._root,
            )

    @asynccontextmanager
    async def temp_copy(
        self, from_version: str | None = None
    ) -> AsyncGenerator[GitWorkspace, None]:
        """Create a temporary worktree, cleaned up on exit."""
        copy_name = f"tmp-{uuid.uuid4().hex[:8]}"

        # Ask the sandbox for a temp directory
        result = await self._sandbox.run(["mktemp", "-d"])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create temp dir: {result.stderr}")
        temporary_root = result.stdout.strip()
        target_path = _join(temporary_root, copy_name)

        try:
            async with self._lock:
                await self._add_worktree(target_path, from_version)
        except BaseException:
            await asyncio.shield(self._sandbox.remove(temporary_root, recursive=True))
            raise

        target_path = await self._sandbox.canonicalize(target_path)

        copied = GitWorkspace(
            sandbox=self._sandbox,
            root=target_path,
            project_path=self._copied_project_path(target_path),
            name=self._name,
            worktree_owner_root=self._root,
        )

        try:
            yield copied
        finally:
            await asyncio.shield(copied.destroy())
            await asyncio.shield(self._sandbox.remove(temporary_root, recursive=True))

    # ── Execution at a version ──────────────────────────────────────

    @asynccontextmanager
    async def at(self, version_id: str) -> AsyncGenerator[None, None]:
        """Temporarily checkout a version, restore previous state on exit."""
        async with self._lock:
            # Remember current state
            try:
                previous_branch = await self._git("symbolic-ref", "--short", "HEAD")
            except RuntimeError:
                previous_branch = None
            previous_commit = await self.current_version()

            try:
                await self._git("checkout", version_id)
                yield
            finally:
                if previous_branch:
                    await self._git("checkout", previous_branch)
                else:
                    await self._git("checkout", previous_commit)

    # ── Optional ────────────────────────────────────────────────────

    async def is_dirty(self) -> bool:
        status = await self._git("status", "--porcelain", self._project_path)
        return bool(status)

    async def destroy(self) -> None:
        """Remove this worktree."""
        if self._worktree_owner_root is None:
            return
        async with self._lock:
            owner = GitWorkspace(
                sandbox=self._sandbox,
                root=self._worktree_owner_root,
            )
            await owner._remove_worktree(self._root)
            self._worktree_owner_root = None

    # ── Git-specific helpers (used by Policy and git tools) ─────────

    async def resolve_ref(self, ref: str) -> str:
        """Resolve a git ref (branch, tag, short hash) to a full commit hash."""
        return await self._git("rev-parse", ref)

    async def current_branch(self) -> str | None:
        """Return current branch name, or None if detached HEAD."""
        try:
            return await self._git("symbolic-ref", "--short", "HEAD")
        except RuntimeError:
            return None

    async def branch_exists(self, branch_name: str) -> bool:
        try:
            await self._git("rev-parse", "--verify", f"refs/heads/{branch_name}")
            return True
        except RuntimeError:
            return False

    async def get_head_commit(self, branch_name: str) -> str:
        return await self._git("rev-parse", f"refs/heads/{branch_name}")

    async def checkout_branch(
        self, branch_name: str, create: bool = False, from_ref: str | None = None
    ) -> None:
        async with self._lock:
            args = ["checkout"]
            if create:
                args.append("-b")
            args.append(branch_name)
            if from_ref:
                args.append(from_ref)
            await self._git(*args)

    async def maybe_fetch(self) -> None:
        """Fetch from remote if one exists."""
        try:
            await self._git("remote", "get-url", "origin")
            await self._git("fetch", "--all")
        except RuntimeError:
            pass
