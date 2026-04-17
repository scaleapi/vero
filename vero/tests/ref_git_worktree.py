from __future__ import annotations

import asyncio
import logging
import os
import threading
import weakref
from contextlib import asynccontextmanager
from os import PathLike
from pathlib import Path
from typing import Any, AsyncGenerator, ClassVar

import yaml
from git import InvalidGitRepositoryError, NoSuchPathError, Repo
from rich.panel import Panel
from rich.syntax import Syntax

from vero.core.utils import is_valid_folder_name
from vero.logging import setup_console
from vero.utils.general import normalize_dash_underscore, random_readable_id

logger = logging.getLogger(__name__)
console = setup_console()


class GitWorktree:
    """Representation of a Git worktree with singleton-per-path semantics."""

    __slots__ = (
        "repo",
        "project_path",
        "sync",
        "_lock",
        "_main_branch",
        "_initialized",
        "__weakref__",
    )

    _instances: ClassVar[weakref.WeakValueDictionary[Path, "GitWorktree"]] = (
        weakref.WeakValueDictionary()
    )
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, repo: Repo, project_path: Path, **kwargs) -> "GitWorktree":
        """Create a new GitWorktree instance. Ensures only one instance per worktree path."""
        assert repo.working_tree_dir, "Repo has not worktree!"
        worktree_path = Path(repo.working_tree_dir).resolve()

        with cls._instances_lock:
            if worktree_path in cls._instances:
                existing = cls._instances[worktree_path]
                if existing.project_path.resolve() != Path(project_path).resolve():
                    raise ValueError(
                        f"GitWorktree for {worktree_path} already exists with different project_path"
                    )
                logger.info(
                    f"[dim]Reusing existing GitWorktree instance for path [/dim] [yellow]{worktree_path}[/yellow]"
                )
                return existing

            instance = object.__new__(cls)
            cls._instances[worktree_path] = instance
            logger.info(
                f"[dim]Instantiated new GitWorktree instance for path [/dim] [yellow]{worktree_path}[/yellow]"
            )
            return instance

    def __init__(self, repo: Repo, project_path: Path, sync: bool = False):
        if getattr(self, "_initialized", False):
            return

        self.repo = repo
        self.project_path = self._validate_project_path(
            project_path, self.worktree_path
        )
        self.sync = sync
        self._lock = asyncio.Lock()
        self._main_branch = GitWorktree.infer_main_branch(repo)
        self._initialized = True

        # Set default author/committer for all commits
        with self.repo.config_writer() as config:
            config.set_value("user", "name", "vero")
            config.set_value("user", "email", "vero@localhost")

    def __repr__(self) -> str:
        return f"GitWorktree(worktree_path={self.worktree_path.as_posix()}, project_path={self.project_path.as_posix()}, sync={self.sync})"

    def __hash__(self) -> int:
        return hash(self.worktree_path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GitWorktree):
            return False
        return self.worktree_path == other.worktree_path

    # -------------------------------------------------------------------------
    # Static andClass methods for instance management
    # -------------------------------------------------------------------------

    @staticmethod
    def infer_main_branch(repo: Repo) -> str:
        """Best-effort detection of the main branch of a repo."""
        try:
            origin = repo.remotes.origin
            ref = origin.refs["HEAD"]
            return ref.reference.name.split("/", 1)[1]
        except Exception:
            for name in ["main", "master", "develop"]:
                if name in repo.heads:
                    return name
            raise FileNotFoundError("Main branch could not be auto-detected!")

    @classmethod
    def get(cls, worktree_path: Path) -> "GitWorktree | None":
        """Get existing instance by path, or None."""
        return cls._instances.get(Path(worktree_path).resolve())

    @classmethod
    def remove(cls, worktree_path: Path) -> bool:
        """Remove instance from registry (e.g., when deleting worktree)."""
        with cls._instances_lock:
            return cls._instances.pop(Path(worktree_path).resolve(), None) is not None

    @classmethod
    def _validate_project_path(cls, project_path: Path, worktree_path: Path) -> Path:
        worktree_path = Path(worktree_path).resolve()
        project_path = Path(project_path)

        if not project_path.is_absolute():
            logger.info(
                f"Obtained relative project path {project_path}, resolving to absolute path with respect to worktree path {worktree_path}..."
            )
            project_path = (worktree_path / project_path).resolve()
            logger.info(f"Resolved project path to {project_path}")

        assert project_path.exists(), f"Project path {project_path} does not exist!"
        assert project_path.is_dir(), f"Project path {project_path} is not a directory!"

        try:
            Path(project_path).resolve().relative_to(Path(worktree_path).resolve())
        except ValueError as e:
            raise ValueError(
                f"Project path {project_path} is not a subfolder of worktree path {worktree_path}!"
            ) from e

        return project_path

    @classmethod
    def from_local_path(
        cls,
        project_path: Path | str | None = None,
        *,
        worktree_path: Path | str | None = None,
        sync: bool = False,
    ) -> GitWorktree:
        """Helper to initialize a worktree from a local repository path."""
        if worktree_path is not None:
            worktree_path = Path(worktree_path).resolve()
            repo = Repo(worktree_path)

            if project_path is None:
                project_path = worktree_path

            if not Path(project_path).is_absolute():
                project_path = (worktree_path / project_path).resolve()

            return cls(repo=repo, project_path=Path(project_path), sync=sync)
        else:
            if project_path is None:
                raise ValueError(
                    "project_path is required when worktree_path is not provided."
                )

            project_path = Path(project_path).resolve()
            repo = Repo(project_path, search_parent_directories=True)
            return cls(repo=repo, project_path=project_path, sync=sync)

    @classmethod
    def from_remote_url(
        cls,
        remote_url: str,
        worktree_path: PathLike[str],
        project_path: PathLike[str] | None = None,
        sync: bool = False,
    ) -> GitWorktree:
        """Helper to initialize a worktree from a remote URL."""
        worktree_path = Path(worktree_path).resolve()

        if not worktree_path.exists():
            if not is_valid_folder_name(worktree_path.name):
                raise ValueError(
                    f"Destination {worktree_path.name} is not a valid folder name!"
                )
            worktree_path.mkdir(parents=True, exist_ok=True)

        if not worktree_path.is_dir():
            raise ValueError(f"Destination {worktree_path} is not a directory!")

        repo = None

        try:
            repo = Repo(worktree_path)
            if repo.remotes.origin.url == remote_url:
                logger.info(
                    f"[dim] Repo to clone is already at {worktree_path}! [/dim]"
                )
            else:
                raise ValueError(
                    f"Repo at {worktree_path} is not the same as the URL {remote_url}! Cannot clone to it."
                )
        except (InvalidGitRepositoryError, NoSuchPathError):
            pass

        if repo is None:
            if any(worktree_path.iterdir()):
                raise ValueError(
                    f"Destination {worktree_path} not empty! Cannot clone here."
                )

            repo = Repo.clone_from(remote_url, worktree_path)
            logger.info(
                f"[bold green] Cloned repo[/bold green] from [cyan]{remote_url}[/cyan] to [yellow]{worktree_path}[/yellow]"
            )

        if project_path is None:
            project_path = worktree_path

        if not Path(project_path).is_absolute():
            project_path = (worktree_path / project_path).resolve()

        return cls(repo=repo, project_path=Path(project_path), sync=sync)

    # -------------------------------------------------------------------------
    # Locking
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def locked(
        self, caller: str | None = None
    ) -> AsyncGenerator[GitWorktree, None]:
        """Acquire exclusive access to this worktree for operations that alter Git repository state."""
        async with self._lock:
            logger.info(
                f"[dim]{caller} acquired lock for GitWorktree[/dim] [yellow]{self.worktree_path}[/yellow]"
            )
            yield self

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def main_branch(self) -> str:
        """The main branch of the repo."""
        if self._main_branch is None:
            self._main_branch = GitWorktree.infer_main_branch(self.repo)
        return self._main_branch

    @main_branch.setter
    def main_branch(self, value: str) -> None:
        assert self.branch_exists(value), "The main branch does not exist in the repo!"
        self._main_branch = value

    @property
    def worktree_path(self) -> Path:
        """The path to the worktree."""
        assert self.repo.working_tree_dir, "Repo has no worktree!"
        return Path(self.repo.working_tree_dir).resolve()

    @property
    def project_relative_path(self) -> Path:
        """The relative path to the project from the worktree."""
        return self.project_path.relative_to(self.worktree_path)

    @property
    def main_worktree_path(self) -> Path:
        """The path to the primary worktree."""
        git_object_dir = (
            Path(self.repo.common_dir).resolve().as_posix().removesuffix(".git")
        )
        return Path(git_object_dir).resolve()

    @property
    def is_main_worktree(self) -> bool:
        """Whether the worktree is the primary worktree."""
        return self.worktree_path == self.main_worktree_path

    @property
    def worktree_name(self) -> str:
        """The name of the current worktree."""
        return self.worktree_path.name

    @property
    def repo_name(self) -> str:
        """The name of the repo, i.e. the name of the primary worktree."""
        return self.main_worktree_path.name

    @property
    def remote_url(self) -> str | None:
        try:
            return self.repo.remotes.origin.url
        except AttributeError:
            return None

    @property
    def http_url(self) -> str | None:
        if self.remote_url is None:
            return None
        return self.remote_url.replace(
            "git@github.com:", "https://github.com/"
        ).removesuffix(".git")

    # -------------------------------------------------------------------------
    # Read-only operations (no lock needed)
    # -------------------------------------------------------------------------

    def as_dict(self) -> dict[str, str | bool | None]:
        """Returns a dictionary representation of the GitWorktree."""
        return {
            "worktree_path": self.worktree_path.as_posix(),
            "project_path": self.project_path.as_posix(),
            "branch": self.current_branch(),
            "commit": self.current_commit(),
            "is_detached": self.current_branch() is None,
            "is_main_worktree": self.is_main_worktree,
        }

    def render(self) -> Panel:
        """Render the GitWorktree as a Rich Panel with YAML."""
        return Panel(
            Syntax(
                yaml.dump(self.as_dict()), "yaml", theme="monokai", line_numbers=False
            ),
            title="[bold green]GitWorktree[/bold green]",
            border_style="green",
            padding=(1, 2),
        )

    def current_branch(self) -> str | None:
        """Gets the currently active branch.

        Returns:
            The name of the currently active branch.
            None if the repository is in a detached HEAD state.
        """
        try:
            active_branch = self.repo.active_branch
        except TypeError as e:
            if "HEAD is a detached symbolic reference" in str(e):
                return None
            raise e
        return active_branch.name

    def current_commit(self) -> str:
        return self.repo.head.commit.hexsha

    def operates_on_full_repo(self) -> bool:
        return self.project_path == self.worktree_path

    def remote_exists(self, remote_name: str = "origin") -> bool:
        return remote_name in [r.name for r in self.repo.remotes]

    def list_branches(self) -> list[str]:
        return [branch.name for branch in self.repo.branches]

    def branch_exists(self, branch_name: str) -> bool:
        """Checks if a branch exists in the repo."""
        return branch_name in [branch.name for branch in self.repo.branches]

    def get_head_commit(self, branch_name: str) -> str:
        """Gets the commit hash of the head of a branch."""
        return self.repo.heads[branch_name].commit.hexsha

    def get_project_status(self) -> str:
        """Returns the status of files in the project path if specified, otherwise repo status."""
        if self.operates_on_full_repo():
            return self.repo.git.status()
        return self.repo.git.status(self.project_relative_path.as_posix())

    def list_project_modified_files(self, untracked_files: bool = True) -> set[str]:
        """Checks if the project has uncommitted changes to tracked files or untracked files."""
        project_relative_path: str = self.project_relative_path.as_posix()

        project_modified = [
            item.a_path
            for item in self.repo.index.diff(None)
            if str(item.a_path).startswith(project_relative_path)
        ]
        project_deleted = [
            item.a_path
            for item in self.repo.index.diff(None, staged=False)
            if str(item.a_path).startswith(project_relative_path)
        ]

        project_untracked = []
        if untracked_files:
            project_untracked = [
                file
                for file in self.repo.untracked_files
                if file.startswith(project_relative_path)
            ]

        modified = project_modified + project_deleted + project_untracked
        return set([file for file in modified if file])

    def is_project_dirty(self, untracked_files: bool = True) -> bool:
        """Checks if the project has any uncommitted changes to tracked files or untracked files."""
        return (
            len(self.list_project_modified_files(untracked_files=untracked_files)) > 0
        )

    def is_dirty(self, untracked_files: bool = True) -> bool:
        """Checks if there are any uncommitted changes in the repo."""
        return self.repo.is_dirty(untracked_files=untracked_files)

    def list_worktrees(self) -> dict[Path, dict[str, Any]]:
        """Lists all worktrees of the repo."""
        lines = [line.split() for line in self.repo.git.worktree("list").split("\n")]
        worktrees: dict[Path, dict[str, Any]] = {}

        def strip_branch_name(name: str | None) -> str | None:
            if name:
                return name.strip().removeprefix("[").removesuffix("]")
            return None

        for line in lines:
            if len(line) == 3:
                path, commit, branch_name = line
            else:
                path, commit, *_ = line
                branch_name = None

            branch_name = strip_branch_name(branch_name)
            path = Path(path).resolve()
            worktrees[path] = {
                "commit": commit,
                "branch_name": branch_name,
                "is_detached": branch_name is None,
            }
        return worktrees

    def view_diff(
        self,
        from_commit: str | None = None,
        to_commit: str | None = None,
        on_github: bool = False,
    ) -> str:
        """Gets a diff between any two commits."""
        from_commit = from_commit or self.get_head_commit(self.main_branch)
        to_commit = to_commit or self.current_commit()

        if on_github:
            assert self.http_url is not None, "No HTTP URL found for the repo."
            return os.path.join(
                self.http_url, "compare", f"{from_commit}...{to_commit}"
            )

        return self.repo.git.diff(from_commit, to_commit)

    # -------------------------------------------------------------------------
    # Write operations (caller should use `async with worktree.locked()` for atomicity)
    # -------------------------------------------------------------------------

    def fetch(self) -> None:
        """Fetches all branches from the remote."""
        logger.info(
            f"[dim]Fetching all branches from remote[/dim] [yellow]{self.worktree_path}[/yellow]"
        )
        self.repo.git.fetch("--all")

    def maybe_fetch(self) -> bool:
        """Fetches all branches from the remote if the remote exists."""
        if self.remote_exists():
            self.fetch()
            return True
        return False

    def checkout_branch(
        self,
        branch_name: str,
        from_: str | None = None,
        maybe_create: bool = False,
    ) -> None:
        """Checks out a branch.

        Args:
            branch_name: The name of the branch to checkout.
            from_: The commit or branch name to checkout from.
            maybe_create: Whether to create the branch if it doesn't exist.
        """
        if maybe_create and not self.branch_exists(branch_name):
            self.repo.git.checkout("-b", branch_name, from_)
            logger.info(
                f"[dim]Created branch[/dim] [yellow]{branch_name}[/yellow] from [cyan]{from_}[/cyan]"
            )
        else:
            self.repo.git.checkout(branch_name)

    def checkout_commit(self, commit_hash: str) -> None:
        """Checks out a commit (detached HEAD)."""
        self.repo.git.checkout(commit_hash)

    def delete_branch(self, branch_name: str, force: bool = False) -> None:
        """Deletes a branch."""
        if force:
            self.repo.git.branch("-D", branch_name)
        else:
            self.repo.git.branch("-d", branch_name)

    def commit_files(
        self, files: list[str], message: str, skip_hooks: bool = True
    ) -> str:
        """Commits a list of files and returns the commit hash."""
        if not files:
            raise ValueError("No files to commit!")
        self.repo.git.add(files)
        self.repo.index.commit(message, skip_hooks=skip_hooks)
        return self.repo.head.commit.hexsha

    def commit_all(
        self, message: str, project_only: bool = True, skip_hooks: bool = True
    ) -> str:
        """Commits all changes and returns the commit hash."""
        if self.operates_on_full_repo() or not project_only:
            self.repo.git.add(all=True)
        else:
            self.repo.git.add(self.project_relative_path.as_posix())

        if self.repo.index.diff("HEAD") or self.repo.untracked_files:
            self.repo.index.commit(message, skip_hooks=skip_hooks)

        logger.info(
            f"[dim]Committed changes[/dim] [yellow]{message}[/yellow] at commit [cyan]{self.current_commit()}[/cyan]"
        )
        return self.repo.head.commit.hexsha

    def reset_to_commit(self, commit_hash: str) -> None:
        """Resets the current branch to a specific commit (hard reset).

        WARNING: This discards commit history. Prefer restore_to_commit for traceable rollbacks.
        """
        logger.info(f"[dim]Resetting to commit[/dim] [yellow]{commit_hash}[/yellow]")
        self.repo.git.reset("--hard", commit_hash)

    def restore_to_commit(
        self, commit_hash: str, message: str | None = None, skip_hooks: bool = True
    ) -> str:
        """Restores the working tree to match a previous commit, preserving history.

        Creates a new commit with the file state from the target commit.
        This is safer than reset_to_commit because it maintains the full commit history.

        Args:
            commit_hash: The commit to restore the file state from
            message: Optional commit message (defaults to "Restore to <commit_hash>")
            skip_hooks: Whether to skip git hooks
        Returns:
            The new commit hash
        """
        message = message or f"Restore to {commit_hash[:8]}"

        # Checkout all files from the target commit into the working tree
        self.repo.git.checkout(commit_hash, "--", ".")

        # Stage all changes
        self.repo.git.add(all=True)

        # Only commit if there are actual changes
        if self.repo.index.diff("HEAD"):
            self.repo.index.commit(message, skip_hooks=skip_hooks)
            logger.info(
                f"[dim]Restored to commit[/dim] [yellow]{commit_hash}[/yellow] "
                f"[dim]with new commit[/dim] [cyan]{self.current_commit()}[/cyan]"
            )
        else:
            logger.info(
                f"[dim]No changes needed to restore to[/dim] [yellow]{commit_hash}[/yellow] "
                f"[dim](already at that state)[/dim]"
            )

        return self.current_commit()

    def push(self, branch_name: str | None = None) -> None:
        """Pushes a branch to the remote."""
        branch_name = branch_name or self.current_branch()
        if branch_name is None:
            raise ValueError(
                "Cannot push from detached HEAD without specifying branch_name"
            )
        logger.info(
            f"[dim]Pushing branch[/dim] [yellow]{branch_name}[/yellow] to remote"
        )
        self.repo.git.push("origin", branch_name)

    def pull(self, branch_name: str | None = None) -> None:
        """Pulls a branch from the remote."""
        branch_name = branch_name or self.current_branch()
        if branch_name is None:
            raise ValueError(
                "Cannot pull to detached HEAD without specifying branch_name"
            )
        logger.info(
            f"[dim]Pulling branch[/dim] [yellow]{branch_name}[/yellow] from remote"
        )
        self.repo.git.pull("origin", branch_name)

    # -------------------------------------------------------------------------
    # Worktree management
    # -------------------------------------------------------------------------

    def get_random_worktree_name(self) -> str:
        """Gets a random worktree path based on this worktree."""
        return f"{self.repo_name}-{random_readable_id()}"

    def get_random_worktree_path(self) -> Path:
        """Gets a random worktree path based on this worktree."""
        return self.worktree_path.parent / self.get_random_worktree_name()

    def add_worktree(
        self,
        target_path: Path | str,
        branch_name: str | None = None,
        from_: str | None = None,
        sync: bool = False,
    ) -> GitWorktree:
        """Creates a new worktree from this one and returns it."""
        target_path = Path(target_path).resolve().as_posix()

        if branch_name is not None and self.branch_exists(branch_name):
            assert from_ is None, "Cannot specify from_ when the branch already exists!"
            self.repo.git.worktree("add", target_path, branch_name)
        elif branch_name is not None:
            self.repo.git.worktree("add", target_path, "-b", branch_name, from_)
        else:
            self.repo.git.worktree("add", target_path, from_)

        worktree = GitWorktree.from_local_path(
            worktree_path=target_path,
            project_path=self.project_relative_path,
            sync=sync,
        )
        logger.info(
            f"[dim]Created worktree[/dim] [yellow]{worktree.worktree_path}[/yellow]"
        )
        return worktree

    def quick_spawn(
        self,
        branch_name: str | None = None,
        from_: str | None = None,
        sync: bool = False,
    ) -> GitWorktree:
        """Creates a new worktree from this one and returns it."""
        branch_name = branch_name or self.get_random_worktree_name()
        target_path = self.worktree_path.parent / normalize_dash_underscore(branch_name)
        worktree = self.add_worktree(target_path, branch_name, from_, sync=sync)
        return worktree

    def remove_worktree(self, force: bool = False) -> bool:
        """Removes this worktree from the repo (cannot be the main worktree).

        Args:
            force: If True, removes the worktree even if it has modified or untracked files.
        """
        assert not self.is_main_worktree, "Cannot remove the main worktree!"
        if force:
            self.repo.git.worktree("remove", "--force", self.worktree_path.as_posix())
        else:
            self.repo.git.worktree("remove", self.worktree_path.as_posix())
        GitWorktree.remove(self.worktree_path)
        logger.info(
            f"[dim]Removed worktree[/dim] [yellow]{self.worktree_path}[/yellow]"
        )
        return True

    def remove_all_worktrees(
        self, remove_branches: bool = False, force: bool = False
    ) -> dict[str, list[str]]:
        """Removes all worktrees from the repo.

        Args:
            remove_branches: Whether to remove the branches of the worktrees.
            force: If True, removes worktrees even if they have modified or untracked files.

        Returns:
            A dictionary containing the worktrees and branches removed.
        """

        if not self.is_main_worktree:
            raise ValueError("Cannot remove all worktrees from a non-main worktree!")

        removed = {"worktrees": [], "branches": []}
        for worktree_path, worktree_info in self.list_worktrees().items():
            if worktree_path == self.worktree_path:
                continue

            worktree = GitWorktree.from_local_path(worktree_path=worktree_path)
            worktree.remove_worktree(force=force)
            removed["worktrees"].append(worktree_path)
            branch_name = worktree_info["branch_name"]

            if (
                remove_branches
                and branch_name is not None
                and self.branch_exists(branch_name)
                and not branch_name == self.main_branch
            ):
                self.delete_branch(branch_name, force=force)
                logger.info(f"[dim]Removed branch[/dim] [yellow]{branch_name}[/yellow]")
                removed["branches"].append(branch_name)

        return removed

    # -------------------------------------------------------------------------
    # Context managers
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def switch_to_commit(self, commit_hash: str) -> AsyncGenerator[None, None]:
        """Context manager that switches to a commit and restores the previous state on exit."""
        async with self._lock:
            previous_commit = self.current_commit()
            previous_branch = self.current_branch()

            if previous_branch is None:
                msg_suffix = (
                    f"[dim]from detached HEAD[/dim] [yellow]{previous_commit}[/yellow]"
                )
            else:
                msg_suffix = f"[dim]from branch[/dim] [yellow]{previous_branch}[/yellow] at commit [cyan]{previous_commit}[/cyan]"

            logger.info(
                f"[dim]Switching to commit[/dim] [cyan]{commit_hash}[/cyan] {msg_suffix}"
            )
            try:
                self.checkout_commit(commit_hash)
                yield
            finally:
                if previous_branch is not None:
                    self.checkout_branch(previous_branch)
                    logger.info(
                        f"[dim]Switched back to branch[/dim] [yellow]{previous_branch}[/yellow] at commit [cyan]{self.current_commit()}[/cyan]"
                    )
                else:
                    self.checkout_commit(previous_commit)
                    logger.info(
                        f"[dim]Switched back to detached HEAD[/dim] at commit [cyan]{self.current_commit()}[/cyan]"
                    )

    @asynccontextmanager
    async def in_new_worktree(
        self,
        target_path: Path | str | None = None,
        branch_name: str | None = None,
        from_: str | None = None,
        sync: bool = True,
    ) -> AsyncGenerator[GitWorktree, None]:
        """Context manager that creates a temporary worktree and removes it on exit."""
        if target_path is None:
            target_path = self.get_random_worktree_path()

        new_worktree = None
        try:
            new_worktree = self.add_worktree(target_path, branch_name, from_, sync=sync)
            yield new_worktree
        finally:
            if new_worktree is not None:
                new_worktree.remove_worktree()
