"""Tests for file write tools."""

import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from git import Repo
from vero.exceptions import (
    AccessDeniedError,
    FileNotTrackedError,
    InputTooLongError,
    StringNotFoundError,
)
from vero.filesystem import AccessRule, AccessType
from vero.sandbox import LocalSandbox
from vero.tools.file_write import FileWrite
from vero.workspace.git import GitWorkspace


@contextmanager
def temp_git_repo():
    """Create a temporary git repository for testing.

    Yields a tuple of (repo, test_dir) where:
    - repo: The git.Repo instance
    - test_dir: Path to the repository root directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir).resolve()

        # Initialize git repo
        repo = Repo.init(test_dir)
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Create initial files and commit them so they're tracked
        (test_dir / "existing.txt").write_text("original content\n")
        (test_dir / "code.py").write_text("def hello():\n    print('world')\n")

        # Create subdirectories with tracked files
        (test_dir / "src").mkdir()
        (test_dir / "src" / "app.py").write_text("# App code\nimport os\n")

        (test_dir / "private").mkdir()
        (test_dir / "private" / "secret.txt").write_text("secret data\n")

        # Add and commit all files
        repo.index.add(
            [
                "existing.txt",
                "code.py",
                "src/app.py",
                "private/secret.txt",
            ]
        )
        repo.index.commit("Initial commit")

        yield repo, test_dir


class TestFileWrite:
    """Tests for FileWrite tool."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    @pytest.fixture
    def file_write(self, git_repo):
        """Create a FileWrite instance with full access."""
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[AccessRule(AccessType.WRITE, "**")], default_access=AccessType.WRITE)
        return FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

    @pytest.mark.asyncio
    async def test_create_new_file(self, file_write, git_repo):
        """Test creating a new file."""
        _, test_dir = git_repo

        result = await file_write.write_file(
            commit_message="Add new file",
            file_path=str(test_dir / "new_file.txt"),
            content="hello world\n",
        )

        assert "Created" in result or "commit" in result.lower()
        assert (test_dir / "new_file.txt").exists()
        assert (test_dir / "new_file.txt").read_text() == "hello world\n"

    @pytest.mark.asyncio
    async def test_overwrite_tracked_file(self, file_write, git_repo):
        """Test overwriting an existing tracked file."""
        _, test_dir = git_repo

        result = await file_write.write_file(
            commit_message="Update existing file",
            file_path=str(test_dir / "existing.txt"),
            content="new content\n",
        )

        assert "commit" in result.lower()
        assert (test_dir / "existing.txt").read_text() == "new content\n"

    @pytest.mark.asyncio
    async def test_overwrite_untracked_file_fails(self, git_repo):
        """Test that overwriting an untracked file raises FileNotTrackedError."""
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[AccessRule(AccessType.WRITE, "**")], default_access=AccessType.WRITE)
        file_write = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

        # Create an untracked file
        (test_dir / "untracked.txt").write_text("untracked content\n")

        with pytest.raises(FileNotTrackedError):
            await file_write.write_file(
                commit_message="Overwrite untracked",
                file_path=str(test_dir / "untracked.txt"),
                content="new content\n",
            )

    @pytest.mark.asyncio
    async def test_create_file_in_new_directory(self, file_write, git_repo):
        """Test creating a file in a non-existent directory creates parent dirs."""
        _, test_dir = git_repo

        result = await file_write.write_file(
            commit_message="Add file in new dir",
            file_path=str(test_dir / "new_dir" / "nested" / "file.txt"),
            content="nested content\n",
        )

        assert "commit" in result.lower()
        assert (test_dir / "new_dir" / "nested" / "file.txt").exists()
        assert (test_dir / "new_dir" / "nested" / "file.txt").read_text() == "nested content\n"

    @pytest.mark.asyncio
    async def test_content_too_long(self, file_write, git_repo):
        """Test that content exceeding limit raises InputTooLongError."""
        _, test_dir = git_repo

        # Create a FileWrite with a small limit
        repo, _ = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[AccessRule(AccessType.WRITE, "**")], default_access=AccessType.WRITE)
        file_write_limited = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
            content_char_limit=100,
        )

        with pytest.raises(InputTooLongError):
            await file_write_limited.write_file(
                commit_message="Too long",
                file_path=str(test_dir / "large.txt"),
                content="x" * 200,
            )

    @pytest.mark.asyncio
    async def test_write_to_directory_fails(self, file_write, git_repo):
        """Test that writing to a directory path fails.

        Note: Directories are not tracked by git, so FileNotTrackedError is raised
        before the IsADirectoryError check.
        """
        _, test_dir = git_repo

        with pytest.raises(FileNotTrackedError):
            await file_write.write_file(
                commit_message="Write to dir",
                file_path=str(test_dir / "src"),
                content="content\n",
            )

    @pytest.mark.asyncio
    async def test_edit_file_single_replacement(self, file_write, git_repo):
        """Test editing a file with single replacement."""
        _, test_dir = git_repo

        result = await file_write.edit_file(
            commit_message="Edit file",
            file_path=str(test_dir / "code.py"),
            old_string="world",
            new_string="universe",
        )

        assert "commit" in result.lower()
        assert "1" in result  # 1 replacement
        content = (test_dir / "code.py").read_text()
        assert "universe" in content
        assert "world" not in content

    @pytest.mark.asyncio
    async def test_edit_file_replace_all(self, git_repo):
        """Test editing a file with replace_all=True."""
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[AccessRule(AccessType.WRITE, "**")], default_access=AccessType.WRITE)
        file_write = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

        # Create a file with multiple occurrences and track it
        (test_dir / "multi.txt").write_text("foo bar foo baz foo\n")
        repo.index.add(["multi.txt"])
        repo.index.commit("Add multi.txt")

        result = await file_write.edit_file(
            commit_message="Replace all foo",
            file_path=str(test_dir / "multi.txt"),
            old_string="foo",
            new_string="qux",
            replace_all=True,
        )

        assert "3" in result  # 3 replacements
        content = (test_dir / "multi.txt").read_text()
        assert content == "qux bar qux baz qux\n"

    @pytest.mark.asyncio
    async def test_edit_file_string_not_found(self, file_write, git_repo):
        """Test that editing with non-existent string raises StringNotFoundError."""
        _, test_dir = git_repo

        with pytest.raises(StringNotFoundError):
            await file_write.edit_file(
                commit_message="Edit",
                file_path=str(test_dir / "code.py"),
                old_string="nonexistent_string_xyz",
                new_string="replacement",
            )

    @pytest.mark.asyncio
    async def test_edit_file_identical_strings(self, file_write, git_repo):
        """Test that editing with identical strings raises ValueError."""
        _, test_dir = git_repo

        with pytest.raises(ValueError, match="identical"):
            await file_write.edit_file(
                commit_message="Edit",
                file_path=str(test_dir / "code.py"),
                old_string="world",
                new_string="world",
            )

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self, file_write, git_repo):
        """Test that editing a non-existent file fails.

        Note: Non-existent files are not tracked by git, so FileNotTrackedError
        is raised before the FileNotFoundError check.
        """
        _, test_dir = git_repo

        with pytest.raises(FileNotTrackedError):
            await file_write.edit_file(
                commit_message="Edit",
                file_path=str(test_dir / "nonexistent.txt"),
                old_string="foo",
                new_string="bar",
            )

    @pytest.mark.asyncio
    async def test_edit_untracked_file_fails(self, git_repo):
        """Test that editing an untracked file raises FileNotTrackedError."""
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[AccessRule(AccessType.WRITE, "**")], default_access=AccessType.WRITE)
        file_write = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

        # Create an untracked file
        (test_dir / "untracked.txt").write_text("some content\n")

        with pytest.raises(FileNotTrackedError):
            await file_write.edit_file(
                commit_message="Edit untracked",
                file_path=str(test_dir / "untracked.txt"),
                old_string="some",
                new_string="other",
            )

    @pytest.mark.asyncio
    async def test_edit_directory_fails(self, file_write, git_repo):
        """Test that editing a directory fails.

        Note: Directories are not tracked by git, so FileNotTrackedError is raised
        before the IsADirectoryError check.
        """
        _, test_dir = git_repo

        with pytest.raises(FileNotTrackedError):
            await file_write.edit_file(
                commit_message="Edit dir",
                file_path=str(test_dir / "src"),
                old_string="foo",
                new_string="bar",
            )


class TestFileWriteAccess:
    """Tests for FileWrite access control."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    @pytest.mark.asyncio
    async def test_access_write_allowed(self, git_repo):
        """Test that files with WRITE access can be written."""
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[AccessRule(AccessType.WRITE, "*.txt")])
        file_write = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

        result = await file_write.write_file(
            commit_message="Write txt",
            file_path=str(test_dir / "new.txt"),
            content="content\n",
        )

        assert "commit" in result.lower()
        assert (test_dir / "new.txt").exists()

    @pytest.mark.asyncio
    async def test_access_read_only_denied(self, git_repo):
        """Test that files with only READ access cannot be written."""
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[AccessRule(AccessType.READ, "**")])  # Only READ, no WRITE
        file_write = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

        with pytest.raises(AccessDeniedError):
            await file_write.write_file(
                commit_message="Write denied",
                file_path=str(test_dir / "new.txt"),
                content="content\n",
            )

    @pytest.mark.asyncio
    async def test_access_exclude_denied(self, git_repo):
        """Test that files with EXCLUDE access cannot be written."""
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[
            AccessRule(AccessType.WRITE, "**"),
            AccessRule(AccessType.EXCLUDE, "private/**"),
        ])
        file_write = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

        with pytest.raises(AccessDeniedError):
            await file_write.write_file(
                commit_message="Write to excluded",
                file_path=str(test_dir / "private" / "new.txt"),
                content="content\n",
            )

    @pytest.mark.asyncio
    async def test_access_mixed_rules(self, git_repo):
        """Test writing with mixed access rules.

        Note: Access rules are evaluated in order, last matching rule wins.
        """
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[
            AccessRule(AccessType.READ, "**"),  # READ to all (base)
            AccessRule(AccessType.WRITE, "src/**"),  # WRITE to src/ (overrides)
            AccessRule(AccessType.EXCLUDE, "private/**"),  # EXCLUDE private/
        ])
        file_write = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

        # Can write to src/ (WRITE access - last matching rule)
        result = await file_write.write_file(
            commit_message="Write to src",
            file_path=str(test_dir / "src" / "new.py"),
            content="# new file\n",
        )
        assert "commit" in result.lower()

        # Cannot write to root (only READ access)
        with pytest.raises(AccessDeniedError):
            await file_write.write_file(
                commit_message="Write to root",
                file_path=str(test_dir / "root.txt"),
                content="content\n",
            )

        # Cannot write to private/ (EXCLUDE)
        with pytest.raises(AccessDeniedError):
            await file_write.write_file(
                commit_message="Write to private",
                file_path=str(test_dir / "private" / "new.txt"),
                content="content\n",
            )

    @pytest.mark.asyncio
    async def test_edit_access_write_required(self, git_repo):
        """Test that editing requires WRITE access."""
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[AccessRule(AccessType.READ, "**")])  # Only READ
        file_write = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

        with pytest.raises(AccessDeniedError):
            await file_write.edit_file(
                commit_message="Edit",
                file_path=str(test_dir / "code.py"),
                old_string="world",
                new_string="universe",
            )

    @pytest.mark.asyncio
    async def test_edit_with_write_access(self, git_repo):
        """Test that editing works with WRITE access."""
        repo, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        workspace = GitWorkspace(sandbox=sandbox, root=str(test_dir))
        workspace.set_access(accesses=[AccessRule(AccessType.WRITE, "*.py")])
        file_write = FileWrite(
            sandbox=sandbox,
            workspace=workspace,
        )

        result = await file_write.edit_file(
            commit_message="Edit py file",
            file_path=str(test_dir / "code.py"),
            old_string="world",
            new_string="universe",
        )

        assert "commit" in result.lower()
        assert "universe" in (test_dir / "code.py").read_text()
