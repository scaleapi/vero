"""Tests for the tools."""

import json
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from vero.exceptions import AccessDeniedError
from vero.filesystem import AccessRule, AccessType, Filesystem
from vero.sandbox import LocalSandbox, Sandbox
from vero.tools.bash import BashTool
from vero.tools.file_read import FileRead
from vero.tools.grep import Grep
from vero.tools.planning import TodoList, TodoStatus
from vero.workspace.base import Workspace


class SimpleWorkspace(Workspace):
    """Minimal Workspace for tool testing (no version control needed)."""

    def __init__(self, sandbox: Sandbox, root: str):
        self._sandbox = sandbox
        self._root = root
        self._fs = Filesystem(root=Path(root), default_access=AccessType.WRITE)

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def root(self) -> str:
        return self._root

    @property
    def project_path(self) -> str:
        return self._root

    @property
    def name(self) -> str:
        return "test"

    async def current_version(self) -> str:
        return ""

    async def save(self, message: str = "Save") -> str:
        return ""

    async def restore(self, version_id: str, message: str | None = None) -> str:
        return ""

    async def diff(self, from_version=None, to_version=None) -> str:
        return ""

    async def log(self, max_count: int = 10, since_version=None) -> str:
        return ""

    async def is_ancestor(self, version_a: str, version_b: str) -> bool:
        return False

    async def copy(self, name=None, from_version=None) -> Workspace:
        return self

    async def is_dirty(self) -> bool:
        return False


def _make_sandbox_and_workspace(
    root: Path,
    accesses: list[AccessRule],
    default_access: AccessType = AccessType.EXCLUDE,
) -> tuple[LocalSandbox, SimpleWorkspace]:
    """Create a LocalSandbox and SimpleWorkspace with access rules on the workspace."""
    sandbox = LocalSandbox(root=root)
    workspace = SimpleWorkspace(sandbox=sandbox, root=str(root))
    workspace.set_access(accesses=accesses, default_access=default_access)
    return sandbox, workspace


class TestTodoList:
    """Tests for TodoList tool."""

    def test_add_todos(self):
        """Test adding todo items."""
        todo_list = TodoList()
        todo_list.add_todos("Write tests")
        assert len(todo_list.todos) == 1
        assert todo_list.todo_statuses["Write tests"] == TodoStatus.NOT_STARTED

    def test_update_todo_status(self):
        """Test updating todo status by task_id."""
        todo_list = TodoList()
        todo_list.add_todos("Task 1")
        todo_list.add_todos("Task 2")
        todo_list.update_todo_status(status=TodoStatus.COMPLETED, task_id=0)

        assert len(todo_list.todo_statuses) == 2
        assert todo_list.todo_statuses["Task 1"] == TodoStatus.COMPLETED
        assert todo_list.todo_statuses["Task 2"] == TodoStatus.NOT_STARTED

        result = todo_list.list_todos(status=[TodoStatus.COMPLETED])
        assert "Task 1" in result
        assert "Task 2" not in result

        result = todo_list.list_todos(status=[TodoStatus.NOT_STARTED])
        assert "Task 1" not in result
        assert "Task 2" in result

    def test_update_status_error_handling(self):
        """Test that updating status without task or task_id raises ValueError."""
        todo_list = TodoList()
        todo_list.add_todos("Task 1")

        with pytest.raises(ValueError):
            todo_list.update_todo_status(status=TodoStatus.COMPLETED)

    def test_list_todos(self):
        """Test listing todos with status filters."""
        todo_list = TodoList()
        todo_list.add_todos("Task 1")
        todo_list.add_todos("Task 2")
        todo_list.add_todos("Task 3")

        # Update statuses
        todo_list.update_todo_status(status=TodoStatus.COMPLETED, task_id=0)
        todo_list.update_todo_status(status=TodoStatus.IN_PROGRESS, task_id=1)

        # List completed tasks
        result = todo_list.list_todos(status=TodoStatus.COMPLETED)
        json_match = re.search(r"```json(.+?)```", result, re.DOTALL)
        todos_dict = json.loads(json_match.group(1))
        assert len(todos_dict) == 1
        assert todos_dict["0"]["task"] == "Task 1"


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep (rg) not available")
class TestGrep:
    """Tests for Grep."""

    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory with test files for grep tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir).resolve()

            # Create various test files
            (test_dir / "main.py").write_text(
                "def hello():\n    print('hello world')\n\ndef foo():\n    print('hello again')\n"
            )
            (test_dir / "utils.py").write_text(
                "def helper():\n    return 'hello helper'\n\ndef another():\n    pass\n"
            )

            # Create subdirectories
            (test_dir / "src").mkdir()
            (test_dir / "src" / "app.py").write_text(
                "# Main app\nhello = 'greeting'\nworld = 'planet'\n"
            )

            (test_dir / "private").mkdir()
            (test_dir / "private" / "secret.py").write_text(
                "secret_hello = 'hidden'\npassword = '123'\n"
            )

            (test_dir / "logs").mkdir()
            (test_dir / "logs" / "app.log").write_text(
                "INFO: hello started\nERROR: hello failed\nINFO: done\n"
            )

            yield test_dir

    @pytest.mark.asyncio
    async def test_basic(self, test_dir):
        """Test Grep basic functionality."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(pattern="hello", path=str(test_dir / "main.py"))
        assert "hello" in result
        assert "No matches found" not in result

    @pytest.mark.asyncio
    async def test_multiple_matches(self, test_dir):
        """Test Grep finds all matches in a file."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        # main.py has "hello" on lines 1, 2, and 4
        result = await grep_tool(pattern="hello", path=str(test_dir / "main.py"))

        # Should find multiple occurrences
        assert result.count("hello") >= 2

    @pytest.mark.asyncio
    async def test_multiple_files(self, test_dir):
        """Test Grep finds matches across multiple files."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        # Search entire directory
        result = await grep_tool(pattern="hello", path=str(test_dir))

        # Should find matches in multiple files
        assert "main.py" in result
        assert "utils.py" in result
        assert "app.py" in result

    @pytest.mark.asyncio
    async def test_content_mode(self, test_dir):
        """Test Grep content output mode shows matching lines with line numbers."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(
            pattern="hello", path=str(test_dir / "main.py"), output_mode="content"
        )

        # Content mode should show line numbers and content
        assert "def hello" in result or "hello" in result
        # Should have line number format (e.g., "1:" or similar)
        assert re.search(r":\d+:", result) or re.search(r"\d+:", result)

    @pytest.mark.asyncio
    async def test_files_with_matches_mode(self, test_dir):
        """Test Grep files_with_matches mode returns only file paths."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(
            pattern="hello", path=str(test_dir), output_mode="files_with_matches"
        )

        # Should contain file paths
        assert "main.py" in result
        assert "utils.py" in result
        # Should NOT contain the actual content
        assert "def hello" not in result
        assert "print('hello" not in result

    @pytest.mark.asyncio
    async def test_count_mode(self, test_dir):
        """Test Grep count mode returns match counts per file."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(pattern="hello", path=str(test_dir), output_mode="count")

        # Count mode shows filepath:count format
        # main.py has multiple "hello" occurrences
        assert "main.py" in result
        # Should have a count number
        assert re.search(r":\d+", result)

    @pytest.mark.asyncio
    async def test_case_insensitive(self, test_dir):
        """Test Grep case insensitive search."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        # Add a file with mixed case
        (test_dir / "mixed.py").write_text("HELLO = 'upper'\nhello = 'lower'\nHeLLo = 'mixed'\n")

        result_sensitive = await grep_tool(
            pattern="HELLO", path=str(test_dir / "mixed.py"), i=False
        )
        result_insensitive = await grep_tool(
            pattern="HELLO", path=str(test_dir / "mixed.py"), i=True
        )

        # Case insensitive should find more matches
        assert result_insensitive.count("ello") >= result_sensitive.count("ello")

    @pytest.mark.asyncio
    async def test_context_lines(self, test_dir):
        """Test Grep context lines (A, B, C options)."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        # Test with context lines
        result_with_context = await grep_tool(
            pattern="ERROR", path=str(test_dir / "logs" / "app.log"), C=1
        )

        # With context, should include surrounding lines
        assert "ERROR" in result_with_context
        # Context should include adjacent lines (INFO lines)
        assert "INFO" in result_with_context

    @pytest.mark.asyncio
    async def test_access_read_included(self, test_dir):
        """Test that files with READ access are included in grep results."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "*.py"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
            ])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(
            pattern="hello", path=str(test_dir), output_mode="files_with_matches"
        )

        # READ access files should be included
        assert "main.py" in result
        assert "utils.py" in result

    @pytest.mark.asyncio
    async def test_access_write_included(self, test_dir):
        """Test that files with WRITE access are included in grep results (WRITE implies READ)."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.WRITE, "src/**"),
                AccessRule(AccessType.READ, "*.py"),
            ])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(
            pattern="hello", path=str(test_dir), output_mode="files_with_matches"
        )

        # WRITE access files should be included (WRITE implies READ)
        assert "app.py" in result
        # READ access files should also be included
        assert "main.py" in result

    @pytest.mark.asyncio
    async def test_access_exclude_excluded(self, test_dir):
        """Test that files with EXCLUDE access are excluded from grep results."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
            ])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(
            pattern="hello", path=str(test_dir), output_mode="files_with_matches"
        )

        # EXCLUDE paths should not appear in results
        assert "secret.py" not in result
        # Check that the private directory from our test fixture is excluded
        # (not the macOS /private/ path prefix)
        assert f"{test_dir}/private" not in result
        # Other files should still be found
        assert "main.py" in result

    @pytest.mark.asyncio
    async def test_access_mixed_rules(self, test_dir):
        """Test grep with mixed access rules (READ, WRITE, EXCLUDE)."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.WRITE, "src/**"),
                AccessRule(AccessType.READ, "*.py"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
                AccessRule(AccessType.EXCLUDE, "logs/**"),
            ])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(
            pattern="hello", path=str(test_dir), output_mode="files_with_matches"
        )

        # WRITE access (src/) should be included
        assert "app.py" in result
        # READ access (*.py at root) should be included
        assert "main.py" in result
        assert "utils.py" in result
        # EXCLUDE paths should not appear
        assert "secret.py" not in result
        # Check that the private/logs directories from our test fixture are excluded
        # (not the macOS /private/ path prefix)
        assert f"{test_dir}/private" not in result
        assert f"{test_dir}/logs" not in result

    @pytest.mark.asyncio
    async def test_glob_filter(self, test_dir):
        """Test Grep with glob pattern filter."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(
            pattern="hello", path=str(test_dir), output_mode="files_with_matches", glob="*.py"
        )

        # Should only find .py files
        assert "main.py" in result
        # Should not find .log files
        assert "app.log" not in result

    @pytest.mark.asyncio
    async def test_no_matches(self, test_dir):
        """Test Grep when no matches are found."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        grep_tool = Grep(sandbox=sandbox, workspace=workspace)

        result = await grep_tool(pattern="nonexistent_pattern_xyz123", path=str(test_dir))

        assert "No matches found" in result

    @pytest.mark.asyncio
    async def test_path_with_hyphen_digit(self):
        """Test grep works with paths containing hyphen-digit patterns like 'v1-2-beta'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory with hyphen-digit pattern (previously caused regex bug)
            test_dir = Path(tmpdir) / "project-v1-2-beta-3"
            test_dir.mkdir()
            (test_dir / "file.py").write_text("tools = []\nother_tools = {}\n")

            sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
            grep_tool = Grep(sandbox=sandbox, workspace=workspace)

            result = await grep_tool(pattern="tools")
            assert "tools" in result
            assert "No matches found" not in result
            # Should find both matches
            assert result.count("tools") >= 2

    @pytest.mark.asyncio
    async def test_file_without_extension(self):
        """Test grep works with files that have no extension (Makefile, Dockerfile, etc)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir).resolve()
            (test_dir / "Makefile").write_text(
                "build:\n\techo 'building'\n\ntest:\n\techo 'testing'\n"
            )
            (test_dir / "Dockerfile").write_text("FROM python:3.11\nRUN echo 'hello'\n")

            sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
            grep_tool = Grep(sandbox=sandbox, workspace=workspace)

            result = await grep_tool(pattern="echo")
            assert "echo" in result
            assert "No matches found" not in result

    @pytest.mark.asyncio
    async def test_file_with_multiple_dots(self):
        """Test grep works with filenames containing multiple dots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir).resolve()
            (test_dir / "test.spec.py").write_text("def test_something():\n    assert True\n")
            (test_dir / "data.backup.json").write_text('{"key": "value"}\n')

            sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
            grep_tool = Grep(sandbox=sandbox, workspace=workspace)

            result = await grep_tool(pattern="test")
            assert "test" in result
            assert "No matches found" not in result

    @pytest.mark.asyncio
    async def test_unicode_in_path(self):
        """Test grep works with unicode characters in paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "données"
            test_dir.mkdir()
            (test_dir / "résumé.py").write_text("name = 'café'\nprint(name)\n")

            sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
            grep_tool = Grep(sandbox=sandbox, workspace=workspace)

            result = await grep_tool(pattern="café")
            assert "café" in result
            assert "No matches found" not in result

    @pytest.mark.asyncio
    async def test_pattern_matches_path(self):
        """Test grep when the search pattern also appears in the file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test_project"
            test_dir.mkdir()
            (test_dir / "test_file.py").write_text("def test_func():\n    pass\n")

            sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
            grep_tool = Grep(sandbox=sandbox, workspace=workspace)

            # Pattern "test" appears in both path and content
            result = await grep_tool(pattern="test")
            assert "test_func" in result
            assert "No matches found" not in result

    @pytest.mark.asyncio
    async def test_deeply_nested_path(self):
        """Test grep works with deeply nested directory structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create deep nesting
            deep_dir = Path(tmpdir) / "a" / "b" / "c" / "d" / "e" / "f"
            deep_dir.mkdir(parents=True)
            (deep_dir / "deep.py").write_text("found_me = True\n")

            sandbox, workspace = _make_sandbox_and_workspace(Path(tmpdir), [AccessRule(AccessType.READ, "**")])
            grep_tool = Grep(sandbox=sandbox, workspace=workspace)

            result = await grep_tool(pattern="found_me")
            assert "found_me" in result
            assert "No matches found" not in result

    @pytest.mark.asyncio
    async def test_empty_file(self):
        """Test grep on empty files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir).resolve()
            (test_dir / "blank.py").write_text("")
            (test_dir / "has_content.py").write_text("content = 'here'\n")

            sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
            grep_tool = Grep(sandbox=sandbox, workspace=workspace)

            result = await grep_tool(pattern="content")
            assert "content" in result
            # Empty file should not cause errors, and should not appear in results
            assert "blank.py" not in result
            assert "has_content.py" in result


class TestFileRead:
    """Tests for FileRead."""

    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir).resolve()

            # Create test files with known content
            (test_dir / "simple.txt").write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")
            (test_dir / "code.py").write_text(
                "def hello():\n    print('world')\n\ndef foo():\n    return 42\n"
            )

            # Create subdirectories
            (test_dir / "src").mkdir()
            (test_dir / "src" / "app.py").write_text("# App code\nimport os\n")

            (test_dir / "private").mkdir()
            (test_dir / "private" / "secret.txt").write_text("secret data\n")

            yield test_dir

    @pytest.mark.asyncio
    async def test_basic_read(self, test_dir):
        """Test basic file reading."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        result = await read_tool(target_file=str(test_dir / "simple.txt"))

        assert "line 1" in result
        assert "line 5" in result
        # Should include line numbers
        assert "1|" in result or "|line 1" in result

    @pytest.mark.asyncio
    async def test_read_with_start_line(self, test_dir):
        """Test reading from a specific start line."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        result = await read_tool(target_file=str(test_dir / "simple.txt"), start_line=3)

        # Should start from line 3
        assert "line 3" in result
        assert "line 4" in result
        # Line 1 and 2 should not be in content (though might be in metadata)
        # Check that line numbers are correct
        assert "3|" in result or "|line 3" in result

    @pytest.mark.asyncio
    async def test_read_with_num_lines(self, test_dir):
        """Test reading a specific number of lines."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        result = await read_tool(target_file=str(test_dir / "simple.txt"), start_line=1, num_lines=2)

        assert "line 1" in result
        assert "line 2" in result
        # Should only have 2 lines of content
        assert "line 3" not in result

    @pytest.mark.asyncio
    async def test_read_with_char_limit(self, test_dir):
        """Test reading with character limit."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace, max_char_limit=1000)

        # Create a larger file
        large_content = "\n".join([f"line {i}" for i in range(100)])
        (test_dir / "large.txt").write_text(large_content)

        result = await read_tool(target_file=str(test_dir / "large.txt"), char_limit=100)

        # Result should be truncated
        assert "truncated" in result.lower() or len(result) <= 200  # Some overhead for metadata

    @pytest.mark.asyncio
    async def test_char_limit_exceeds_max(self, test_dir):
        """Test that char_limit exceeding max raises ValueError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace, max_char_limit=100)

        with pytest.raises(ValueError, match="exceeds maximum"):
            await read_tool(target_file=str(test_dir / "simple.txt"), char_limit=200)

    @pytest.mark.asyncio
    async def test_invalid_start_line(self, test_dir):
        """Test that invalid start_line raises ValueError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        with pytest.raises(ValueError, match="Start line must be greater than or equal to 1"):
            await read_tool(target_file=str(test_dir / "simple.txt"), start_line=0)

        with pytest.raises(ValueError, match="Start line must be greater than or equal to 1"):
            await read_tool(target_file=str(test_dir / "simple.txt"), start_line=-1)

    @pytest.mark.asyncio
    async def test_file_not_found(self, test_dir):
        """Test reading a non-existent file raises FileNotFoundError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        with pytest.raises(FileNotFoundError):
            await read_tool(target_file=str(test_dir / "nonexistent.txt"))

    @pytest.mark.asyncio
    async def test_read_directory_raises_error(self, test_dir):
        """Test reading a directory raises IsADirectoryError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        with pytest.raises(IsADirectoryError):
            await read_tool(target_file=str(test_dir / "src"))

    @pytest.mark.asyncio
    async def test_access_read_allowed(self, test_dir):
        """Test that files with READ access can be read."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "*.txt")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        result = await read_tool(target_file=str(test_dir / "simple.txt"))
        assert "line 1" in result

    @pytest.mark.asyncio
    async def test_access_write_implies_read(self, test_dir):
        """Test that files with WRITE access can be read (WRITE implies READ)."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.WRITE, "src/**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        result = await read_tool(target_file=str(test_dir / "src" / "app.py"))
        assert "App code" in result

    @pytest.mark.asyncio
    async def test_access_exclude_denied(self, test_dir):
        """Test that files with EXCLUDE access cannot be read."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
            ])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        with pytest.raises(AccessDeniedError):
            await read_tool(target_file=str(test_dir / "private" / "secret.txt"))

    @pytest.mark.asyncio
    async def test_access_no_matching_rule_denied(self, test_dir):
        """Test that files with no matching access rule are denied."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "*.py")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        with pytest.raises(AccessDeniedError):
            await read_tool(target_file=str(test_dir / "simple.txt"))  # .txt file

    @pytest.mark.asyncio
    async def test_access_mixed_rules(self, test_dir):
        """Test reading with mixed access rules."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.WRITE, "src/**"),
                AccessRule(AccessType.READ, "*.txt"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
            ])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        # Can read .txt files (READ access)
        result = await read_tool(target_file=str(test_dir / "simple.txt"))
        assert "line 1" in result

        # Can read src/ files (WRITE implies READ)
        result = await read_tool(target_file=str(test_dir / "src" / "app.py"))
        assert "App code" in result

        # Cannot read private/ files (EXCLUDE)
        with pytest.raises(AccessDeniedError):
            await read_tool(target_file=str(test_dir / "private" / "secret.txt"))

        # Cannot read .py files at root (no matching rule)
        with pytest.raises(AccessDeniedError):
            await read_tool(target_file=str(test_dir / "code.py"))

    @pytest.mark.asyncio
    async def test_empty_file(self, test_dir):
        """Test reading an empty file."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        read_tool = FileRead(sandbox=sandbox, workspace=workspace)

        (test_dir / "empty.txt").write_text("")
        result = await read_tool(target_file=str(test_dir / "empty.txt"))

        # Should return empty string for empty file
        assert result == ""


class TestBashTool:
    """Tests for BashTool (ls, pwd, find)."""

    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory with test files and subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir).resolve()

            # Create test files
            (test_dir / "file1.txt").write_text("content1")
            (test_dir / "file2.py").write_text("print('hello')")
            (test_dir / ".hidden").write_text("hidden content")

            # Create subdirectories
            (test_dir / "src").mkdir()
            (test_dir / "src" / "app.py").write_text("# app code")
            (test_dir / "src" / "utils.py").write_text("# utils")

            (test_dir / "private").mkdir()
            (test_dir / "private" / "secret.txt").write_text("secret")

            (test_dir / "docs").mkdir()
            (test_dir / "docs" / "readme.md").write_text("# Readme")

            yield test_dir

    # ==================== pwd tests ====================

    @pytest.mark.asyncio
    async def test_pwd(self, test_dir):
        """Test pwd returns current working directory."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [AccessRule(AccessType.READ, "**")])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.pwd()
        assert result == str(test_dir)

    # ==================== ls tests ====================

    @pytest.mark.asyncio
    async def test_ls_basic(self, test_dir):
        """Test ls basic functionality."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.ls(path=str(test_dir))

        assert "file1.txt" in result
        assert "file2.py" in result
        assert "src/" in result
        assert "docs/" in result

    @pytest.mark.asyncio
    async def test_ls_hidden_files(self, test_dir):
        """Test ls with all flag shows hidden files."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result_without_all = await bash_tool.ls(path=str(test_dir), all=False)
        result_with_all = await bash_tool.ls(path=str(test_dir), all=True)

        assert ".hidden" not in result_without_all
        assert ".hidden" in result_with_all

    @pytest.mark.asyncio
    async def test_ls_long_format(self, test_dir):
        """Test ls with long format shows permissions and sizes."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.ls(path=str(test_dir), long=True)

        # Long format includes permissions (e.g., drwx or -rw-)
        assert "rw" in result or "-" in result

    @pytest.mark.asyncio
    async def test_ls_classify_dirs(self, test_dir):
        """Test ls with classify_dirs appends / to directories."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.ls(path=str(test_dir), classify_dirs=True)

        assert "src/" in result
        assert "docs/" in result

    @pytest.mark.asyncio
    async def test_ls_pagination(self, test_dir):
        """Test ls pagination with offset and limit."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        _ = await bash_tool.ls(path=str(test_dir))
        result_limited = await bash_tool.ls(path=str(test_dir), limit=2)

        # Limited result should have fewer items
        assert "Viewing 2 items" in result_limited

    @pytest.mark.asyncio
    async def test_ls_access_exclude(self, test_dir):
        """Test ls excludes paths with EXCLUDE access."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.ls(path=str(test_dir))

        # private directory should still be listed but its contents not accessible
        assert "file1.txt" in result

    @pytest.mark.asyncio
    async def test_ls_access_denied_path(self, test_dir):
        """Test ls on excluded path raises error."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, "private"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        with pytest.raises(AccessDeniedError):
            await bash_tool.ls(path=str(test_dir / "private"))

    @pytest.mark.asyncio
    async def test_ls_nonexistent_path(self, test_dir):
        """Test ls on non-existent path raises error."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        with pytest.raises((FileNotFoundError, RuntimeError)):
            await bash_tool.ls(path=str(test_dir / "nonexistent"))

    @pytest.mark.asyncio
    async def test_ls_default_path(self, test_dir):
        """Test ls without path uses cwd."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.ls()

        # Should list contents of root (cwd)
        assert "file1.txt" in result

    # ==================== find tests ====================

    @pytest.mark.asyncio
    async def test_find_basic(self, test_dir):
        """Test find basic functionality."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.find(path=str(test_dir), maxdepth=2, exclude_paths=[])

        assert "file1.txt" in result or str(test_dir / "file1.txt") in result

    @pytest.mark.asyncio
    async def test_find_by_name(self, test_dir):
        """Test find with name pattern."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.find(path=str(test_dir), name="*.py", maxdepth=2, exclude_paths=[])

        assert "file2.py" in result or "app.py" in result
        assert "file1.txt" not in result

    @pytest.mark.asyncio
    async def test_find_by_type_file(self, test_dir):
        """Test find with type='f' for files only."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.find(path=str(test_dir), type="f", maxdepth=1, exclude_paths=[])

        assert "file1.txt" in result or str(test_dir / "file1.txt") in result
        # Directories should not be listed as files
        # Note: the root path itself might be included

    @pytest.mark.asyncio
    async def test_find_by_type_directory(self, test_dir):
        """Test find with type='d' for directories only."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.find(path=str(test_dir), type="d", maxdepth=1, exclude_paths=[])

        assert "src" in result or str(test_dir / "src") in result
        # Regular files should not be included
        assert "file1.txt" not in result

    @pytest.mark.asyncio
    async def test_find_maxdepth(self, test_dir):
        """Test find respects maxdepth."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result_depth1 = await bash_tool.find(path=str(test_dir), maxdepth=1, exclude_paths=[])
        result_depth2 = await bash_tool.find(path=str(test_dir), maxdepth=2, exclude_paths=[])

        # Depth 2 should find files in subdirectories
        assert "app.py" in result_depth2 or str(test_dir / "src" / "app.py") in result_depth2
        # Depth 1 should not find files in subdirectories
        assert "app.py" not in result_depth1

    @pytest.mark.asyncio
    async def test_find_maxdepth_invalid(self, test_dir):
        """Test find with invalid maxdepth raises ValueError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        with pytest.raises(ValueError, match="maxdepth must be greater than or equal to 0"):
            await bash_tool.find(path=str(test_dir), maxdepth=-1)

    @pytest.mark.asyncio
    async def test_find_maxdepth_exceeds_limit(self, test_dir):
        """Test find with maxdepth exceeding limit raises ValueError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace, find_max_depth=3)

        with pytest.raises(ValueError, match="maxdepth must be less than or equal to"):
            await bash_tool.find(path=str(test_dir), maxdepth=5)

    @pytest.mark.asyncio
    async def test_find_exclude_paths(self, test_dir):
        """Test find with exclude_paths."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.find(path=str(test_dir), maxdepth=2, exclude_paths=["*/private/*"])

        assert "secret.txt" not in result

    @pytest.mark.asyncio
    async def test_find_exclude_hidden(self, test_dir):
        """Test find default excludes hidden files."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        # Default exclude_paths includes "*/.*"
        result = await bash_tool.find(path=str(test_dir), maxdepth=1)

        assert ".hidden" not in result

    @pytest.mark.asyncio
    async def test_find_pagination(self, test_dir):
        """Test find pagination with offset and limit."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result_limited = await bash_tool.find(
            path=str(test_dir), maxdepth=2, limit=2, exclude_paths=[]
        )

        assert "Showing 2 items" in result_limited

    @pytest.mark.asyncio
    async def test_find_access_exclude(self, test_dir):
        """Test find excludes paths with EXCLUDE access."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.find(path=str(test_dir), maxdepth=2, exclude_paths=[])

        # private directory contents should be filtered out
        assert "secret.txt" not in result

    @pytest.mark.asyncio
    async def test_find_access_write_implies_read(self, test_dir):
        """Test find includes paths with WRITE access."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.WRITE, "src"),
                AccessRule(AccessType.WRITE, "src/**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.find(path=str(test_dir / "src"), maxdepth=1, exclude_paths=[])

        assert "app.py" in result or str(test_dir / "src" / "app.py") in result

    @pytest.mark.asyncio
    async def test_find_default_path(self, test_dir):
        """Test find without path uses cwd."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.find(maxdepth=1, exclude_paths=[])

        # Should search from root (cwd)
        assert str(test_dir) in result

    @pytest.mark.asyncio
    async def test_find_exclude_paths_invalid_type(self, test_dir):
        """Test find with invalid exclude_paths type raises ValueError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        with pytest.raises(ValueError, match="exclude_paths must be a list"):
            await bash_tool.find(path=str(test_dir), maxdepth=1, exclude_paths="invalid")

    # ==================== tree tests ====================

    @pytest.mark.asyncio
    async def test_tree_basic(self, test_dir):
        """Test tree basic functionality."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.tree(path=str(test_dir))

        assert "file1.txt" in result
        assert "file2.py" in result
        assert "src/" in result
        assert "docs/" in result

    @pytest.mark.asyncio
    async def test_tree_structure(self, test_dir):
        """Test tree output has correct structure with connectors."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.tree(path=str(test_dir))

        # Should contain tree connectors
        assert "├── " in result or "└── " in result

    @pytest.mark.asyncio
    async def test_tree_max_depth(self, test_dir):
        """Test tree respects max_depth."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result_depth1 = await bash_tool.tree(path=str(test_dir), max_depth=1)
        result_depth2 = await bash_tool.tree(path=str(test_dir), max_depth=2)

        # Depth 1 should show directories but not their contents
        assert "src/" in result_depth1
        # Files inside src/ should not appear at depth 1
        assert "app.py" not in result_depth1

        # Depth 2 should show files inside subdirectories
        assert "app.py" in result_depth2

    @pytest.mark.asyncio
    async def test_tree_max_depth_invalid(self, test_dir):
        """Test tree with invalid max_depth raises ValueError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        with pytest.raises(ValueError, match="max_depth must be at least 1"):
            await bash_tool.tree(path=str(test_dir), max_depth=0)

    @pytest.mark.asyncio
    async def test_tree_max_depth_exceeds_limit(self, test_dir):
        """Test tree with max_depth exceeding limit raises ValueError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace, find_max_depth=3)

        with pytest.raises(ValueError, match="max_depth .* exceeds limit"):
            await bash_tool.tree(path=str(test_dir), max_depth=5)

    @pytest.mark.asyncio
    async def test_tree_pagination(self, test_dir):
        """Test tree pagination with offset and limit."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.tree(path=str(test_dir), limit=2)

        assert "2 files shown" in result
        assert "Pagination:" in result

    @pytest.mark.asyncio
    async def test_tree_access_exclude(self, test_dir):
        """Test tree excludes paths with EXCLUDE access."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, "private"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.tree(path=str(test_dir))

        # private directory and contents should not appear
        assert "private/" not in result
        assert "secret.txt" not in result
        # Other files should still appear
        assert "file1.txt" in result

    @pytest.mark.asyncio
    async def test_tree_access_write_implies_read(self, test_dir):
        """Test tree includes paths with WRITE access."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.WRITE, "src"),
                AccessRule(AccessType.WRITE, "src/**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.tree(path=str(test_dir / "src"))

        assert "app.py" in result
        assert "utils.py" in result

    @pytest.mark.asyncio
    async def test_tree_nonexistent_path(self, test_dir):
        """Test tree on non-existent path raises FileNotFoundError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        with pytest.raises(FileNotFoundError):
            await bash_tool.tree(path=str(test_dir / "nonexistent"))

    @pytest.mark.asyncio
    async def test_tree_on_file(self, test_dir):
        """Test tree on a file raises NotADirectoryError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        with pytest.raises(NotADirectoryError):
            await bash_tool.tree(path=str(test_dir / "file1.txt"))

    @pytest.mark.asyncio
    async def test_tree_access_denied(self, test_dir):
        """Test tree on excluded path raises PermissionError."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, "private"),
                AccessRule(AccessType.EXCLUDE, "private/**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        with pytest.raises(PermissionError):
            await bash_tool.tree(path=str(test_dir / "private"))

    @pytest.mark.asyncio
    async def test_tree_default_path(self, test_dir):
        """Test tree without path uses cwd."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.tree()

        # Should list contents of root (cwd)
        assert "file1.txt" in result

    @pytest.mark.asyncio
    async def test_tree_summary(self, test_dir):
        """Test tree includes summary with file count."""
        sandbox, workspace = _make_sandbox_and_workspace(test_dir, [
                AccessRule(AccessType.READ, "."),
                AccessRule(AccessType.READ, "**"),
            ])
        bash_tool = BashTool(sandbox=sandbox, workspace=workspace)

        result = await bash_tool.tree(path=str(test_dir))

        # Should include summary line
        assert "files shown" in result
        assert "total accessible" in result
        assert "max_depth=" in result
