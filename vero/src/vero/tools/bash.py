from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import Literal, NamedTuple

from vero.filesystem import AccessType
from vero.sandbox import Sandbox
from vero.tools.utils import is_tool
from vero.workspace import Workspace
from vero.utils import paginate, strip_ansi


class BashCommandResult(NamedTuple):
    """Result from a bash command execution."""

    message: str
    command: str
    return_code: int


def format_called_process_error(e: CalledProcessError) -> str:
    """Format a CalledProcessError into a string."""
    error_str = f"Command '{e.cmd}' failed with exit code {e.returncode}."
    if e.output:
        error_str += f" Output: {e.output}"
    if e.stderr:
        error_str += f" Stderr: {e.stderr}"
    return error_str


async def run_bash_command(cmd: list[str], timeout: int | None = None) -> str:
    """Run a Bash command and return its output.

    Deprecated: tools should use sandbox.run() instead.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def terminate_process():
        """Terminate process, shielded from cancellation."""
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()
        if proc.returncode == 0:
            return stdout_str
        else:
            raise CalledProcessError(
                returncode=proc.returncode,
                cmd=cmd,
                output=stdout_str,
                stderr=stderr_str,
            )
    except asyncio.TimeoutError:
        await asyncio.shield(terminate_process())
        cmd_str = " ".join(cmd)
        raise TimeoutError(f"Command '{cmd_str}' timed out after {timeout} seconds")
    except (KeyboardInterrupt, asyncio.CancelledError):
        await asyncio.shield(terminate_process())
        raise


@dataclass
class BashTool:
    """Restricted shell commands for filesystem exploration."""

    exclude_tools: list[str] = field(default_factory=list)
    timeout: int = 30
    max_chars: int = 10_000
    find_max_depth: int = 3

    # Runtime fields — set during bind()
    sandbox: Sandbox | None = None
    workspace: Workspace | None = None

    def bind(self, session) -> None:
        if session.workspace:
            self.sandbox = session.workspace.sandbox
            self.workspace = session.workspace

    @is_tool
    async def ls(
        self,
        path: str | None = None,
        all: bool = False,
        long: bool = False,
        human_readable: bool = False,
        classify_dirs: bool = True,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """List directory contents with pagination.

        Args:
            path: The directory path to list (defaults to workdir)
            all: Include hidden files (starting with .)
            long: Use long listing format
            human_readable: Print human-readable sizes (with -l)
            classify_dirs: Append / to directories
            offset: Offset to start from when paginating results
            limit: Maximum number of items in a single paginated response
        Returns:
            String with the directory listing
        """
        absolute_path = await self.workspace.validate_read_path(path or ".")
        readable_entries: list[tuple[str, str]] = []
        for name in await self.sandbox.list_dir(absolute_path):
            if not all and name.startswith("."):
                continue
            absolute_subpath = f"{absolute_path.rstrip('/')}/{name}"
            try:
                canonical = await self.sandbox.canonicalize(absolute_subpath)
            except FileNotFoundError:
                continue
            if self.workspace.can_read(canonical):
                readable_entries.append((name, absolute_subpath))

        if long and readable_entries:
            cmd = ["ls", "-ld"]
            if human_readable:
                cmd.append("-h")
            if classify_dirs:
                cmd.append("-F")
            cmd.extend(entry_path for _, entry_path in readable_entries)
            result = await self.sandbox.run(cmd, timeout=self.timeout)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ls failed (exit {result.returncode}): {result.stderr}"
                )
            readable_subpaths = strip_ansi(result.stdout).splitlines()
        else:
            readable_subpaths = []
            for name, entry_path in readable_entries:
                suffix = (
                    "/"
                    if classify_dirs and await self.sandbox.is_dir(entry_path)
                    else ""
                )
                readable_subpaths.append(f"{name}{suffix}")

        paginated_output = paginate(
            items=readable_subpaths,
            max_chars=self.max_chars,
            offset=offset,
            limit=limit,
        )
        paginated_output_str = "\n".join(paginated_output)

        return f"Viewing {len(paginated_output)} items from {offset + 1} to {offset + len(paginated_output)} with paths relative to {absolute_path}:\n{paginated_output_str}"

    @is_tool
    async def pwd(self) -> str:
        """Show the current working directory.

        Returns:
            String with the current working directory path
        """
        return self.workspace.project_path

    @is_tool
    async def find(
        self,
        path: str | None = None,
        name: str | None = None,
        exclude_paths: list[str] = ["*/.*", "*/_*"],
        type: Literal["f", "d"] | None = None,
        maxdepth: int = 1,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """Locate files and directories with pagination. By default, the tool
        will exclude hidden/private paths, i.e. those starting with . or _.

        Args:
            path: The directory path to search in (defaults to workdir)
            name: Pattern to match file/directory names (supports wildcards like *.py)
            exclude_paths: Path patterns to exclude from the search results (supports wildcards like *.py)
            type: Type of items to find - 'f' for files, 'd' for directories
            maxdepth: Maximum depth to descend into directories (defaults to 1, maximum is 2)
            offset: Offset to start from when paginating results
            limit: Maximum number of items in a single paginated response
        Returns:
            String with the list of found files/directories
        """

        assert maxdepth is not None, "maxdepth must be provided."

        if maxdepth < 0:
            raise ValueError("maxdepth must be greater than or equal to 0.")

        if maxdepth > self.find_max_depth:
            raise ValueError(
                f"maxdepth must be less than or equal to {self.find_max_depth}. Change the path to search in to a subdirectory to search deeper."
            )

        if not isinstance(exclude_paths, list):
            raise ValueError("exclude_paths must be a list of strings.")

        absolute_path = await self.workspace.validate_read_path(path or ".")

        cmd = ["find", absolute_path]

        if maxdepth is not None:
            cmd.extend(["-maxdepth", str(maxdepth)])
        if type:
            cmd.extend(["-type", type])
        if name:
            cmd.extend(["-name", name])

        for exclude_path in exclude_paths:
            cmd.extend(["!", "-path", exclude_path])

        result = await self.sandbox.run(cmd, timeout=self.timeout)
        if result.returncode != 0:
            raise RuntimeError(f"find failed (exit {result.returncode}): {result.stderr}")

        stdout = result.stdout
        if not stdout or stdout.startswith("Error:"):
            return stdout

        # Filter results to only include paths within allowed directories
        filtered_output: list[str] = []
        for line in stdout.split("\n"):
            line = line.strip()
            if not line:
                continue

            current_path = self.workspace.resolve_path(line)
            try:
                canonical = await self.sandbox.canonicalize(current_path)
            except FileNotFoundError:
                continue
            if self.workspace.can_read(canonical):
                filtered_output.append(str(current_path))

        paginated_output = paginate(
            items=filtered_output, max_chars=self.max_chars, offset=offset, limit=limit
        )
        paginated_output_str = "\n".join(paginated_output)
        return f"Showing {len(paginated_output)} items from {offset + 1} to {offset + len(paginated_output)} with paths relative to {absolute_path}:\n{paginated_output_str}"

    @is_tool
    async def tree(
        self,
        path: str | None = None,
        max_depth: int = 3,
        offset: int = 0,
        limit: int | None = 100,
    ) -> str:
        """Display a directory tree structure showing accessible files and directories.

        Only files with read or write access are shown. Excluded files and directories
        are hidden from the output.

        Args:
            path: Path to the directory to display (defaults to workdir)
            max_depth: Maximum depth to traverse. Defaults to 3.
            offset: Number of files to skip from the beginning. Defaults to 0.
            limit: Maximum number of files to display. Defaults to 100.

        Returns:
            A string representation of the directory tree.
        """
        if max_depth > self.find_max_depth:
            raise ValueError(
                f"max_depth {max_depth} exceeds limit of {self.find_max_depth}"
            )
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        resolved_path = await self.workspace.validate_read_path(path or ".")
        if not await self.sandbox.exists(resolved_path):
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not await self.sandbox.is_dir(resolved_path):
            raise NotADirectoryError(f"Path is not a directory: {path}")

        file_count = 0
        files_shown = 0
        end_idx = offset + limit if limit is not None else None
        lines: list[str] = []

        def get_display_path(p: str) -> str:
            rel = self.workspace.get_relative_path(p)
            return rel if rel is not None else p

        def join_path(parent: str, child: str) -> str:
            return f"{parent.rstrip('/')}/{child}"

        def basename(p: str) -> str:
            return p.rstrip("/").rsplit("/", 1)[-1]

        async def build_tree(dir_path: str, prefix: str, current_depth: int) -> int:
            nonlocal file_count, files_shown

            if current_depth > max_depth:
                return 0

            subtree_files = 0
            try:
                entry_names = await self.sandbox.list_dir(dir_path)
                entries = [join_path(dir_path, name) for name in entry_names]
                # Sort: directories first, then files, alphabetically
                dirs = []
                files = []
                for e in entries:
                    if await self.sandbox.is_dir(e):
                        dirs.append(e)
                    else:
                        files.append(e)
                accessible = []
                for entry in dirs + files:
                    try:
                        canonical = await self.sandbox.canonicalize(entry)
                    except FileNotFoundError:
                        continue
                    if self.workspace.get_access(canonical) in (
                        AccessType.READ,
                        AccessType.WRITE,
                    ):
                        accessible.append(entry)

                for i, entry in enumerate(accessible):
                    is_last = i == len(accessible) - 1
                    connector = "└── " if is_last else "├── "
                    child_prefix = prefix + ("    " if is_last else "│   ")

                    if await self.sandbox.is_file(entry):
                        subtree_files += 1
                        file_count += 1
                        if file_count <= offset:
                            continue
                        if end_idx is not None and file_count > end_idx:
                            continue
                        lines.append(f"{prefix}{connector}{basename(entry)}")
                        files_shown += 1
                    elif await self.sandbox.is_dir(entry):
                        lines.append(f"{prefix}{connector}{basename(entry)}/")
                        child_files = await build_tree(entry, child_prefix, current_depth + 1)
                        subtree_files += child_files
            except PermissionError:
                pass

            return subtree_files

        display_path = get_display_path(resolved_path)
        lines.append(f"{display_path}/")
        total_files = await build_tree(resolved_path, "", 1)

        result_lines = lines.copy()
        result_lines.append("")
        result_lines.append(
            f"{files_shown} files shown ({total_files} total accessible, max_depth={max_depth})"
        )
        if offset > 0 or (limit is not None and files_shown < total_files):
            result_lines.append(f"Pagination: offset={offset}, limit={limit}")

        return "\n".join(result_lines)
