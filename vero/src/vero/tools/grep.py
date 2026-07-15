from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from vero.sandbox import Sandbox
from vero.workspace import Workspace


@dataclass
class Grep:
    """Search for patterns in file contents with regex support (requires ripgrep)."""

    exclude_tools: list[str] = field(default_factory=list)

    # Runtime fields — set during bind()
    sandbox: Sandbox | None = None
    workspace: Workspace | None = None

    def bind(self, session) -> None:
        if session.workspace:
            self.sandbox = session.workspace.sandbox
            self.workspace = session.workspace

    async def __call__(
        self,
        pattern: str,
        path: str | None = None,
        output_mode: Literal["content", "files_with_matches", "count"] = "content",
        i: bool = False,
        A: int | None = None,
        B: int | None = None,
        C: int | None = None,
        multiline: bool = False,
        head_limit: int | None = None,
        glob: str | None = None,
        type: str | None = None,
    ) -> str:
        """
        Search for patterns in file contents using regex.

        Args:
            pattern: Regular expression pattern to search for
            path: File or directory to search in (defaults to current directory)
            output_mode: Output format - 'content' shows matching lines,
                        'files_with_matches' shows file paths, 'count' shows match counts
            i: Case insensitive search
            A: Number of lines to show after each match
            B: Number of lines to show before each match
            C: Number of lines to show before and after each match
            multiline: Enable multiline mode where patterns can span lines
            head_limit: Limit output to first N entries
            glob: Glob pattern to filter files (e.g., '*.py')
            type: File type filter (e.g., 'py', 'js', 'ts')

        Returns:
            String with the search results

        Raises:
            RuntimeError: If ripgrep is not installed
        """
        check = await self.sandbox.run(["which", "rg"])
        if check.returncode != 0:
            raise RuntimeError(
                "ripgrep (rg) is required but not found in the sandbox. "
                "Install it with: brew install ripgrep (macOS) or apt install ripgrep (Linux)"
            )

        return await self._search_with_rg(
            pattern, path, output_mode, i, A, B, C, multiline, head_limit, glob, type
        )

    async def _search_with_rg(
        self,
        pattern: str,
        path: str | None,
        output_mode: str,
        case_insensitive: bool,
        lines_after: int | None,
        lines_before: int | None,
        lines_context: int | None,
        multiline: bool,
        head_limit: int | None,
        file_glob: str | None,
        file_type: str | None,
    ) -> str:
        """Search using ripgrep (rg) command with JSON output for reliable parsing."""
        search_path = path or "."
        # For files, validate read access. For directories, just resolve the path
        # and let the filtering handle access control on individual results.
        resolved = self.workspace.resolve_path(search_path)
        if await self.sandbox.is_file(resolved):
            search_path = await self.workspace.validate_read_path(search_path)
        else:
            search_path = resolved

        # Build ripgrep command with JSON output
        cmd = ["rg", "--json", pattern, search_path]

        # Add flags
        if case_insensitive:
            cmd.append("-i")

        if lines_context is not None:
            cmd.extend(["-C", str(lines_context)])
        else:
            if lines_after is not None:
                cmd.extend(["-A", str(lines_after)])
            if lines_before is not None:
                cmd.extend(["-B", str(lines_before)])

        if multiline:
            cmd.append("-U")

        if file_glob:
            cmd.extend(["--glob", file_glob])

        if file_type:
            cmd.extend(["-t", file_type])

        result = await self.sandbox.run(cmd, timeout=30)
        if result.returncode == 0:
            stdout = result.stdout
        elif result.returncode == 1:
            # Exit code 1 means no matches found - this is not an error
            stdout = ""
        else:
            raise RuntimeError(f"ripgrep failed (exit {result.returncode}): {result.stderr}")

        if not stdout:
            return f"No matches found for pattern: {pattern}"

        # Parse JSON output and filter by readable paths
        return await self._process_rg_json_output(
            stdout, output_mode, head_limit, pattern
        )

    async def _process_rg_json_output(
        self,
        stdout: str,
        output_mode: str,
        head_limit: int | None,
        pattern: str,
    ) -> str:
        """Process ripgrep JSON output, filter by readable paths, and format results."""
        output_lines = []
        files_seen = set()
        match_counts: dict[str, int] = {}
        entry_count = 0

        for line in stdout.strip().split("\n"):
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")
            if msg_type not in ("match", "context"):
                continue

            # Extract path from JSON (no regex needed!)
            filepath = data.get("data", {}).get("path", {}).get("text")
            if not filepath:
                continue

            # Check if path is readable
            try:
                absolute_path = self.workspace.resolve_path(filepath)
                canonical = await self.sandbox.canonicalize(absolute_path)
                if not self.workspace.can_read(canonical):
                    continue
            except Exception:
                continue

            line_number = data["data"].get("line_number", 0)
            line_text = data["data"].get("lines", {}).get("text", "").rstrip("\n")

            if output_mode == "files_with_matches":
                if filepath not in files_seen:
                    files_seen.add(filepath)
                    entry_count += 1
                    if head_limit and entry_count > head_limit:
                        break
                    output_lines.append(filepath)

            elif output_mode == "count":
                if msg_type == "match":
                    match_counts[filepath] = match_counts.get(filepath, 0) + 1

            else:  # content mode
                if head_limit and entry_count >= head_limit:
                    break
                # Format: filepath:linenum:content for matches, filepath-linenum-content for context
                sep = ":" if msg_type == "match" else "-"
                output_lines.append(f"{filepath}{sep}{line_number}{sep}{line_text}")
                if msg_type == "match":
                    entry_count += 1

        # Handle count mode output
        if output_mode == "count":
            for filepath, count in match_counts.items():
                output_lines.append(f"{filepath}:{count}")

        result = "\n".join(output_lines)
        if (
            head_limit
            and output_mode == "files_with_matches"
            and entry_count > head_limit
        ):
            result += f"\n\n(Results limited to first {head_limit} entries)"

        return result if result else f"No matches found for pattern: {pattern}"
