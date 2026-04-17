from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field

from vero.sandbox import Sandbox
from vero.utils import paginate
from vero.workspace import Workspace


@dataclass
class FileRead:
    """Read files contents."""

    exclude_tools: list[str] = field(default_factory=list)
    max_char_limit: int = 200_000

    # Runtime fields — set during bind()
    sandbox: Sandbox | None = None
    workspace: Workspace | None = None

    def bind(self, session) -> None:
        if session.workspace:
            self.sandbox = session.workspace.sandbox
            self.workspace = session.workspace

    async def assert_not_binary(self, file_path: str) -> None:
        """Assert that a file is not a binary file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and not mime_type.startswith("text"):
            try:
                content = await self.sandbox.read_file_bytes(file_path, limit=8192)
                if b"\x00" in content:
                    raise ValueError(f"'{file_path}' appears to be a binary file.")
            except ValueError:
                raise
            except Exception:
                pass

    async def __call__(
        self,
        target_file: str,
        start_line: int = 1,
        num_lines: int | None = None,
        char_limit: int | None = None,
    ) -> str:
        """
        Reads a file from the filesystem with line numbering. Will return an error if the provided path is not a file.

        Args:
            target_file: Path to the file to read (relative or absolute)
            start_line: Line number to start reading from (1-indexed)
            num_lines: Number of lines to read. If not provided, the entire file will be read.
            char_limit: Maximum number of characters to read. If not provided, the system default limit will be used.

        Returns:
            String with the content of the file
        """

        if char_limit is not None and char_limit > self.max_char_limit:
            raise ValueError(
                f"Input character limit {char_limit} exceeds maximum {self.max_char_limit} allowed by system."
            )

        if start_line < 1:
            raise ValueError(
                f"Start line must be greater than or equal to 1. Got {start_line}."
            )

        file_path = self.workspace.validate_read(target_file)

        if not await self.sandbox.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        if not await self.sandbox.is_file(file_path):
            raise IsADirectoryError(f"'{file_path}' is not a file.")

        await self.assert_not_binary(str(file_path))

        content = await self.sandbox.read_file(file_path)
        lines = content.splitlines(keepends=True)

        if not lines:
            return ""

        lines = [f"{i + 1:6d}|{line}" for i, line in enumerate(lines)]
        char_limit = min(char_limit or self.max_char_limit, self.max_char_limit)
        selected_lines = paginate(
            items=lines, max_chars=char_limit, offset=start_line - 1, limit=num_lines
        )

        info = f"Viewing {len(selected_lines)} lines of {file_path} starting from line {start_line}. "

        if len(selected_lines) < len(lines):
            info += (
                f"The output has been truncated to less than {char_limit} characters. "
            )

        return info + "\n\n" + "\n".join(selected_lines)
