from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from vero.exceptions import FileNotTrackedError, InputTooLongError, StringNotFoundError
from vero.tools.base import FileSystemWriteBase
from vero.tools.utils import is_tool


class FileWriteToolResult(NamedTuple):
    message: str
    bytes_written: int
    file_path: str


class FileEditToolResult(NamedTuple):
    message: str
    replacements: int
    file_path: str


@dataclass
class FileWrite(FileSystemWriteBase):
    """Write or edit files. Auto-commits changes for observability."""

    content_char_limit: int = 2_000_000

    async def _write_file(self, file_path: str, content: str) -> FileWriteToolResult:
        """Helper to write content to a file, creating it if it doesn't exist or overwriting if it does."""

        absolute_path = self.workspace.validate_write(file_path)
        file_exists = await self.sandbox.exists(absolute_path)

        if not await self._is_file_tracked(absolute_path) and file_exists:
            raise FileNotTrackedError(
                f"File '{absolute_path}' exists and is not tracked by the repository. It cannot be overwritten."
            )

        # Check content length
        if len(content) > self.content_char_limit:
            raise InputTooLongError(
                f"The content is too long. It must be less than {self.content_char_limit} characters."
            )

        if file_exists and not await self.sandbox.is_file(absolute_path):
            raise IsADirectoryError(f"'{absolute_path}' exists but is not a file.")

        # Write the content to the file (mkdir handled by sandbox)
        await self.sandbox.write_file(absolute_path, content)

        # Get the file size in bytes
        st = await self.sandbox.stat(absolute_path)
        bytes_written = st.st_size

        # Generate success message
        action = "Overwrote" if file_exists else "Created"
        message = f"{action} file '{absolute_path}' with {bytes_written} bytes."

        return FileWriteToolResult(
            message=message,
            bytes_written=bytes_written,
            file_path=str(absolute_path),
        )

    async def _edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> FileEditToolResult:
        """Helper to replace text in a file."""

        absolute_path = self.workspace.validate_write(file_path)

        if not await self._is_file_tracked(absolute_path):
            raise FileNotTrackedError(
                f"File '{absolute_path}' is not tracked by the repository and cannot be edited."
            )

        if len(new_string) > self.content_char_limit:
            raise InputTooLongError(
                f"new_string is too long. Must be less than {self.content_char_limit} characters. Got {len(new_string)} characters."
            )

        if old_string == new_string:
            raise ValueError("old_string and new_string are identical! No changes made.")

        if not await self.sandbox.exists(absolute_path):
            raise FileNotFoundError(f"File '{absolute_path}' does not exist.")

        if not await self.sandbox.is_file(absolute_path):
            raise IsADirectoryError(f"'{absolute_path}' is not a file.")

        # Read the file
        content = await self.sandbox.read_file(absolute_path)

        # Validate that old_string exists in the file
        if old_string not in content:
            raise StringNotFoundError(
                f"The string to replace `old_string` was not found in '{absolute_path}'. For complex replacements, consider using `write_file` to rewrite the file."
            )

        # Perform the replacement
        if replace_all:
            replacements = content.count(old_string)
            new_content = content.replace(old_string, new_string)
        else:
            replacements = 1
            new_content = content.replace(old_string, new_string, 1)

        # Write the file back
        await self.sandbox.write_file(absolute_path, new_content)

        # Generate success message
        occurrence_text = "occurrences" if replacements > 1 else "occurrence"
        message = f"Successfully replaced {replacements} {occurrence_text} in '{absolute_path}'."

        return FileEditToolResult(
            message=message,
            replacements=replacements,
            file_path=str(absolute_path),
        )

    @is_tool
    async def write_file(self, commit_message: str, file_path: str, content: str) -> str:
        """Write content to a file, creating it if it doesn't exist or overwriting if it does.

        Args:
            commit_message: The message to commit the changes with
            file_path: The path to the file to write
            content: The content to write to the file

        Returns:
            String message indicating the success or error of the operation
        """
        output = await self.run_and_commit(self._write_file(file_path, content), commit_message)
        return f"Created a new commit {output.commit}. {output.result.message}"

    @is_tool
    async def edit_file(
        self,
        commit_message: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace text in a file.

        Args:
            commit_message: The message to commit the changes with
            file_path: The absolute path to the file to modify
            old_string: The text to replace
            new_string: The text to replace it with
            replace_all: Replace all occurrences (default False)

        Returns:
            String message indicating the success or error of the operation
        """
        output = await self.run_and_commit(
            self._edit_file(file_path, old_string, new_string, replace_all), commit_message
        )
        return f"Created a new commit {output.commit}. {output.result.message}"
