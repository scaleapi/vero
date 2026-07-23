"""Function tools that drive an SDK ``SandboxSession``.

These give the coding agent a real shell plus file read/write that execute
INSIDE the sandbox — so they inherit the sandbox's isolation, workspace binding
(``Manifest(root=...)``), and containment (the sandbox client is the seam:
unix-local / docker / modal / e2b). Crucially they are *plain function tools*,
not the SDK's hosted ``Shell``/``Filesystem`` capabilities, so they work with
ANY model/provider over ChatCompletions *or* Responses. The hosted capabilities
would restrict the native path to the OpenAI Responses API.

All operations go through ``session.exec`` (a real shell), which keeps the tool
surface tiny and provider-neutral.
"""

from __future__ import annotations

import base64
import shlex
from typing import TYPE_CHECKING

from agents import function_tool

if TYPE_CHECKING:
    from agents.sandbox.session.sandbox_session import SandboxSession


def _format(result: object) -> str:
    """Render an ExecResult as text (stdout, optional stderr, non-zero exit)."""
    out = getattr(result, "stdout", None)
    if out is None:
        out = getattr(result, "output", "")
    if isinstance(out, (bytes, bytearray)):
        out = bytes(out).decode(errors="replace")
    err = getattr(result, "stderr", "") or ""
    if isinstance(err, (bytes, bytearray)):
        err = bytes(err).decode(errors="replace")
    code = getattr(result, "exit_code", getattr(result, "returncode", 0))
    text = str(out)
    if str(err).strip():
        text += f"\n[stderr]\n{err}"
    if code not in (0, None):
        text = f"[exit={code}]\n{text}"
    return text or f"[exit={code}]"


def build_sandbox_tools(session: "SandboxSession") -> list:
    """Build shell / read_file / write_file function tools bound to ``session``."""

    @function_tool
    async def shell(command: str) -> str:
        """Run a shell command in the program workspace and return its output.

        The working directory is the program's workspace root. Use this to run
        the program, tests, git, package managers, and any Unix tooling.
        """
        return _format(await session.exec(command))

    @function_tool
    async def read_file(path: str) -> str:
        """Read a UTF-8 text file. ``path`` is relative to the workspace root."""
        return _format(await session.exec(f"cat -- {shlex.quote(path)}"))

    @function_tool
    async def write_file(path: str, content: str) -> str:
        """Create or overwrite a file with ``content``.

        ``path`` is relative to the workspace root; parent directories are
        created as needed.
        """
        encoded = base64.b64encode(content.encode()).decode()
        quoted = shlex.quote(path)
        command = (
            f'mkdir -p "$(dirname -- {quoted})" && '
            f"printf %s {shlex.quote(encoded)} | base64 -d > {quoted} && "
            f"printf 'wrote %s bytes to %s\\n' {len(content)} {quoted}"
        )
        return _format(await session.exec(command))

    return [shell, read_file, write_file]
