import asyncio
import sys
from dataclasses import dataclass
from subprocess import CalledProcessError
from typing import Any, AsyncIterator, TypeVar

T = TypeVar("T")


@dataclass
class SubprocessResult:
    """Result of a subprocess execution, always includes captured output."""

    args: list[str]
    returncode: int | None  # None if process didn't complete
    stdout: str
    stderr: str
    pid: int | None = None  # Process ID, useful for debugging orphaned processes
    timed_out: bool = False
    cancelled: bool = False


class SubprocessTimeoutError(Exception):
    """Timeout with captured output preserved."""

    def __init__(self, result: SubprocessResult):
        self.result = result
        super().__init__(f"Subprocess timed out: {result.args}")


class SubprocessCancelledError(Exception):
    """Cancellation with captured output preserved."""

    def __init__(self, result: SubprocessResult):
        self.result = result
        super().__init__(f"Subprocess cancelled: {result.args}")


async def anext_with_timeout(it: AsyncIterator[T], timeout: int = 5) -> T:
    """Get the next item from an async iterator with a timeout."""
    async with asyncio.timeout(timeout):
        return await anext(it)


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


async def run_subprocess_with_tee(
    cmd: list[str],
    timeout: float | None = None,
    cwd: str | None = None,
    check: bool = False,
    flush: bool = False,
    chunk_size: int = 8192,
    tee_stdout: bool = True,
    tee_stderr: bool = True,
    **kwargs: Any,
) -> SubprocessResult:
    """Run a subprocess, streaming output to console while capturing it.

    Args:
        cmd: The command to run.
        timeout: Timeout in seconds (None = no timeout).
        cwd: Working directory for the subprocess.
        check: Raise CalledProcessError on non-zero exit.
        flush: Flush console output after each chunk.
        chunk_size: Bytes to read per iteration (default 8192).
        tee_stdout: Whether to print stdout to console.
        tee_stderr: Whether to print stderr to console.
        **kwargs: Additional arguments to asyncio.create_subprocess_exec.

    Returns:
        SubprocessResult with captured stdout/stderr and status.

    Raises:
        SubprocessTimeoutError: If timeout exceeded (includes partial output).
        SubprocessCancelledError: If cancelled (includes partial output).
        CalledProcessError: If check=True and non-zero exit code.
    """
    process: asyncio.subprocess.Process | None = None
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def build_result(
        returncode: int | None = None,
        timed_out: bool = False,
        cancelled: bool = False,
    ) -> SubprocessResult:
        return SubprocessResult(
            args=cmd,
            returncode=returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            pid=process.pid if process else None,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    async def read_stream(
        stream: asyncio.StreamReader | None,
        chunks: list[str],
        output_stream,
        tee: bool,
    ):
        if not stream:
            return

        while True:
            chunk = await stream.read(chunk_size)
            if not chunk:
                break
            decoded = chunk.decode(errors="replace")
            chunks.append(decoded)
            if tee:
                output_stream.write(decoded)
                if flush:
                    output_stream.flush()

    async def terminate_process_safe():
        """Terminate process, handling edge cases."""
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        except ProcessLookupError:
            pass  # Already dead

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            **kwargs,
        )

        async def run_and_wait():
            # Run stream readers as tasks so we can cancel them
            stdout_task = asyncio.create_task(
                read_stream(process.stdout, stdout_chunks, sys.stdout, tee_stdout)
            )
            stderr_task = asyncio.create_task(
                read_stream(process.stderr, stderr_chunks, sys.stderr, tee_stderr)
            )

            try:
                # Wait for process to exit (don't wait for streams - child processes may hold them open)
                await process.wait()
            finally:
                # Always cleanup stream tasks, even on cancellation/timeout
                _, pending = await asyncio.wait(
                    [stdout_task, stderr_task],
                    timeout=1.0,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass

        if timeout is not None:
            await asyncio.wait_for(run_and_wait(), timeout=timeout)
        else:
            await run_and_wait()

    except asyncio.TimeoutError:
        # Terminate but preserve partial output
        try:
            await asyncio.shield(terminate_process_safe())
        except asyncio.CancelledError:
            # Shield was pierced, do sync cleanup
            if process and process.returncode is None:
                process.kill()
        raise SubprocessTimeoutError(
            build_result(
                returncode=process.returncode if process else None,
                timed_out=True,
            )
        )

    except asyncio.CancelledError:
        try:
            await asyncio.shield(terminate_process_safe())
        except asyncio.CancelledError:
            if process and process.returncode is None:
                process.kill()
        raise SubprocessCancelledError(
            build_result(
                returncode=process.returncode if process else None,
                cancelled=True,
            )
        )

    result = build_result(returncode=process.returncode)

    if check and process.returncode != 0:
        if process.returncode is None:
            returncode = -1
        else:
            returncode = process.returncode

        raise CalledProcessError(
            returncode=returncode,
            cmd=cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    return result
