"""A compact, tool-using SWE-bench-Pro baseline built on the OpenAI Responses API.

This module is the editable optimization target. A VeRO coding agent may rewrite
the prompts, control flow, tool definitions, patching strategy, or dependencies,
but it must keep the Harbor agent interface (``harbor.agents.base.BaseAgent``) and
must not touch the dataset, verifier, split, model, or test target.

The task-source verifier is what grades a solution: it applies whatever the agent
left in the repository and runs the task's own test suite for the reward. The agent
does NOT self-grade and does NOT write an answer file; it edits code in place and
calls ``submit`` to signal that it believes the task is complete.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI

# SWE-bench-Pro tasks build a real repository and run its test suite, so the agent
# needs more turns than a short-answer benchmark like GAIA.
MAX_TURNS = 50
MAX_TOOL_OUTPUT_CHARS = 20_000
MAX_FILE_READ_CHARS = 60_000

# Repository checkout location inside the task environment. SWE-bench-Pro task
# packages check the target project out here; adjust in one place if the pinned
# task source uses a different mount.
REPO_DIR = "/app/repo"

# Retry policy for the Responses API. The GAIA baseline scored 0.0 in the first
# VeRO run because a single transient error on ``responses.create`` crashed the
# whole rollout; the optimizer's winning fix was exactly this retry-with-backoff.
# It ships here from the start so the baseline is robust before optimization.
MAX_API_RETRIES = 6
API_RETRY_BASE_DELAY = 2.0
API_RETRY_MAX_DELAY = 60.0

INSTRUCTIONS = f"""You are a careful software engineer solving a SWE-bench-Pro task.

A real project is checked out at {REPO_DIR}. The task describes a bug to fix or a
feature to implement. Your job is to edit the repository so that the task's hidden
test suite passes. Work like an engineer: read the failing area, reproduce the
problem, make the smallest correct change, and re-run the relevant tests.

Tools available to you:
- run_shell: run non-interactive shell commands with the repo as the working dir.
- read_file: read a file (optionally a line range) from the repo.
- write_file: overwrite a file with new contents.
- apply_patch: apply a unified diff to the repo (prefer this for surgical edits).
- run_tests: run the project's test suite (or a targeted subset you name).
- submit: signal that you believe the task is complete.

Rules:
- Change only what the task requires. Do not rewrite unrelated code or reformat.
- Never edit, delete, or weaken the task's own tests, the verifier, or any grading
  files, and never hard-code expected outputs. The grader runs the hidden suite; a
  solution that only edits tests will be rejected.
- Verify your change with run_tests before calling submit. There is no answer file
  and no self-grading: the task-source verifier runs the suite for the reward.

When you are confident the change is correct and the tests you can see pass, call
submit. Do not merely describe what you would do."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "run_shell",
        "description": (
            "Run a non-interactive shell command with the repository at "
            f"{REPO_DIR} as the working directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"}
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a repository-relative file. Optionally restrict to a 1-indexed, "
            "inclusive line range to focus on a region."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Repo-relative path under {REPO_DIR}",
                },
                "start_line": {
                    "type": ["integer", "null"],
                    "description": "1-indexed first line to read, or null for start",
                },
                "end_line": {
                    "type": ["integer", "null"],
                    "description": "1-indexed last line to read, or null for end",
                },
            },
            "required": ["path", "start_line", "end_line"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "write_file",
        "description": (
            "Overwrite a repository-relative file with the given contents, creating "
            "parent directories as needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Repo-relative path under {REPO_DIR}",
                },
                "content": {"type": "string", "description": "Full new file contents"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "apply_patch",
        "description": (
            "Apply a unified diff to the repository with `git apply`. The diff must "
            "use repo-relative paths (a/ and b/ prefixes are accepted)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Unified diff text"}
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_tests",
        "description": (
            "Run the project's tests. Provide an optional command override (for "
            "example a single pytest node id) to target a subset; otherwise a "
            "sensible default is used."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": ["string", "null"],
                    "description": "Test command to run, or null for the default",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "submit",
        "description": (
            "Signal that the edits are complete and the task should be graded. "
            "Takes no arguments; the task-source verifier runs the hidden suite."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class SweBenchProAgent(BaseAgent):
    """Code-editing agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "swe-bench-pro-responses-baseline"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("SWE-bench-Pro agent requires a Harbor model")
        self._api_model = self.model_name.removeprefix("openai/")
        # The metered per-evaluation gateway arrives on dedicated variables when
        # VeRO relocates the upstream key; fall back to OPENAI_* for a plain run.
        self._client = AsyncOpenAI(
            api_key=os.environ.get("VERO_AGENT_INFERENCE_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("VERO_AGENT_INFERENCE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL"),
        )

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(f"test -d {REPO_DIR}", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(
                result.stderr or f"task repository is missing at {REPO_DIR}"
            )

    async def _responses_create(self, **request: Any) -> Any:
        """Call the Responses API with retry-and-backoff on transient errors.

        This is the load-bearing robustness fix: an unguarded ``create`` call is
        what made the first GAIA baseline score 0.0. Any exception is retried with
        exponential backoff and jitter up to ``MAX_API_RETRIES``; the final failure
        is re-raised so a genuine, persistent error still surfaces.
        """
        last_error: Exception | None = None
        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                return await self._client.responses.create(**request)
            except Exception as error:  # noqa: BLE001 - transient classes vary by SDK
                last_error = error
                if attempt == MAX_API_RETRIES:
                    break
                delay = min(
                    API_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    API_RETRY_MAX_DELAY,
                )
                delay += random.uniform(0, delay / 2)
                self._trace(
                    {
                        "event": "responses_retry",
                        "attempt": attempt,
                        "delay": round(delay, 2),
                        "error": repr(error),
                    }
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= MAX_TOOL_OUTPUT_CHARS:
            return value
        half = MAX_TOOL_OUTPUT_CHARS // 2
        omitted = len(value) - (2 * half)
        return f"{value[:half]}\n...[{omitted} characters omitted]...\n{value[-half:]}"

    def _trace(self, event: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.logs_dir / "swe-bench-pro-trace.jsonl"
        with trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    async def _run_shell(
        self, environment: BaseEnvironment, command: str, timeout_sec: int = 300
    ) -> dict[str, Any]:
        result = await environment.exec(command, cwd=REPO_DIR, timeout_sec=timeout_sec)
        return {
            "return_code": result.return_code,
            "stdout": self._truncate(result.stdout or ""),
            "stderr": self._truncate(result.stderr or ""),
        }

    async def _read_file(
        self,
        environment: BaseEnvironment,
        path: str,
        start_line: int | None,
        end_line: int | None,
    ) -> dict[str, Any]:
        # `cat -A`-free read via shell so we do not need a download round-trip.
        result = await environment.exec(
            f"cat -- {json.dumps(path)}", cwd=REPO_DIR, timeout_sec=60
        )
        if result.return_code != 0:
            return {"error": self._truncate(result.stderr or "could not read file")}
        lines = (result.stdout or "").splitlines()
        total = len(lines)
        lo = max(1, start_line or 1)
        hi = min(total, end_line or total)
        selected = lines[lo - 1 : hi] if total else []
        body = "\n".join(selected)
        if len(body) > MAX_FILE_READ_CHARS:
            body = body[:MAX_FILE_READ_CHARS] + "\n...[truncated]..."
        return {"path": path, "start_line": lo, "end_line": hi, "content": body}

    async def _write_file(
        self, environment: BaseEnvironment, path: str, content: str
    ) -> dict[str, Any]:
        local_path = self.logs_dir / "staged" / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")
        remote_path = f"{REPO_DIR}/{path}"
        mkdir = await environment.exec(
            f"mkdir -p -- {json.dumps(os.path.dirname(remote_path) or REPO_DIR)}",
            timeout_sec=30,
        )
        if mkdir.return_code != 0:
            return {"error": self._truncate(mkdir.stderr or "could not create dirs")}
        await environment.upload_file(local_path, remote_path)
        return {"written": path, "bytes": len(content.encode("utf-8"))}

    async def _apply_patch(
        self, environment: BaseEnvironment, patch: str
    ) -> dict[str, Any]:
        local_path = self.logs_dir / "patches" / f"patch-{self._patch_index}.diff"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(patch if patch.endswith("\n") else patch + "\n", "utf-8")
        remote_path = "/tmp/swebp-agent.patch"
        await environment.upload_file(local_path, remote_path)
        self._patch_index += 1
        result = await environment.exec(
            f"git apply --whitespace=nowarn {remote_path}",
            cwd=REPO_DIR,
            timeout_sec=120,
        )
        if result.return_code != 0:
            # Retry with a looser fuzz factor before giving up.
            result = await environment.exec(
                f"git apply --whitespace=nowarn -C1 {remote_path}",
                cwd=REPO_DIR,
                timeout_sec=120,
            )
        return {
            "applied": result.return_code == 0,
            "return_code": result.return_code,
            "stderr": self._truncate(result.stderr or ""),
        }

    async def _run_tests(
        self, environment: BaseEnvironment, command: str | None
    ) -> dict[str, Any]:
        # Default is intentionally generic; the pinned task source may ship its own
        # entrypoint. The verifier runs the authoritative suite regardless.
        test_command = command or "python -m pytest -q"
        return await self._run_shell(environment, test_command, timeout_sec=1200)

    @staticmethod
    def _usage_value(value: Any, name: str) -> int:
        result = getattr(value, name, 0) if value is not None else 0
        return int(result or 0)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._patch_index = 0
        next_input: Any = instruction
        previous_response_id: str | None = None
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        for turn in range(1, MAX_TURNS + 1):
            request: dict[str, Any] = {
                "model": self._api_model,
                "instructions": INSTRUCTIONS,
                "input": next_input,
                "tools": TOOLS,
                "reasoning": {"effort": "high"},
                "max_output_tokens": 12_000,
                "parallel_tool_calls": False,
            }
            if previous_response_id is not None:
                request["previous_response_id"] = previous_response_id
            response = await self._responses_create(**request)
            usage = response.usage
            input_tokens += self._usage_value(usage, "input_tokens")
            output_tokens += self._usage_value(usage, "output_tokens")
            cached_tokens += self._usage_value(
                getattr(usage, "input_tokens_details", None), "cached_tokens"
            )

            calls = [item for item in response.output if item.type == "function_call"]
            self._trace(
                {
                    "turn": turn,
                    "response_id": response.id,
                    "output_text": response.output_text,
                    "function_calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in calls
                    ],
                }
            )
            if not calls:
                # No tool call and no submit: the model believes it is done. The
                # verifier grades the repository state as-is; nothing to write.
                context.metadata = {"turns": turn, "trace": "swe-bench-pro-trace.jsonl"}
                break

            next_input = []
            submitted = False
            for call in calls:
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError as error:
                    result: dict[str, Any] = {"error": f"invalid arguments: {error}"}
                else:
                    if call.name == "run_shell":
                        result = await self._run_shell(
                            environment, arguments["command"]
                        )
                    elif call.name == "read_file":
                        result = await self._read_file(
                            environment,
                            arguments["path"],
                            arguments.get("start_line"),
                            arguments.get("end_line"),
                        )
                    elif call.name == "write_file":
                        result = await self._write_file(
                            environment, arguments["path"], arguments["content"]
                        )
                    elif call.name == "apply_patch":
                        result = await self._apply_patch(
                            environment, arguments["patch"]
                        )
                    elif call.name == "run_tests":
                        result = await self._run_tests(
                            environment, arguments.get("command")
                        )
                    elif call.name == "submit":
                        result = {"submitted": True}
                        submitted = True
                    else:
                        result = {"error": f"unknown tool: {call.name}"}
                self._trace({"turn": turn, "tool": call.name, "result": result})
                next_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
                if submitted:
                    break
            if submitted:
                context.metadata = {"turns": turn, "trace": "swe-bench-pro-trace.jsonl"}
                break
            previous_response_id = response.id
        else:
            raise RuntimeError(f"SWE-bench-Pro agent exceeded {MAX_TURNS} turns")

        context.n_input_tokens = input_tokens
        context.n_output_tokens = output_tokens
        context.n_cache_tokens = cached_tokens
