"""Shell-loop terminal agent: the editable optimization target.

One shell command per turn via function calling. Grading runs each task's own
tests against the container's final state, so there is nothing to submit.
"""

from __future__ import annotations

import json
import os
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI, BadRequestError

MAX_STEPS = 40
MAX_OUTPUT_CHARS = 8_000
COMMAND_TIMEOUT_SEC = 300

# Providers word a full context window differently; match on any of them.
_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "prompt is too long",
    "reduce the length",
    "input length and `max_tokens` exceed",
)


def _is_context_overflow(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _OVERFLOW_MARKERS)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run one non-interactive shell command in the task container, "
                "from /app. Interactive programs will hang."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call when the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """You are a terminal agent working inside a Linux container.

Solve the task by calling run_shell, one command at a time, reading its output
before deciding the next command. Install whatever you need; the container has
network access.

Your work is graded by running the task's own tests against the container's final
state, so leave the environment correct. Do not modify or delete those tests.

Call finish when the task is complete.
"""


class TerminalBenchAgent(BaseAgent):
    """Shell-loop agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "terminal-bench-shell-baseline"

    @override
    def version(self) -> str:
        return "0.2.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("Terminal-Bench agent requires a Harbor model")
        self._api_model = self.model_name.removeprefix("openai/")
        # Dedicated names first; they carry the gateway when OPENAI_* is upstream.
        gateway_key = os.environ.get("VERO_AGENT_INFERENCE_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        gateway_url = os.environ.get("VERO_AGENT_INFERENCE_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL"
        )
        if not gateway_key or not gateway_url:
            raise RuntimeError(
                "Terminal-Bench target inference requires a scoped API key and "
                "base URL on VERO_AGENT_INFERENCE_* or OPENAI_*"
            )
        # An unretried rate limit inside a trial scores the failure value.
        self._client = AsyncOpenAI(
            api_key=gateway_key, base_url=gateway_url, max_retries=8
        )

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec("mkdir -p /app", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "could not prepare /app")

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= MAX_OUTPUT_CHARS:
            return value
        half = MAX_OUTPUT_CHARS // 2
        omitted = len(value) - (2 * half)
        return f"{value[:half]}\n...[{omitted} characters omitted]...\n{value[-half:]}"

    @staticmethod
    def _command_of(call: Any) -> str | None:
        """The command in a run_shell call, or None if the arguments are unusable."""
        raw = getattr(getattr(call, "function", None), "arguments", None) or "{}"
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        command = parsed.get("command") if isinstance(parsed, dict) else None
        if not isinstance(command, str) or not command.strip():
            return None
        return command

    def _trace(self, event: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "terminal-bench-trace.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    async def _complete(self, messages: list[dict[str, Any]]) -> Any | None:
        """The next assistant message, or None if the context window is full.

        None rather than an exception: a full window is a property of the harness,
        so it should land as a low score, not an infrastructure failure. swe-atlas
        lost 5 trials this way (128k overflow, BadRequestError).
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._api_model,
                messages=messages,  # type: ignore[arg-type]
                tools=TOOLS,  # type: ignore[arg-type]
            )
        except BadRequestError as error:
            if _is_context_overflow(error):
                return None
            raise
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("upstream returned no completion choices")
        return choices[0].message

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        for step in range(1, MAX_STEPS + 1):
            message = await self._complete(messages)
            if message is None:
                self._trace({"step": step, "context_exhausted": True})
                return
            calls = list(getattr(message, "tool_calls", None) or [])
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in calls
                ]
            messages.append(entry)

            if not calls:
                self._trace({"step": step, "no_tool_call": True})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Call run_shell with the next command, or finish if "
                            "the task is complete."
                        ),
                    }
                )
                continue

            # Every call needs a reply or the next request is rejected.
            for extra in calls[1:]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": extra.id,
                        "content": "Skipped: issue one command per turn.",
                    }
                )

            call = calls[0]
            if call.function.name == "finish":
                self._trace({"step": step, "finished": True})
                return

            command = self._command_of(call)
            if command is None:
                self._trace({"step": step, "bad_arguments": True})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": 'Could not read a command. Send {"command": "..."}.',
                    }
                )
                continue

            result = await environment.exec(
                command, cwd="/app", timeout_sec=COMMAND_TIMEOUT_SEC
            )
            stdout = self._truncate(result.stdout or "")
            stderr = self._truncate(result.stderr or "")
            self._trace(
                {
                    "step": step,
                    "command": command,
                    "return_code": result.return_code,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": (
                        f"exit={result.return_code}\n"
                        f"--- stdout ---\n{stdout}\n"
                        f"--- stderr ---\n{stderr}"
                    ),
                }
            )

        self._trace({"step": MAX_STEPS, "exhausted_steps": True})
