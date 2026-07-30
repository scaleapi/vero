"""A minimal shell-loop terminal agent: the editable optimization target.

A clean-room re-implementation of the mini-SWE-agent design (SWE-agent project,
MIT), not a copy of its source. The design is deliberately spare, and that is the
point: this is a *seed*, and a benchmark that keeps the model frozen only measures
anything if editing the harness has room to move the score.

The shape, which is mini-SWE-agent's:

* One bash command per turn. No other tools, no subagents, no planning phase.
* **No function calling.** The model replies in prose and the command is pulled
  out of a fenced ``bash`` block. This is a real design choice of the original --
  it makes the agent work against any completion endpoint -- and it is also
  visibly brittle, which is a fair thing for an optimizer to notice.
* Linear, ever-growing history. Nothing is summarised, compacted or dropped.
* A fixed step budget and a fixed per-command output cap.

Known weaknesses, left in on purpose. Each is a plausible thing for an optimizer
to find and fix, and none of them is a bug -- the agent works:

* History grows without bound, so long tasks can crowd the context window.
* One command per turn wastes turns on sequences that could be batched.
* Command extraction is regex-on-markdown and fails on unfenced replies.
* The step budget is a constant, unrelated to the task's actual clock.
* Output is truncated head-and-tail at a fixed size, losing the middle.
* Nothing verifies the work before finishing.

Terminal-Bench decides pass/fail by running each task's own tests against the
container's final state. There is no answer file to write (1 of 89 tasks even
mentions one), so "finishing" means leaving the environment correct and stopping.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI

MAX_STEPS = 40
MAX_OUTPUT_CHARS = 8_000
COMMAND_TIMEOUT_SEC = 300
FINISH_SENTINEL = "TASK_COMPLETE"

#: Fenced ``bash``/``sh`` block, or a bare fence as a fallback. Ordered so a
#: language-tagged block wins over an untagged one in the same reply.
_COMMAND_BLOCK = re.compile(r"```(?:bash|sh)\s*\n(.*?)```|```\s*\n(.*?)```", re.DOTALL)

SYSTEM_PROMPT = f"""You are a terminal agent working inside a Linux container.

You solve the task by running shell commands, one per reply. Reply with a short
sentence saying what you are doing next, then exactly one fenced bash block:

```bash
your command here
```

Rules:
- Exactly one bash block per reply. It runs non-interactively in /app.
- Interactive programs will hang. Prefer flags that avoid prompts.
- Install what you need; the container has network access.
- When the task is fully done, reply with a bash block that runs:
  echo {FINISH_SENTINEL}
- Do not modify or delete the task's tests. Your work is graded by running them.
"""


class TerminalBenchAgent(BaseAgent):
    """Shell-loop agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "terminal-bench-shell-baseline"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("Terminal-Bench agent requires a Harbor model")
        # The gateway allow-lists the model under the name the upstream serves, so
        # send that exact name. Harbor may hand us a provider-prefixed form.
        self._api_model = self.model_name.removeprefix("openai/")
        # Fail closed. OPENAI_* in this container can point at the unmetered
        # upstream, so falling back to it would silently bypass metering and the
        # per-scope allow-list rather than erroring.
        gateway_key = os.environ.get("VERO_AGENT_INFERENCE_API_KEY")
        gateway_url = os.environ.get("VERO_AGENT_INFERENCE_BASE_URL")
        if not gateway_key or not gateway_url:
            raise RuntimeError(
                "Terminal-Bench target inference requires "
                "VERO_AGENT_INFERENCE_API_KEY and VERO_AGENT_INFERENCE_BASE_URL; "
                "refusing to fall back to OPENAI_*, which points at the "
                "unmetered upstream"
            )
        # Absorb transient 429s in-client: an unretried rate limit inside a trial
        # scores the failure value, so it costs a candidate a 0.0 outright.
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
    def _extract_command(reply: str) -> str | None:
        """The single shell command in `reply`, or None if there is not one."""
        match = _COMMAND_BLOCK.search(reply or "")
        if match is None:
            return None
        command = (match.group(1) or match.group(2) or "").strip()
        return command or None

    def _trace(self, event: dict[str, Any]) -> None:
        """Append one JSONL record, so a failure can be diagnosed after the run.

        The optimizer reads these to work out *why* a case failed; without them a
        zero is indistinguishable from a crash.
        """
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "terminal-bench-trace.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    async def _complete(self, messages: list[dict[str, str]]) -> str:
        response = await self._client.chat.completions.create(
            model=self._api_model,
            messages=messages,  # type: ignore[arg-type]
        )
        choices = getattr(response, "choices", None)
        if not choices:
            # A gateway can answer 200 with a body the SDK does not turn into a
            # completion. Reject it here rather than crashing on attribute access.
            raise RuntimeError("upstream returned no completion choices")
        return choices[0].message.content or ""

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        for step in range(1, MAX_STEPS + 1):
            reply = await self._complete(messages)
            messages.append({"role": "assistant", "content": reply})
            command = self._extract_command(reply)

            if command is None:
                self._trace({"step": step, "no_command": True})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "No bash block found. Reply with exactly one fenced "
                            "bash block containing the next command."
                        ),
                    }
                )
                continue

            if FINISH_SENTINEL in command:
                self._trace({"step": step, "finished": True})
                return

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
                    "role": "user",
                    "content": (
                        f"exit={result.return_code}\n"
                        f"--- stdout ---\n{stdout}\n"
                        f"--- stderr ---\n{stderr}"
                    ),
                }
            )

        # Out of steps. Nothing to submit -- the container's state is the answer --
        # so record it and stop. An optimizer that notices tasks ending here has a
        # clear lead: the budget is a constant, not a function of the task.
        self._trace({"step": MAX_STEPS, "exhausted_steps": True})
