"""A compact, tool-using DABstep baseline built on the Chat Completions API.

Derived from the officeqa seed, which shares the shape: explore a fixed local
corpus with a shell, then write one exact answer to a file. Three deliberate
departures from that template, each recorded because a reviewer will ask:

* No ``read_image``. officeqa ships one against a text-only pinned target, so
  every call returns ``400 This model does not support image inputs``, which is
  not in the retry class and scores the case at the failure value. DABstep's
  corpus is CSV, JSON and Markdown, so the tool has no purpose here either way.
* An empty model turn continues with a nudge instead of raising, matching gaia.
  A raise ends the trial at the failure value; a nudge lets the case score
  best-effort.
* Token accounting runs in ``finally``, so a crashed trial still reports its
  usage. Assigning it on the last line of ``run`` (every other seed) means a
  raise discards the record, which biases per-trial cost low on exactly the
  expensive cases.
"""

from __future__ import annotations

import json
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI


def _is_reasoning_model(model: str) -> bool:
    """Whether `model` is an OpenAI reasoning model.

    Capability, not provider: Azure gpt-4o is not Fireworks yet still rejects
    reasoning_effort, and every gpt-5 model rejects max_tokens. Fireworks-served
    open models match none of these prefixes, so they keep the legacy shape.
    """
    name = model.lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4")) or "codex" in name


# Measured on the 33-case development partition, 2026-07-30: 9.1s per turn, so the
# 1800s case clock this dataset declares would allow roughly 197 turns. 24 (copied
# from officeqa, whose own value came from gaia's 600s clock, three times shorter
# than ours) truncated 18 of 33 cases and forced a best-effort answer on 17 of
# them. 40 matches swe-atlas-qna, brings the median case under the cap, and costs
# 2-3x tokens rather than the ~50x that filling the clock would: history is resent
# every turn, and the seed already spends a median 419k input tokens per case.
#
# Deliberately not sized so the clock binds. On this benchmark the binding budget
# is tokens, not wall time, and reward does not price tokens (CONFIGURATION.md:
# accuracy is the only input to selection). Leaving the cap where the seed is
# truncated on half its cases would hand the optimizer a large one-integer win that
# generalises to nothing.
MAX_TURNS = 40
MAX_TOOL_OUTPUT_CHARS = 20_000
SHELL_TIMEOUT_SEC = 120

INSTRUCTIONS = """You are a careful data analyst answering factoid questions about a
fixed set of payment files (DABstep).

The data lives in /app/data/ and includes documentation. Read the documentation before
you reason about the tables: several fields mean something other than their name
suggests, and the graders' expected answers follow the documented definitions. Run
shell commands in the task's Linux environment to inspect files and compute with
Python; pandas is installed.

Answer formatting decides many cases. Give exactly what the question asks for and
nothing else: no units unless requested, no thousands separators unless requested, and
the rounding the question specifies. If the question says a particular string should be
used when nothing applies, use that string verbatim.

The grader reads /app/answer.txt. When ready, call submit_answer with only the exact
answer: no explanation, label, markdown, or surrounding prose. Never modify benchmark
tests, verifier files, or expected answers.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a non-interactive shell command inside the DABstep task "
                "environment. The working directory is /app; the data and its "
                "documentation are under /app/data/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Write the exact final answer and finish the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Exact answer only, with no explanation",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


class DabstepAgent(BaseAgent):
    """Data-analysis agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "dabstep-chat-baseline"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("DABstep agent requires a Harbor model")
        self._api_model = self.model_name.removeprefix("openai/")
        self._client = AsyncOpenAI(max_retries=8)

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec("mkdir -p /app", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "could not prepare /app")

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= MAX_TOOL_OUTPUT_CHARS:
            return value
        half = MAX_TOOL_OUTPUT_CHARS // 2
        omitted = len(value) - (2 * half)
        return f"{value[:half]}\n...[{omitted} characters omitted]...\n{value[-half:]}"

    def _trace(self, event: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with (self.logs_dir / "dabstep-trace.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    async def _run_shell(
        self, environment: BaseEnvironment, command: str
    ) -> dict[str, Any]:
        result = await environment.exec(
            command, cwd="/app", timeout_sec=SHELL_TIMEOUT_SEC
        )
        return {
            "return_code": result.return_code,
            "stdout": self._truncate(result.stdout or ""),
            "stderr": self._truncate(result.stderr or ""),
        }

    async def _submit(self, environment: BaseEnvironment, answer: str) -> None:
        normalized = answer.strip()
        if not normalized:
            raise ValueError("answer must not be empty")
        local_path = self.logs_dir / "answer.txt"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # The scorer reads the first line of /app/answer.txt, so a single line.
        local_path.write_text(normalized.splitlines()[0] + "\n", encoding="utf-8")
        await environment.upload_file(local_path, "/app/answer.txt")

    @staticmethod
    def _usage_value(value: Any, name: str) -> int:
        result = getattr(value, name, 0) if value is not None else 0
        return int(result or 0)

    def _completion_kwargs(
        self, messages: list[dict[str, Any]], *, tools: bool = True
    ) -> dict[str, Any]:
        # Reasoning models replaced max_tokens with max_completion_tokens and
        # reject the old name outright. Same capability test as the
        # reasoning_effort gate below, so the two stay consistent.
        token_limit_key = (
            "max_completion_tokens"
            if _is_reasoning_model(self._api_model)
            else "max_tokens"
        )
        kwargs: dict[str, Any] = {"model": self._api_model, "messages": messages}
        kwargs[token_limit_key] = 8000
        if tools:
            kwargs["tools"] = TOOLS
        if _is_reasoning_model(self._api_model):
            kwargs["reasoning_effort"] = "medium"
        # parallel_tool_calls is a separate axis: Fireworks-served models reject
        # it, but gpt-4o supports it, so this one stays a provider check.
        if "fireworks" not in self._api_model:
            kwargs["parallel_tool_calls"] = False
        return kwargs

    def _account(self, usage: Any, totals: dict[str, int]) -> None:
        totals["input"] += self._usage_value(usage, "prompt_tokens")
        totals["output"] += self._usage_value(usage, "completion_tokens")
        totals["cached"] += self._usage_value(
            getattr(usage, "prompt_tokens_details", None), "cached_tokens"
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Stateless Chat Completions: the full message history is resent each
        # turn (provider prompt-caching handles the repeated prefix), which works
        # across every provider, unlike the OpenAI-only Responses API.
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": instruction},
        ]
        totals = {"input": 0, "output": 0, "cached": 0}
        turns = 0

        try:
            for turn in range(1, MAX_TURNS + 1):
                turns = turn
                response = await self._client.chat.completions.create(
                    **self._completion_kwargs(messages)
                )
                self._account(response.usage, totals)
                message = response.choices[0].message
                calls = message.tool_calls or []
                self._trace(
                    {
                        "turn": turn,
                        "content": message.content,
                        "tool_calls": [
                            {"name": c.function.name, "arguments": c.function.arguments}
                            for c in calls
                        ],
                    }
                )
                assistant: dict[str, Any] = {"role": "assistant"}
                if message.content:
                    assistant["content"] = message.content
                if calls:
                    assistant["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ]
                messages.append(assistant)

                if not calls:
                    text = (message.content or "").strip()
                    if text:
                        await self._submit(environment, text)
                        context.metadata = {
                            "turns": turn,
                            "trace": "dabstep-trace.jsonl",
                        }
                        break
                    # No tool call and no text: the model only reasoned, or was
                    # truncated at the token limit. Carry the chain forward with a
                    # nudge instead of crashing, so the case can still score.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That turn produced no answer and no tool call. "
                                "Continue: either run a command or call "
                                "submit_answer."
                            ),
                        }
                    )
                    self._trace({"turn": turn, "empty_turn": True})
                    continue

                submitted = False
                for call in calls:
                    # The dispatch runs inside the try rather than an else: the
                    # argument lookups are the likelier failure, since the tool
                    # schemas are not strict and the model can return valid JSON
                    # that omits a required key. Feed that back as a tool error
                    # instead of letting a KeyError end the whole trial.
                    try:
                        arguments = json.loads(call.function.arguments)
                        if call.function.name == "run_shell":
                            result: dict[str, Any] = await self._run_shell(
                                environment, arguments["command"]
                            )
                        elif call.function.name == "submit_answer":
                            await self._submit(environment, arguments["answer"])
                            result = {"submitted": True}
                            submitted = True
                        else:
                            result = {"error": f"unknown tool: {call.function.name}"}
                    except (
                        json.JSONDecodeError,
                        KeyError,
                        OSError,
                        TypeError,
                        ValueError,
                    ) as error:
                        result = {"error": f"invalid arguments: {error}"}
                    self._trace(
                        {"turn": turn, "tool": call.function.name, "result": result}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    if submitted:
                        break
                if submitted:
                    context.metadata = {"turns": turn, "trace": "dabstep-trace.jsonl"}
                    break
            else:
                # Turn budget exhausted: force one final tool-free answer from
                # what was gathered rather than crashing, so the case scores
                # best-effort instead of being lost with no answer recorded.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have used your full analysis budget. Give your "
                            "single best final answer now, formatted exactly as the "
                            "question asks, based on what you have computed."
                        ),
                    }
                )
                final = await self._client.chat.completions.create(
                    **self._completion_kwargs(messages, tools=False)
                )
                self._account(final.usage, totals)
                answer = (final.choices[0].message.content or "").strip() or (
                    "Not Applicable"
                )
                await self._submit(environment, answer)
                self._trace({"turn": MAX_TURNS, "forced_final_answer": answer})
                context.metadata = {
                    "turns": MAX_TURNS,
                    "trace": "dabstep-trace.jsonl",
                    "forced_final": True,
                }
        finally:
            # In `finally` so a raised trial still reports its usage; see the
            # module docstring.
            context.n_input_tokens = totals["input"]
            context.n_output_tokens = totals["output"]
            context.n_cache_tokens = totals["cached"]
            if context.metadata is None:
                context.metadata = {"turns": turns, "trace": "dabstep-trace.jsonl"}
