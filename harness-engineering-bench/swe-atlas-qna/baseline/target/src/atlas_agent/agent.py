"""A compact codebase investigation agent built on the Chat Completions API."""

from __future__ import annotations

import json
import os
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI

MAX_TURNS = 40
MAX_TOOL_OUTPUT_CHARS = 30_000

INSTRUCTIONS = """You investigate software repositories and answer deep codebase questions.

The task repository is checked out at /app. Gather concrete evidence before answering:
map the relevant subsystem, search definitions and call sites, read tests and history when
useful, and run focused read-only experiments if the question asks for observed behavior.
Do not modify source files. Distinguish what the code proves from your own inferences.

Your final answer should be comprehensive and precise. Cite repository-relative paths,
symbols, and line numbers wherever they support a claim. When finished, call
submit_answer with the complete answer; do not merely describe what you would inspect.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a non-interactive command in the repository at /app. Use this "
                "for read-only exploration and temporary experiments; never edit "
                "repository files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Write the complete evidence-backed answer and finish.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Complete final answer",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


class AtlasAgent(BaseAgent):
    """Repository Q&A agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "swe-atlas-responses-baseline"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("SWE-Atlas agent requires a Harbor model")
        self._api_model = self.model_name.removeprefix("openai/")
        # The metered per-evaluation gateway arrives on dedicated variables
        # (OPENAI_* carries the upstream for the task's rubric judge instead).
        self._client = AsyncOpenAI(
            max_retries=8,
            api_key=os.environ.get("VERO_AGENT_INFERENCE_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("VERO_AGENT_INFERENCE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL"),
        )

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec("test -d /app", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "task repository is missing at /app")

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= MAX_TOOL_OUTPUT_CHARS:
            return value
        half = MAX_TOOL_OUTPUT_CHARS // 2
        omitted = len(value) - (2 * half)
        return f"{value[:half]}\n...[{omitted} characters omitted]...\n{value[-half:]}"

    def _trace(self, event: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with (self.logs_dir / "atlas-trace.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    async def _run_shell(
        self, environment: BaseEnvironment, command: str
    ) -> dict[str, Any]:
        # exec decodes stdout as UTF-8; a binary file (e.g. `cat`ing an image)
        # raises UnicodeDecodeError. Surface it as a tool error so the model can
        # retry a different command instead of crashing the whole trial.
        try:
            result = await environment.exec(command, cwd="/app", timeout_sec=180)
        except UnicodeDecodeError:
            return {
                "return_code": 1,
                "stdout": "",
                "stderr": "command produced non-UTF-8 (binary) output; avoid reading binary files as text",
            }
        return {
            "return_code": result.return_code,
            "stdout": self._truncate(result.stdout or ""),
            "stderr": self._truncate(result.stderr or ""),
        }

    def _submit(self, answer: str) -> None:
        normalized = answer.strip()
        if not normalized:
            raise ValueError("answer must not be empty")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "answer.txt").write_text(
            f"<<FINAL_ANSWER>>\n{normalized}\n<<FINAL_ANSWER>>\n",
            encoding="utf-8",
        )

    @staticmethod
    def _usage_value(value: Any, name: str) -> int:
        result = getattr(value, name, 0) if value is not None else 0
        return int(result or 0)

    def _completion_kwargs(
        self, messages: list[dict[str, Any]], *, tools: bool = True
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._api_model,
            "messages": messages,
            "max_tokens": 12_000,
        }
        if tools:
            kwargs["tools"] = TOOLS
        # Only reasoning-capable models accept reasoning_effort. A provider
        # check is not sufficient: Azure gpt-4o is not Fireworks and still
        # rejects it with "Unrecognized request argument supplied:
        # reasoning_effort".
        _model = self._api_model.lower()
        if _model.startswith(("gpt-5", "o1", "o3", "o4")) or "codex" in _model:
            kwargs["reasoning_effort"] = "high"
        return kwargs

    def _account(
        self, usage: Any, totals: dict[str, int]
    ) -> None:
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
        # turn (provider prompt-caching handles the repeated prefix), which
        # works across every provider, unlike the OpenAI-only Responses API.
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": instruction},
        ]
        totals = {"input": 0, "output": 0, "cached": 0}

        for turn in range(1, MAX_TURNS + 1):
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
            # Record the assistant turn (with any tool calls) in the history.
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
                if (message.content or "").strip():
                    self._submit(message.content)
                    context.metadata = {"turns": turn, "trace": "atlas-trace.jsonl"}
                    break
                raise RuntimeError("model returned neither an answer nor a tool call")

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
                        self._submit(arguments["answer"])
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
                self._trace({"turn": turn, "tool": call.function.name, "result": result})
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
                context.metadata = {"turns": turn, "trace": "atlas-trace.jsonl"}
                break
        else:
            # Investigation budget exhausted. Force one final tool-free answer
            # from what was gathered instead of crashing: a best-effort answer
            # is a real (if low) score, while a raised exception loses the case
            # entirely and records no usage.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have exhausted your investigation budget. Give your "
                        "single best final answer now, based on what you have "
                        "gathered so far."
                    ),
                }
            )
            final = await self._client.chat.completions.create(
                **self._completion_kwargs(messages, tools=False)
            )
            self._account(final.usage, totals)
            answer = (final.choices[0].message.content or "").strip() or (
                "No conclusive answer was determined within the investigation "
                "budget."
            )
            self._submit(answer)
            self._trace({"turn": MAX_TURNS, "forced_final_answer": answer})
            context.metadata = {
                "turns": MAX_TURNS,
                "trace": "atlas-trace.jsonl",
                "forced_final": True,
            }

        context.n_input_tokens = totals["input"]
        context.n_output_tokens = totals["output"]
        context.n_cache_tokens = totals["cached"]
