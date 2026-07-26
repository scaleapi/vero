"""A compact, tool-using OfficeQA baseline built on the Chat Completions API."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI

MAX_TURNS = 24
MAX_TOOL_OUTPUT_CHARS = 20_000
MAX_IMAGE_BYTES = 20 * 1024 * 1024

INSTRUCTIONS = """You are a careful grounded-reasoning agent answering questions from a
document corpus (OfficeQA).

Work until you have a well-supported exact answer grounded in the corpus. Run shell
commands in the task's Linux environment to search and read the documents. The corpus
location is given in the task instructions (typically a directory such as /app/corpus/
with an index file); grep/search it to find the relevant documents, then read them.
Use Python and install focused packages when they make an analysis more reliable.
Cross-check facts and calculations; scoring uses a numeric tolerance, so report precise
numbers in the requested units.

The grader reads /app/answer.txt. When ready, call submit_answer with only the exact
answer requested by the task: no explanation, label, markdown, or surrounding prose.
Never modify benchmark tests, verifier files, or expected answers.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a non-interactive shell command inside the OfficeQA task environment. "
                "Use /app as the working directory and inspect the corpus directory named in the task instructions."
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
            "name": "read_image",
            "description": "Load an image from the task environment for visual inspection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute image path, normally under the task corpus dir",
                    }
                },
                "required": ["path"],
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


class OfficeQaAgent(BaseAgent):
    """Research agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "officeqa-responses-baseline"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("OfficeQA agent requires a Harbor model")
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
        with (self.logs_dir / "officeqa-trace.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    async def _run_shell(
        self, environment: BaseEnvironment, command: str
    ) -> dict[str, Any]:
        result = await environment.exec(command, cwd="/app", timeout_sec=120)
        return {
            "return_code": result.return_code,
            "stdout": self._truncate(result.stdout or ""),
            "stderr": self._truncate(result.stderr or ""),
        }

    async def _read_image(
        self, environment: BaseEnvironment, remote_path: str
    ) -> tuple[dict[str, Any], str | None]:
        if not remote_path.startswith("/app/"):
            return {"error": "image path must be under /app"}, None
        local_path = self.logs_dir / "images" / Path(remote_path).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        await environment.download_file(remote_path, local_path)
        if local_path.stat().st_size > MAX_IMAGE_BYTES:
            return {"error": "image exceeds the 20 MiB tool limit"}, None
        media_type = mimetypes.guess_type(local_path.name)[0]
        if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            return {"error": f"unsupported image media type: {media_type}"}, None
        encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
        return {
            "loaded": remote_path,
            "media_type": media_type,
        }, f"data:{media_type};base64,{encoded}"

    async def _submit(self, environment: BaseEnvironment, answer: str) -> None:
        normalized = answer.strip()
        if not normalized:
            raise ValueError("answer must not be empty")
        local_path = self.logs_dir / "answer.txt"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(normalized + "\n", encoding="utf-8")
        await environment.upload_file(local_path, "/app/answer.txt")

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
            "max_tokens": 8000,
        }
        if tools:
            kwargs["tools"] = TOOLS
        # OpenAI reasoning models accept reasoning_effort and parallel_tool_calls;
        # other providers (e.g. Fireworks-served open models) reject them.
        if "fireworks" not in self._api_model:
            kwargs["reasoning_effort"] = "medium"
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
                    await self._submit(environment, message.content)
                    context.metadata = {"turns": turn, "trace": "officeqa-trace.jsonl"}
                    break
                raise RuntimeError("model returned neither an answer nor a tool call")

            submitted = False
            for call in calls:
                image_url = None
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
                    elif call.function.name == "read_image":
                        result, image_url = await self._read_image(
                            environment, arguments["path"]
                        )
                    elif call.function.name == "submit_answer":
                        await self._submit(environment, arguments["answer"])
                        result = {"submitted": True}
                        submitted = True
                    else:
                        result = {"error": f"unknown tool: {call.function.name}"}
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    result = {"error": f"invalid arguments: {error}"}
                    image_url = None
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
                if image_url is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Image loaded from {arguments['path']}.",
                                },
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    )
            if submitted:
                context.metadata = {"turns": turn, "trace": "officeqa-trace.jsonl"}
                break
        else:
            # Turn budget exhausted: force one final tool-free answer from what
            # was gathered rather than crashing, so the case scores best-effort
            # instead of being lost with no answer recorded.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have used your full research budget. Give your single "
                        "best final answer now, based on what you have gathered."
                    ),
                }
            )
            final = await self._client.chat.completions.create(
                **self._completion_kwargs(messages, tools=False)
            )
            self._account(final.usage, totals)
            answer = (final.choices[0].message.content or "").strip() or (
                "No answer could be determined within the research budget."
            )
            await self._submit(environment, answer)
            self._trace({"turn": MAX_TURNS, "forced_final_answer": answer})
            context.metadata = {
                "turns": MAX_TURNS,
                "trace": "officeqa-trace.jsonl",
                "forced_final": True,
            }

        context.n_input_tokens = totals["input"]
        context.n_output_tokens = totals["output"]
        context.n_cache_tokens = totals["cached"]
