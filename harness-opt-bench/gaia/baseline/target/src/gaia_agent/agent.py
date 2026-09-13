"""A compact, tool-using GAIA baseline built on the OpenAI Responses API."""

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


def _is_reasoning_model(model: str) -> bool:
    """Whether `model` accepts `reasoning.effort`.

    gpt-4o and other non-reasoning models reject it with HTTP 400, so every
    request that sets it has to ask first -- including the forced-final one.
    """
    name = model.lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4")) or "codex" in name

INSTRUCTIONS = """You are a careful general-purpose research agent solving a GAIA task.

Work until you have a well-supported exact answer. You can search the web, run shell
commands in the task's Linux environment, and inspect attached images. Attached files
named in the task are available under /app/files. Use Python and install focused
packages when they make an analysis more reliable. Cross-check facts and calculations.

The grader reads /app/answer.txt. When ready, call submit_answer with only the exact
answer requested by the task: no explanation, label, markdown, or surrounding prose.
Never modify benchmark tests, verifier files, or expected answers.
"""

TOOLS: list[dict[str, Any]] = [
    {"type": "web_search"},
    {
        "type": "function",
        "name": "run_shell",
        "description": (
            "Run a non-interactive shell command inside the GAIA task environment. "
            "Use /app as the working directory and inspect /app/files for attachments."
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
        "name": "read_image",
        "description": "Load an image from the task environment for visual inspection.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute image path, normally under /app/files",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
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
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class GaiaAgent(BaseAgent):
    """Research agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "gaia-responses-baseline"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("GAIA agent requires a Harbor model")
        self._api_model = self.model_name.removeprefix("openai/")
        # Absorb transient 429s in-client: a within-trial infra failure scores
        # at the failure value for competitive evaluations, so an unretried
        # rate limit costs a candidate a 0.0.
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
        with (self.logs_dir / "gaia-trace.jsonl").open("a", encoding="utf-8") as file:
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

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
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
                "max_output_tokens": 8000,
                "parallel_tool_calls": False,
            }
            if _is_reasoning_model(self._api_model):
                request["reasoning"] = {"effort": "medium"}
            if previous_response_id is not None:
                request["previous_response_id"] = previous_response_id
            response = await self._client.responses.create(**request)
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
                if response.output_text.strip():
                    await self._submit(environment, response.output_text)
                    context.metadata = {"turns": turn, "trace": "gaia-trace.jsonl"}
                    break
                # No custom tool call and no message: the model only reasoned,
                # ran a hosted web_search, or was truncated at max_output_tokens
                # this turn. Carry the chain forward with a nudge instead of
                # crashing; MAX_TURNS + the forced-final below bound the loop.
                previous_response_id = response.id
                next_input = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Continue. When you have the answer, call "
                                    "submit_answer with the exact answer."
                                ),
                            }
                        ],
                    }
                ]
                continue

            next_input = []
            submitted = False
            for call in calls:
                image_url = None
                # The dispatch runs inside the try rather than an else, and
                # OSError is caught alongside the argument errors: read_image
                # downloads a model-supplied path and then stats and reads it,
                # so a hallucinated path fails in the filesystem rather than in
                # json.loads. Uncaught, any of these ends the trial instead of
                # telling the model its tool call did not work.
                try:
                    arguments = json.loads(call.arguments)
                    if call.name == "run_shell":
                        result: dict[str, Any] = await self._run_shell(
                            environment, arguments["command"]
                        )
                    elif call.name == "read_image":
                        result, image_url = await self._read_image(
                            environment, arguments["path"]
                        )
                    elif call.name == "submit_answer":
                        await self._submit(environment, arguments["answer"])
                        result = {"submitted": True}
                        submitted = True
                    else:
                        result = {"error": f"unknown tool: {call.name}"}
                except (
                    json.JSONDecodeError,
                    KeyError,
                    OSError,
                    TypeError,
                    ValueError,
                ) as error:
                    result = {"error": f"invalid arguments: {error}"}
                    image_url = None
                self._trace({"turn": turn, "tool": call.name, "result": result})
                next_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
                if image_url is not None:
                    next_input.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"Image loaded from {arguments['path']}.",
                                },
                                {"type": "input_image", "image_url": image_url},
                            ],
                        }
                    )
            if submitted:
                context.metadata = {"turns": turn, "trace": "gaia-trace.jsonl"}
                break
            previous_response_id = response.id
        else:
            # Turn budget exhausted: force one final tool-free answer from what was
            # gathered rather than crashing, so the case scores best-effort instead
            # of being lost with no answer recorded.
            next_input.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You have used your full research budget. Give your "
                                "single best exact answer now, based on what you have "
                                "gathered."
                            ),
                        }
                    ],
                }
            )
            final_request: dict[str, Any] = {
                "model": self._api_model,
                "instructions": INSTRUCTIONS,
                "input": next_input,
                "max_output_tokens": 8000,
                "previous_response_id": previous_response_id,
            }
            if _is_reasoning_model(self._api_model):
                final_request["reasoning"] = {"effort": "medium"}
            final = await self._client.responses.create(**final_request)
            input_tokens += self._usage_value(final.usage, "input_tokens")
            output_tokens += self._usage_value(final.usage, "output_tokens")
            cached_tokens += self._usage_value(
                getattr(final.usage, "input_tokens_details", None), "cached_tokens"
            )
            answer = (final.output_text or "").strip() or "unknown"
            await self._submit(environment, answer)
            self._trace({"turn": MAX_TURNS, "forced_final_answer": answer})
            context.metadata = {
                "turns": MAX_TURNS,
                "trace": "gaia-trace.jsonl",
                "forced_final": True,
            }

        context.n_input_tokens = input_tokens
        context.n_output_tokens = output_tokens
        context.n_cache_tokens = cached_tokens
