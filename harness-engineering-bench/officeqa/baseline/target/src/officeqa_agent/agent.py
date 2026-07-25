"""A compact, tool-using OfficeQA baseline built on the OpenAI Responses API."""

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
    {"type": "web_search"},
    {
        "type": "function",
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
                    "description": "Absolute image path, normally under the task corpus dir",
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
        self._client = AsyncOpenAI()

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
            # gpt-4o and other non-reasoning models reject `reasoning.effort`
            # with HTTP 400; only send it to reasoning-capable models.
            _model = self._api_model.lower()
            if _model.startswith(("gpt-5", "o1", "o3", "o4")) or "codex" in _model:
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
                    context.metadata = {"turns": turn, "trace": "officeqa-trace.jsonl"}
                    break
                raise RuntimeError("model returned neither an answer nor a tool call")

            next_input = []
            submitted = False
            for call in calls:
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError as error:
                    result: dict[str, Any] = {"error": f"invalid arguments: {error}"}
                    image_url = None
                else:
                    image_url = None
                    if call.name == "run_shell":
                        result = await self._run_shell(
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
                context.metadata = {"turns": turn, "trace": "officeqa-trace.jsonl"}
                break
            previous_response_id = response.id
        else:
            raise RuntimeError(f"OfficeQA agent exceeded {MAX_TURNS} turns")

        context.n_input_tokens = input_tokens
        context.n_output_tokens = output_tokens
        context.n_cache_tokens = cached_tokens
