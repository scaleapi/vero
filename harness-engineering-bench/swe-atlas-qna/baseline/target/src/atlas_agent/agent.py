"""A compact codebase investigation agent built on the Responses API."""

from __future__ import annotations

import json
import os
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI

MAX_TURNS = 30
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
        "name": "run_shell",
        "description": (
            "Run a non-interactive command in the repository at /app. Use this for "
            "read-only exploration and temporary experiments; never edit repository files."
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
        "name": "submit_answer",
        "description": "Write the complete evidence-backed answer and finish the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Complete final answer"}
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
        "strict": True,
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
        result = await environment.exec(command, cwd="/app", timeout_sec=180)
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
                "max_output_tokens": 12_000,
                "parallel_tool_calls": False,
            }
            # gpt-4o and other non-reasoning models reject `reasoning.effort`
            # with HTTP 400; only send it to reasoning-capable models.
            _model = self._api_model.lower()
            if _model.startswith(("gpt-5", "o1", "o3", "o4")) or "codex" in _model:
                request["reasoning"] = {"effort": "high"}
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
                    self._submit(response.output_text)
                    context.metadata = {"turns": turn, "trace": "atlas-trace.jsonl"}
                    break
                raise RuntimeError("model returned neither an answer nor a tool call")

            next_input = []
            submitted = False
            for call in calls:
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError as error:
                    result: dict[str, Any] = {"error": f"invalid arguments: {error}"}
                else:
                    if call.name == "run_shell":
                        result = await self._run_shell(
                            environment, arguments["command"]
                        )
                    elif call.name == "submit_answer":
                        self._submit(arguments["answer"])
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
            previous_response_id = response.id
            if submitted:
                context.metadata = {"turns": turn, "trace": "atlas-trace.jsonl"}
                break
        else:
            raise RuntimeError(f"agent exceeded {MAX_TURNS} turns")

        context.n_input_tokens = input_tokens
        context.n_output_tokens = output_tokens
        context.n_cache_tokens = cached_tokens
