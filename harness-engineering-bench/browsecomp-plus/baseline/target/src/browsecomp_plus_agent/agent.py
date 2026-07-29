"""A compact BrowseComp-Plus agent built on the Chat Completions API."""

from __future__ import annotations

import json
import os
import shlex
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

MAX_TURNS = 32
MAX_TOOL_OUTPUT_CHARS = 40_000

INSTRUCTIONS = """You are a persistent deep-research agent working only against the
fixed BrowseComp-Plus corpus. Find the precise answer by issuing focused searches,
opening promising documents, reformulating queries, and cross-checking evidence.
Do not use the live web.

Your final response must contain exactly these labeled sections:
Explanation: your concise reasoning with supporting document ids cited as [docid]
Exact Answer: the succinct answer requested by the question
Confidence: a calibrated confidence from 0% to 100%

Call submit_response with the complete formatted response. Never inspect or modify
benchmark tests, verifier files, or expected answers.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the fixed BrowseComp-Plus BM25 index and return the top five "
                "documents as ids, scores, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Retrieve the full text of one document by its exact id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docid": {"type": "string", "description": "Document id"}
                },
                "required": ["docid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_response",
            "description": (
                "Submit the complete explanation, exact answer, and confidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Complete final response in the required format",
                    }
                },
                "required": ["response"],
            },
        },
    },
]


class BrowseCompPlusAgent(BaseAgent):
    """Deep-research agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "browsecomp-plus-responses-baseline"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("BrowseComp-Plus agent requires a Harbor model")
        self._api_model = self.model_name.removeprefix("openai/")
        # This benchmark sets task_services_use_upstream, which points OPENAI_* at
        # the real upstream so the in-container grader can reach it. The candidate
        # agent's metered, allow-listed gateway arrives on dedicated vars instead,
        # and reading them is what keeps target inference on the gateway -- without
        # this, the agent silently ran on the raw upstream credential: unmetered,
        # and with the pinned target model unenforced.
        # Fail closed. Falling back to OPENAI_* here would silently restore the
        # original bug -- an unmetered 7.5-hour run whose only symptom is an empty
        # evaluation scope in the request log, which is how it went unnoticed the
        # first time. Missing credentials are a wiring fault, so say so at once.
        gateway_key = os.environ.get("VERO_AGENT_INFERENCE_API_KEY")
        gateway_url = os.environ.get("VERO_AGENT_INFERENCE_BASE_URL")
        if not gateway_key or not gateway_url:
            raise RuntimeError(
                "BrowseComp-Plus target inference requires "
                "VERO_AGENT_INFERENCE_API_KEY and VERO_AGENT_INFERENCE_BASE_URL; "
                "refusing to fall back to OPENAI_*, which points at the "
                "unmetered upstream under task_services_use_upstream"
            )
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
        if len(value) <= MAX_TOOL_OUTPUT_CHARS:
            return value
        half = MAX_TOOL_OUTPUT_CHARS // 2
        omitted = len(value) - (2 * half)
        return f"{value[:half]}\n...[{omitted} characters omitted]...\n{value[-half:]}"

    def _trace(self, event: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with (self.logs_dir / "browsecomp-plus-trace.jsonl").open(
            "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    async def _index_command(
        self, environment: BaseEnvironment, command: str, value: str
    ) -> dict[str, Any]:
        flag = "--query" if command == "search" else "--docid"
        result = await environment.exec(
            f"python3 /opt/browsecomp/search.py {command} {flag} {shlex.quote(value)}",
            cwd="/app",
            timeout_sec=120,
        )
        if result.return_code != 0:
            return {
                "error": self._truncate(result.stderr or "retrieval command failed"),
                "return_code": result.return_code,
            }
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"error": "retriever returned invalid JSON", "output": result.stdout}

    async def _submit(self, environment: BaseEnvironment, response: str) -> None:
        normalized = response.strip()
        if not normalized:
            raise ValueError("response must not be empty")
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
        # Reasoning models replaced max_tokens with max_completion_tokens and
        # reject the old name outright ("Unsupported parameter: 'max_tokens'
        # is not supported with this model"). Same capability test as the
        # reasoning_effort gate below, so the two stay consistent.
        _token_limit_key = (
            "max_completion_tokens"
            if _is_reasoning_model(self._api_model)
            else "max_tokens"
        )
        kwargs: dict[str, Any] = {
            "model": self._api_model,
            "messages": messages,
        }
        kwargs[_token_limit_key] = 12_000
        if tools:
            kwargs["tools"] = TOOLS
        # Capability, not provider: see _is_reasoning_model.
        if _is_reasoning_model(self._api_model):
            kwargs["reasoning_effort"] = "medium"
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
                    context.metadata = {
                        "turns": turn,
                        "trace": "browsecomp-plus-trace.jsonl",
                    }
                    break
                raise RuntimeError("model returned neither a response nor a tool call")

            submitted = False
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    if call.function.name == "search":
                        result = await self._index_command(
                            environment, "search", arguments["query"]
                        )
                    elif call.function.name == "get_document":
                        result = await self._index_command(
                            environment, "get-document", arguments["docid"]
                        )
                    elif call.function.name == "submit_response":
                        await self._submit(environment, arguments["response"])
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
                    result = {"error": str(error)}
                self._trace(
                    {"turn": turn, "tool": call.function.name, "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": self._truncate(
                            json.dumps(result, ensure_ascii=False, default=str)
                        ),
                    }
                )
                if submitted:
                    break
            if submitted:
                context.metadata = {
                    "turns": turn,
                    "trace": "browsecomp-plus-trace.jsonl",
                }
                break
        else:
            # Turn budget exhausted: force one final tool-free response from what
            # was retrieved rather than crashing, so the case scores best-effort
            # instead of being lost with no answer recorded.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have used your full search budget. Give your single "
                        "best final response now, in the required format, based on "
                        "the documents you have retrieved."
                    ),
                }
            )
            final = await self._client.chat.completions.create(
                **self._completion_kwargs(messages, tools=False)
            )
            self._account(final.usage, totals)
            answer = (final.choices[0].message.content or "").strip() or (
                "Explanation: no answer could be determined within the search "
                "budget.\nExact Answer: unknown\nConfidence: 0%"
            )
            await self._submit(environment, answer)
            self._trace({"turn": MAX_TURNS, "forced_final_answer": answer})
            context.metadata = {
                "turns": MAX_TURNS,
                "trace": "browsecomp-plus-trace.jsonl",
                "forced_final": True,
            }

        context.n_input_tokens = totals["input"]
        context.n_output_tokens = totals["output"]
        context.n_cache_tokens = totals["cached"]
