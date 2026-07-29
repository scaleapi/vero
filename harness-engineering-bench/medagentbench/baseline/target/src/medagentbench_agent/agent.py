"""A compact MedAgentBench baseline built on the Chat Completions API.

The task serves a FHIR electronic health record on ``localhost:8080`` *inside the
task container*, while ``BaseAgent.run`` executes host-side. Every tool here
therefore shells the container's own CLI through ``environment.exec``, the way
tau3, browsecomp-plus and officeqa reach their in-container services.

Routing through ``medagentbench_cli.py`` rather than curl is load-bearing, not a
convenience:

* ``post`` records the POST and its acceptance message in the history file, and
  ``finish`` folds that history into ``/workspace/answer.json``. The verifier
  grades action tasks from that history, so a raw curl POST executes but does not
  count.
* The CLI retries connection errors with backoff for up to 120 attempts, which
  absorbs the H2 database's boot time. A first request issued directly would race
  the server coming up and fail the case for a reason unrelated to the harness.
"""

from __future__ import annotations

import json
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


MAX_TURNS = 24
MAX_TOOL_OUTPUT_CHARS = 20_000
CLI = "/usr/local/bin/medagentbench_cli.py"
# The CLI's own startup retries can span a couple of minutes on a cold container.
CLI_TIMEOUT_SEC = 300

INSTRUCTIONS = """You are a careful clinical agent working against a FHIR electronic
health record (MedAgentBench).

Query the record with fhir_get before you conclude anything. Some tasks also require
an action: when the task says to order a lab, place a referral, or record a value, you
must issue the matching POST with fhir_post. An answer that describes an action without
performing it scores zero.

Respect the task's stated reference timestamp rather than today's date, and use the
codes the task gives you verbatim. When a task says to return a specific sentinel (for
example [-1]) if no measurement exists, return exactly that.

Finish by calling finish with the answer as a JSON list. That records your POST history
alongside the result, which the grader reads. Never modify benchmark tests, verifier
files, or expected answers.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fhir_get",
            "description": (
                "GET a FHIR resource. Pass the full URL, for example "
                "http://localhost:8080/fhir/Observation?patient=S6227720&code=A1C"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full FHIR GET URL"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fhir_post",
            "description": (
                "POST a FHIR resource to perform a required action. The request and "
                "its acceptance are recorded in the history the grader reads."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full FHIR POST URL, e.g. http://localhost:8080/fhir/ServiceRequest",
                    },
                    "payload": {
                        "type": "object",
                        "description": "FHIR resource body as a JSON object",
                    },
                },
                "required": ["url", "payload"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Write the final answer and end the task. The result must be a JSON "
                "list; use the sentinel the task specifies when nothing applies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "array",
                        "description": "Final answer as a JSON list",
                        "items": {},
                    }
                },
                "required": ["result"],
            },
        },
    },
]


class MedAgentBenchAgent(BaseAgent):
    """Clinical FHIR agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "medagentbench-chat-baseline"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("MedAgentBench agent requires a Harbor model")
        self._api_model = self.model_name.removeprefix("openai/")
        self._client = AsyncOpenAI(max_retries=8)

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(f"test -x {CLI}", timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(f"{CLI} is missing from the task environment")

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= MAX_TOOL_OUTPUT_CHARS:
            return value
        half = MAX_TOOL_OUTPUT_CHARS // 2
        omitted = len(value) - (2 * half)
        return f"{value[:half]}\n...[{omitted} characters omitted]...\n{value[-half:]}"

    def _trace(self, event: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "medagentbench-trace.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    async def _cli(
        self, environment: BaseEnvironment, *arguments: str
    ) -> dict[str, Any]:
        command = " ".join([CLI, *(shlex.quote(argument) for argument in arguments)])
        result = await environment.exec(command, timeout_sec=CLI_TIMEOUT_SEC)
        if result.return_code != 0:
            return {
                "return_code": result.return_code,
                "error": self._truncate(result.stderr or "CLI call failed"),
            }
        return {"return_code": 0, "output": self._truncate(result.stdout or "")}

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
                    # Prose is not an answer here: the grader reads
                    # /workspace/answer.json, which only `finish` writes. Nudge
                    # rather than raise, so the case can still score.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That turn recorded nothing. The grader only reads "
                                "the file written by finish. Either query with "
                                "fhir_get, act with fhir_post, or call finish."
                            ),
                        }
                    )
                    self._trace({"turn": turn, "empty_turn": True})
                    continue

                finished = False
                for call in calls:
                    # The dispatch runs inside the try because the argument
                    # lookups are the likelier failure: the schemas are not
                    # strict, so the model can return valid JSON missing a
                    # required key. Feed that back instead of ending the trial.
                    try:
                        arguments = json.loads(call.function.arguments)
                        if call.function.name == "fhir_get":
                            result = await self._cli(
                                environment, "get", str(arguments["url"])
                            )
                        elif call.function.name == "fhir_post":
                            result = await self._cli(
                                environment,
                                "post",
                                str(arguments["url"]),
                                json.dumps(arguments["payload"]),
                            )
                        elif call.function.name == "finish":
                            payload = arguments["result"]
                            if not isinstance(payload, list):
                                raise ValueError("finish result must be a JSON list")
                            result = await self._cli(
                                environment, "finish", json.dumps(payload)
                            )
                            finished = result.get("return_code") == 0
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
                    if finished:
                        break
                if finished:
                    context.metadata = {
                        "turns": turn,
                        "trace": "medagentbench-trace.jsonl",
                    }
                    break
            else:
                # Turn budget exhausted. Ask for the answer, then write it through
                # the CLI so the case scores best-effort rather than leaving
                # /workspace/answer.json absent.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have used your full budget. Reply with only the "
                            "final answer as a JSON list, no prose."
                        ),
                    }
                )
                final = await self._client.chat.completions.create(
                    **self._completion_kwargs(messages, tools=False)
                )
                self._account(final.usage, totals)
                text = (final.choices[0].message.content or "").strip()
                try:
                    parsed = json.loads(text)
                    payload = parsed if isinstance(parsed, list) else [parsed]
                except json.JSONDecodeError:
                    payload = [text] if text else [-1]
                result = await self._cli(environment, "finish", json.dumps(payload))
                self._trace({"turn": MAX_TURNS, "forced_final": payload, "result": result})
                context.metadata = {
                    "turns": MAX_TURNS,
                    "trace": "medagentbench-trace.jsonl",
                    "forced_final": True,
                }
        finally:
            # In `finally` so a raised trial still reports its usage: assigning
            # this on the last line of run() loses the record on every crash,
            # which biases per-trial cost low on the expensive cases.
            context.n_input_tokens = totals["input"]
            context.n_output_tokens = totals["output"]
            context.n_cache_tokens = totals["cached"]
            if context.metadata is None:
                context.metadata = {
                    "turns": turns,
                    "trace": "medagentbench-trace.jsonl",
                }
