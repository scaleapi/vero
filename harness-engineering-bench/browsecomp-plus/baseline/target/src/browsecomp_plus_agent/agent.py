"""A compact BrowseComp-Plus agent built on the Chat Completions API."""

from __future__ import annotations

import json
import os
import shlex
import time
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

# Wall clock, not turn count, is this agent's binding constraint. Most of a turn
# is the sequential `search.py` invocations, each of which boots a fresh
# pyserini/Lucene JVM against the BM25 index (~13s apiece), against ~3-9s of
# inference. Trials therefore run out of clock long before MAX_TURNS, so the
# turn-budget fallback below never fires and harbor instead kills the trial
# mid-search with no answer recorded. This gives that same best-effort safety
# net a deadline to watch. Harbor does not tell an agent its own timeout, so the
# budget is read from the environment; keep it in step with whatever
# --agent-timeout-multiplier the run uses.
AGENT_BUDGET_SEC = float(os.environ.get("BROWSECOMP_AGENT_BUDGET_SEC", "900"))
# Enough to prefill the accumulated history and write a short answer (~8s for a
# 120k-token prompt measured against the target endpoint), plus the answer
# upload and margin for one retry.
FINAL_RESERVE_SEC = float(os.environ.get("BROWSECOMP_FINAL_RESERVE_SEC", "120"))
# A malformed, empty, or length-truncated assistant turn is recoverable; nudge
# rather than abort, and only give up after this many consecutive nudges.
MAX_EMPTY_RETRIES = 3
# The required answer is three short labeled lines, so the forced-final call is
# capped well below the per-turn budget to stop the model from spending its last
# seconds on chain-of-thought and getting truncated a second time.
ANSWER_MAX_TOKENS = 3_000

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
        gateway_key = os.environ.get("VERO_AGENT_INFERENCE_API_KEY")
        gateway_url = os.environ.get("VERO_AGENT_INFERENCE_BASE_URL")
        if gateway_key and gateway_url:
            self._client = AsyncOpenAI(
                api_key=gateway_key, base_url=gateway_url, max_retries=8
            )
        else:
            self._client = AsyncOpenAI(max_retries=8)  # OPENAI_* from the env

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
        self,
        environment: BaseEnvironment,
        command: str,
        value: str,
        timeout_sec: float = 120,
    ) -> dict[str, Any]:
        flag = "--query" if command == "search" else "--docid"
        result = await environment.exec(
            f"python3 /opt/browsecomp/search.py {command} {flag} {shlex.quote(value)}",
            cwd="/app",
            timeout_sec=max(15, int(timeout_sec)),
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
        self,
        messages: list[dict[str, Any]],
        *,
        tools: bool = True,
        max_tokens: int = 12_000,
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
        kwargs[_token_limit_key] = max_tokens
        if tools:
            kwargs["tools"] = TOOLS
        # Capability, not provider: see _is_reasoning_model.
        if _is_reasoning_model(self._api_model):
            kwargs["reasoning_effort"] = "medium"
        return kwargs

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: bool = True,
        timeout: float,
        max_retries: int | None = None,
        max_tokens: int = 12_000,
    ) -> Any:
        """One chat completion, validated for shape.

        A gateway can answer 200 with a body the SDK does not turn into a
        completion (observed once as `'str' object has no attribute 'usage'`),
        so anything without usable `.choices` is rejected here rather than
        crashing the trial on an attribute access two lines later.
        """
        options: dict[str, Any] = {"timeout": max(30.0, timeout)}
        if max_retries is not None:
            options["max_retries"] = max_retries
        response = await self._client.with_options(**options).chat.completions.create(
            **self._completion_kwargs(messages, tools=tools, max_tokens=max_tokens)
        )
        choices = getattr(response, "choices", None)
        if not choices or getattr(choices[0], "message", None) is None:
            raise RuntimeError(
                f"malformed completion from provider: {type(response).__name__}"
            )
        return response

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
        deadline = time.monotonic() + AGENT_BUDGET_SEC

        def remaining() -> float:
            return deadline - time.monotonic()

        completed = False
        out_of_time = False
        out_of_answers = False
        empty_turns = 0
        last_turn = 0

        for turn in range(1, MAX_TURNS + 1):
            last_turn = turn
            # Stop starting new work once only the final-answer reserve is left,
            # so the answer is written before harbor's AgentTimeoutError fires.
            if remaining() <= FINAL_RESERVE_SEC:
                out_of_time = True
                self._trace({"turn": turn, "stopping": "deadline", "left": remaining()})
                break
            call_started = time.monotonic()
            response = await self._complete(
                messages, timeout=remaining() - FINAL_RESERVE_SEC / 2
            )
            llm_sec = time.monotonic() - call_started
            self._account(getattr(response, "usage", None), totals)
            message = response.choices[0].message
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            calls = message.tool_calls or []
            self._trace(
                {
                    "turn": turn,
                    "content": message.content,
                    "finish_reason": finish_reason,
                    "llm_sec": round(llm_sec, 2),
                    "left_sec": round(remaining(), 1),
                    "tool_calls": [
                        {"name": c.function.name, "arguments": c.function.arguments}
                        for c in calls
                    ],
                }
            )
            # Record the assistant turn (with any tool calls) in the history.
            # An assistant message carrying neither is rejected by strict
            # OpenAI-compatible servers, so it is dropped rather than appended.
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
            if len(assistant) > 1:
                messages.append(assistant)

            if not calls:
                # `length` means the model was cut off mid-thought, so its
                # content is truncated reasoning, not an answer. This target
                # writes its chain-of-thought into ordinary `content` (it has no
                # separate reasoning_content channel) and can run past the token
                # cap, and the truncated ramble was previously submitted verbatim
                # as the final answer, which the judge always scores 0. Ask for
                # the answer instead of submitting the thinking.
                truncated = finish_reason == "length"
                if (message.content or "").strip() and not truncated:
                    await self._submit(environment, message.content)
                    completed = True
                    context.metadata = {
                        "turns": turn,
                        "trace": "browsecomp-plus-trace.jsonl",
                    }
                    break
                empty_turns += 1
                self._trace(
                    {"turn": turn, "recover": empty_turns, "truncated": truncated}
                )
                if empty_turns >= MAX_EMPTY_RETRIES:
                    out_of_answers = True
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You were cut off mid-thought. Stop reasoning and "
                            "reply with only the three required labeled lines "
                            "(Explanation, Exact Answer, Confidence), keeping the "
                            "explanation to a few sentences."
                            if truncated
                            else "Your last message was empty. Either call a tool "
                            "or give your final response in the required format."
                        ),
                    }
                )
                continue
            empty_turns = 0

            submitted = False
            for call in calls:
                # Every tool_call must still get a tool message even once the
                # deadline hits, or the forced-final request below is rejected
                # for an unanswered tool call. So stub, never skip.
                if out_of_time or remaining() <= FINAL_RESERVE_SEC:
                    if not out_of_time:
                        out_of_time = True
                        self._trace({"turn": turn, "stopping": "deadline-midturn"})
                    result = {"error": "search budget exhausted"}
                else:
                    try:
                        arguments = json.loads(call.function.arguments)
                        budget = remaining() - FINAL_RESERVE_SEC
                        if call.function.name == "search":
                            result = await self._index_command(
                                environment, "search", arguments["query"], budget
                            )
                        elif call.function.name == "get_document":
                            result = await self._index_command(
                                environment, "get-document", arguments["docid"], budget
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
                # Traced on both paths. A stub is precisely what a post-hoc audit
                # needs to see (which searches were abandoned to the clock), and
                # tracing only the executed branch left it invisible.
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
                completed = True
                context.metadata = {
                    "turns": turn,
                    "trace": "browsecomp-plus-trace.jsonl",
                }
                break
            if out_of_time:
                break

        if not completed:
            # Budget exhausted (wall clock, turns, or a model that kept coming
            # back empty): force one final tool-free response from what was
            # retrieved rather than crashing, so the case scores best-effort
            # instead of being lost with no answer recorded.
            stub = (
                "Explanation: no answer could be determined within the search "
                "budget.\nExact Answer: unknown\nConfidence: 0%"
            )
            # Three distinct exits land here and a post-hoc scorer has to tell
            # them apart: the wall clock, repeated truncated/empty turns, and
            # plain MAX_TURNS exhaustion.
            reason = (
                "deadline"
                if out_of_time
                else "empty_retries"
                if out_of_answers
                else "turns"
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have used your full search budget. Do not reason "
                        "further. Reply with only the three required labeled "
                        "lines (Explanation, Exact Answer, Confidence), keeping "
                        "the explanation to a few sentences, giving your single "
                        "best answer from the documents you have retrieved."
                    ),
                }
            )
            # This call is the last thing standing between a completed trial and
            # a lost one, so a failure here degrades to the stub answer instead
            # of propagating.
            answer = stub
            try:
                # Bounded retries: the SDK default of 8 could outlast the
                # reserve on its own and hand the trial back to the killer.
                # A tight token cap keeps the model from spending the whole
                # reserve on chain-of-thought and getting truncated again.
                final = await self._complete(
                    messages,
                    tools=False,
                    timeout=max(30.0, remaining() - 20),
                    max_retries=1,
                    max_tokens=ANSWER_MAX_TOKENS,
                )
                self._account(getattr(final, "usage", None), totals)
                answer = (final.choices[0].message.content or "").strip() or stub
            except Exception as error:  # noqa: BLE001 - last-resort safety net
                self._trace({"forced_final_error": f"{type(error).__name__}: {error}"})
            await self._submit(environment, answer)
            self._trace(
                {
                    "turn": last_turn,
                    "forced_final_answer": answer,
                    "reason": reason,
                }
            )
            context.metadata = {
                "turns": last_turn,
                "trace": "browsecomp-plus-trace.jsonl",
                "forced_final": True,
                # Lets a post-hoc conservative score treat exactly the trials
                # that the pre-fix agent would have lost as zeros.
                "forced_final_reason": reason,
            }

        context.n_input_tokens = totals["input"]
        context.n_output_tokens = totals["output"]
        context.n_cache_tokens = totals["cached"]
