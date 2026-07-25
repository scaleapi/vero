"""Harbor agent for tau3: a host-side MCP conversation loop.

The tau3 task exposes its runtime as a streamable-http MCP server that lives on
the task's internal compose network (`tau3-runtime:8000`), reachable only from
inside the `main` container. Harbor's ``BaseAgent.run`` runs host-side, so this
agent drives the whole conversation from the host: it calls the target model
directly (Harbor already places the inference credentials in this process's
environment), and it reaches the MCP server by shelling ``curl`` into ``main``
via ``environment.exec``. The MCP session is header-keyed (``Mcp-Session-Id``),
so it survives across independent ``curl`` invocations — no persistent
in-container process, no uploaded runner, no credential forwarding.

This whole file is the optimizable surface: the system instructions and the
tool-use loop are what an optimizer edits to improve the agent.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI

MAX_TURNS = 80
MAX_TOOL_OUTPUT_CHARS = 30_000
PROTOCOL_VERSION = "2025-06-18"

INSTRUCTIONS = """You are a careful customer-service agent in a simulated environment.

The task instruction contains the binding domain policy. Follow it exactly. Continue the
conversation by making one MCP tool call at a time. Use send_message_to_user for every
message to the customer, use domain tools only after satisfying policy prerequisites,
and never invent tool results, customer data, or policy. Before any consequential action,
double-check identity, state, allowed parameters, and required confirmation. Use
end_conversation only when the request is resolved or the policy requires ending it.
"""


def _truncate(value: str) -> str:
    if len(value) <= MAX_TOOL_OUTPUT_CHARS:
        return value
    half = MAX_TOOL_OUTPUT_CHARS // 2
    omitted = len(value) - (2 * half)
    return f"{value[:half]}\n...[{omitted} characters omitted]...\n{value[-half:]}"


def _usage_value(value: Any, name: str) -> int:
    result = getattr(value, name, 0) if value is not None else 0
    return int(result or 0)


def _looks_stopped(value: str) -> bool:
    return "###STOP###" in value


class Tau3Agent(BaseAgent):
    """Drive the tau3 MCP conversation host-side, reaching MCP via ``curl``."""

    @staticmethod
    @override
    def name() -> str:
        return "tau3-responses-baseline"

    @override
    def version(self) -> str:
        return "0.2.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("tau3 agent requires a Harbor model")
        self._api_model = self.model_name.removeprefix("openai/")
        self._request_id = 0

    def _server_url(self) -> str:
        urls = [
            str(server.url)
            for server in self.mcp_servers
            if getattr(server, "transport", None) == "streamable-http"
            and getattr(server, "url", None)
        ]
        if len(urls) != 1:
            raise RuntimeError(
                "tau3 agent requires exactly one streamable-http MCP server"
            )
        return urls[0]

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        # The loop reaches MCP by shelling `curl` into `main`; ensure it (and the
        # base64 decoder we pipe payloads through) are present. base64 is coreutils;
        # curl is installed only if the base image lacks it.
        command = (
            "mkdir -p /logs/agent; "
            "if ! command -v curl >/dev/null 2>&1; then "
            "  (apt-get update && apt-get install -y --no-install-recommends curl) "
            "  || apk add --no-cache curl || true; "
            "fi; "
            "command -v curl >/dev/null 2>&1 && command -v base64 >/dev/null 2>&1"
        )
        result = await environment.exec(command, timeout_sec=300)
        if result.return_code != 0:
            raise RuntimeError(
                "tau3 agent requires curl and base64 in the task environment"
            )

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _mcp_raw(
        self,
        environment: BaseEnvironment,
        payload: dict[str, Any],
        *,
        session_id: str | None,
    ) -> str:
        """POST one JSON-RPC message to the MCP server via `curl` inside `main`.

        The payload is base64-encoded so arbitrary tool arguments survive the shell
        untouched. Returns curl's raw stdout (response headers followed by the SSE
        body), which the callers parse for the session id and the `data:` line.
        """
        encoded = base64.b64encode(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        session_header = (
            f"-H 'mcp-session-id: {session_id}' " if session_id is not None else ""
        )
        command = (
            f"printf '%s' '{encoded}' | base64 -d | "
            f"curl -sS -D - -X POST '{self._mcp_url}' "
            f"-H 'Content-Type: application/json' "
            f"-H 'Accept: application/json, text/event-stream' "
            f"{session_header}--data-binary @-"
        )
        result = await environment.exec(command, timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"MCP request failed (rc={result.return_code}): "
                f"{result.stderr or result.stdout}"
            )
        return result.stdout

    @staticmethod
    def _session_id_from(stdout: str) -> str | None:
        for line in stdout.splitlines():
            if line.lower().startswith("mcp-session-id:"):
                return line.split(":", 1)[1].strip()
        return None

    @staticmethod
    def _jsonrpc_from(stdout: str) -> dict[str, Any] | None:
        parsed: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                data = stripped[len("data:"):].strip()
                if not data:
                    continue
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    parsed.append(message)
        for message in parsed:
            if "result" in message or "error" in message:
                return message
        return parsed[-1] if parsed else None

    async def _call_tool(
        self,
        environment: BaseEnvironment,
        session_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        stdout = await self._mcp_raw(
            environment,
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            session_id=session_id,
        )
        response = self._jsonrpc_from(stdout)
        if response is None:
            raise RuntimeError(f"no JSON-RPC response for tool {name!r}")
        if "error" in response:
            return _truncate(json.dumps(response["error"], ensure_ascii=False))
        return _truncate(
            json.dumps(response["result"], ensure_ascii=False, default=str)
        )

    async def _open_session(self, environment: BaseEnvironment) -> tuple[str, list[Any]]:
        init = await self._mcp_raw(
            environment,
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "vero-tau3", "version": self.version()},
                },
            },
            session_id=None,
        )
        session_id = self._session_id_from(init)
        if not session_id:
            raise RuntimeError("MCP initialize did not return a session id")
        await self._mcp_raw(
            environment,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=session_id,
        )
        listed = self._jsonrpc_from(
            await self._mcp_raw(
                environment,
                {"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list"},
                session_id=session_id,
            )
        )
        if listed is None or "result" not in listed:
            raise RuntimeError("MCP tools/list returned no result")
        tools = listed["result"].get("tools", [])
        available = {tool["name"] for tool in tools}
        required = {"start_conversation", "send_message_to_user", "end_conversation"}
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(f"tau3 MCP server is missing tools: {missing}")
        return session_id, tools

    @staticmethod
    def _openai_tools(tools: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description") or f"Call {tool['name']}.",
                "parameters": tool["inputSchema"],
            }
            for tool in tools
            if tool["name"] != "start_conversation"
        ]

    def _trace(self, event: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "tau3-trace.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._mcp_url = self._server_url()
        # When the eval reroutes OPENAI_* to the task's own LLM services
        # (user-sim/grader), the candidate agent's metered gateway is supplied on
        # dedicated vars; otherwise fall back to OPENAI_* from the environment.
        gateway_key = os.environ.get("VERO_AGENT_INFERENCE_API_KEY")
        gateway_url = os.environ.get("VERO_AGENT_INFERENCE_BASE_URL")
        if gateway_key and gateway_url:
            client = AsyncOpenAI(api_key=gateway_key, base_url=gateway_url)
        else:
            client = AsyncOpenAI()  # OPENAI_API_KEY / OPENAI_BASE_URL from the env
        input_tokens = output_tokens = cached_tokens = turns = 0

        session_id, tools = await self._open_session(environment)
        openai_tools = self._openai_tools(tools)

        first_text = await self._call_tool(
            environment, session_id, "start_conversation", {}
        )
        self._trace({"turn": 0, "tool": "start_conversation", "result": first_text})
        next_input: Any = (
            f"{instruction}\n\nThe conversation has been started exactly once. "
            f"The current user message is in this MCP result:\n{first_text}"
        )
        previous_response_id: str | None = None

        for turn in range(1, MAX_TURNS + 1):
            turns = turn
            request: dict[str, Any] = {
                "model": self._api_model,
                "instructions": INSTRUCTIONS,
                "input": next_input,
                "tools": openai_tools,
                "max_output_tokens": 8_000,
                "parallel_tool_calls": False,
            }
            # gpt-4o and other non-reasoning models reject `reasoning.effort`
            # with HTTP 400; only send it to reasoning-capable models.
            _model = self._api_model.lower()
            if _model.startswith(("gpt-5", "o1", "o3", "o4")) or "codex" in _model:
                request["reasoning"] = {"effort": "medium"}
            if previous_response_id is not None:
                request["previous_response_id"] = previous_response_id
            response = await client.responses.create(**request)
            usage = response.usage
            input_tokens += _usage_value(usage, "input_tokens")
            output_tokens += _usage_value(usage, "output_tokens")
            cached_tokens += _usage_value(
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

            if calls:
                call = calls[0]
                try:
                    arguments = json.loads(call.arguments)
                    result_text = await self._call_tool(
                        environment, session_id, call.name, arguments
                    )
                except Exception as error:  # noqa: BLE001 - feed failures back to model
                    result_text = json.dumps(
                        {"error": f"{type(error).__name__}: {error}"}
                    )
                self._trace({"turn": turn, "tool": call.name, "result": result_text})
                next_input = [
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result_text,
                    }
                ]
                previous_response_id = response.id
                if call.name == "end_conversation" or _looks_stopped(result_text):
                    break
                continue

            message = response.output_text.strip()
            if not message:
                raise RuntimeError(
                    "model returned neither a customer message nor a tool call"
                )
            fallback_tool = (
                "end_conversation" if message == "###STOP###" else "send_message_to_user"
            )
            result_text = await self._call_tool(
                environment, session_id, fallback_tool, {"message": message}
            )
            self._trace({"turn": turn, "tool": fallback_tool, "result": result_text})
            if fallback_tool == "end_conversation" or _looks_stopped(result_text):
                break
            next_input = (
                "Your previous text was delivered to the user. "
                f"Their next response is in this MCP result:\n{result_text}"
            )
            previous_response_id = None
        else:
            await self._call_tool(
                environment,
                session_id,
                "end_conversation",
                {"message": "I'm sorry, but I'm unable to complete this request."},
            )

        context.n_input_tokens = input_tokens
        context.n_output_tokens = output_tokens
        context.n_cache_tokens = cached_tokens
        context.metadata = {"turns": turns, "trace": "tau3-trace.jsonl"}
