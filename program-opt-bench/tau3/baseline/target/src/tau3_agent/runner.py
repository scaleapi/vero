"""Editable tau3 MCP conversation loop executed inside the Harbor task."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import AsyncOpenAI

MAX_TURNS = 80
MAX_TOOL_OUTPUT_CHARS = 30_000
TRACE_PATH = Path("/logs/agent/tau3-trace.jsonl")
CONTEXT_PATH = Path("/logs/agent/tau3-context.json")

INSTRUCTIONS = """You are a careful customer-service agent in a simulated environment.

The task instruction contains the binding domain policy. Follow it exactly. Continue the
conversation by making one MCP tool call at a time. Use send_message_to_user for every
message to the customer, use domain tools only after satisfying policy prerequisites,
and never invent tool results, customer data, or policy. Before any consequential action,
double-check identity, state, allowed parameters, and required confirmation. Use
end_conversation only when the request is resolved or the policy requires ending it.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def _openai_tools(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description or f"Call {tool.name}.",
            "parameters": tool.inputSchema,
        }
        for tool in tools
        if tool.name != "start_conversation"
    ]


def _usage_value(value: Any, name: str) -> int:
    result = getattr(value, name, 0) if value is not None else 0
    return int(result or 0)


def _truncate(value: str) -> str:
    if len(value) <= MAX_TOOL_OUTPUT_CHARS:
        return value
    half = MAX_TOOL_OUTPUT_CHARS // 2
    omitted = len(value) - (2 * half)
    return f"{value[:half]}\n...[{omitted} characters omitted]...\n{value[-half:]}"


def _tool_result_text(result: Any) -> str:
    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json", exclude_none=True)
    else:
        payload = result
    return _truncate(json.dumps(payload, ensure_ascii=False, default=str))


def _trace(event: dict[str, Any]) -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _looks_stopped(value: str) -> bool:
    return "###STOP###" in value


async def _run(instruction: str, mcp_url: str, model: str) -> None:
    client = AsyncOpenAI()
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    turns = 0

    async with streamable_http_client(mcp_url) as streams:
        read_stream, write_stream, _ = streams
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools_by_name = {tool.name: tool for tool in listed.tools}
            required = {
                "start_conversation",
                "send_message_to_user",
                "end_conversation",
            }
            missing = sorted(required - set(tools_by_name))
            if missing:
                raise RuntimeError(f"tau3 MCP server is missing tools: {missing}")

            first = await session.call_tool("start_conversation", {})
            first_text = _tool_result_text(first)
            _trace({"turn": 0, "tool": "start_conversation", "result": first_text})
            next_input: Any = (
                f"{instruction}\n\nThe conversation has been started exactly once. "
                f"The current user message is in this MCP result:\n{first_text}"
            )
            previous_response_id: str | None = None
            openai_tools = _openai_tools(listed.tools)

            for turn in range(1, MAX_TURNS + 1):
                turns = turn
                request: dict[str, Any] = {
                    "model": model,
                    "instructions": INSTRUCTIONS,
                    "input": next_input,
                    "tools": openai_tools,
                    "reasoning": {"effort": "medium"},
                    "max_output_tokens": 8_000,
                    "parallel_tool_calls": False,
                }
                if previous_response_id is not None:
                    request["previous_response_id"] = previous_response_id
                response = await client.responses.create(**request)
                usage = response.usage
                input_tokens += _usage_value(usage, "input_tokens")
                output_tokens += _usage_value(usage, "output_tokens")
                cached_tokens += _usage_value(
                    getattr(usage, "input_tokens_details", None), "cached_tokens"
                )

                calls = [
                    item for item in response.output if item.type == "function_call"
                ]
                _trace(
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
                        result = await session.call_tool(call.name, arguments)
                        result_text = _tool_result_text(result)
                    except Exception as error:
                        result_text = json.dumps(
                            {"error": f"{type(error).__name__}: {error}"}
                        )
                    _trace({"turn": turn, "tool": call.name, "result": result_text})
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
                    "end_conversation"
                    if message == "###STOP###"
                    else "send_message_to_user"
                )
                result = await session.call_tool(fallback_tool, {"message": message})
                result_text = _tool_result_text(result)
                _trace({"turn": turn, "tool": fallback_tool, "result": result_text})
                if fallback_tool == "end_conversation" or _looks_stopped(result_text):
                    break
                next_input = (
                    "Your previous text was delivered to the user. "
                    f"Their next response is in this MCP result:\n{result_text}"
                )
                previous_response_id = None
            else:
                await session.call_tool(
                    "end_conversation",
                    {"message": "I’m sorry, but I’m unable to complete this request."},
                )

    CONTEXT_PATH.write_text(
        json.dumps(
            {
                "turns": turns,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    instruction = args.instruction.read_text(encoding="utf-8")
    asyncio.run(_run(instruction, args.mcp_url, args.model))


if __name__ == "__main__":
    main()
