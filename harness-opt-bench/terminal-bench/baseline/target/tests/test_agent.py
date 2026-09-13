"""Behaviour a candidate must keep, so a change cannot break it silently."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from terminal_bench_agent.agent import (
    MAX_OUTPUT_CHARS,
    TOOLS,
    TerminalBenchAgent,
)


@dataclass
class _Result:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class _Environment:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, **_: Any) -> _Result:
        self.commands.append(command)
        return _Result(stdout="ok")


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _Call:
    function: _Function
    id: str = "call_1"


@dataclass
class _Message:
    content: str = ""
    tool_calls: list[_Call] = field(default_factory=list)


def _shell(command: str, call_id: str = "call_1") -> _Call:
    return _Call(_Function("run_shell", json.dumps({"command": command})), call_id)


def _finish(call_id: str = "call_done") -> _Call:
    return _Call(_Function("finish", json.dumps({"summary": "done"})), call_id)


class TestToolSchema:
    def test_exposes_run_shell_and_finish(self) -> None:
        assert {t["function"]["name"] for t in TOOLS} == {"run_shell", "finish"}

    def test_run_shell_requires_a_command(self) -> None:
        shell = next(t for t in TOOLS if t["function"]["name"] == "run_shell")
        assert shell["function"]["parameters"]["required"] == ["command"]


class TestArgumentParsing:
    def test_well_formed(self) -> None:
        assert TerminalBenchAgent._command_of(_shell("ls -la")) == "ls -la"

    def test_malformed_json_yields_none(self) -> None:
        assert (
            TerminalBenchAgent._command_of(_Call(_Function("run_shell", "{x"))) is None
        )

    def test_missing_or_blank_command_yields_none(self) -> None:
        assert (
            TerminalBenchAgent._command_of(_Call(_Function("run_shell", "{}"))) is None
        )
        assert (
            TerminalBenchAgent._command_of(
                _Call(_Function("run_shell", json.dumps({"command": "  "})))
            )
            is None
        )


class TestOutputTruncation:
    def test_short_output_untouched(self) -> None:
        assert TerminalBenchAgent._truncate("hello") == "hello"

    def test_long_output_keeps_both_ends(self) -> None:
        value = ("a" * MAX_OUTPUT_CHARS) + "MIDDLE" + ("z" * MAX_OUTPUT_CHARS)
        result = TerminalBenchAgent._truncate(value)
        assert result.startswith("a") and result.endswith("z")
        assert "MIDDLE" not in result


class TestGatewayCredentials:
    """The target must find the gateway on either pair of variables."""

    @staticmethod
    def _construct(tmp_path: Any) -> TerminalBenchAgent:
        return TerminalBenchAgent(logs_dir=tmp_path, model_name="test-model")

    def test_missing_credentials_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        for name in (
            "VERO_AGENT_INFERENCE_API_KEY",
            "VERO_AGENT_INFERENCE_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
        ):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(RuntimeError, match="scoped API key"):
            self._construct(tmp_path)

    def test_openai_variables_are_used_when_vero_ones_are_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.delenv("VERO_AGENT_INFERENCE_API_KEY", raising=False)
        monkeypatch.delenv("VERO_AGENT_INFERENCE_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "scoped-token")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example/v1")
        agent = self._construct(tmp_path)
        assert "gw.example" in str(agent._client.base_url)

    def test_dedicated_variables_win_over_openai_ones(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """OPENAI_* may be the unmetered upstream; precedence keeps us off it."""
        monkeypatch.setenv("VERO_AGENT_INFERENCE_API_KEY", "scoped-token")
        monkeypatch.setenv("VERO_AGENT_INFERENCE_BASE_URL", "https://gw.example/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "upstream-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://upstream.example/v1")
        agent = self._construct(tmp_path)
        assert "gw.example" in str(agent._client.base_url)
        assert agent._client.api_key == "scoped-token"


class TestRunLoop:
    @staticmethod
    def _prepare(
        monkeypatch: pytest.MonkeyPatch, replies: list[_Message], tmp_path: Any
    ) -> tuple[TerminalBenchAgent, _Environment, list[list[dict[str, Any]]]]:
        agent = TerminalBenchAgent.__new__(TerminalBenchAgent)
        remaining = list(replies)
        seen: list[list[dict[str, Any]]] = []

        async def _complete(messages: list[dict[str, Any]]) -> _Message:
            seen.append([dict(m) for m in messages])
            return remaining.pop(0) if remaining else _Message(content="…")

        monkeypatch.setattr(agent, "_complete", _complete, raising=False)
        monkeypatch.setattr(
            type(agent), "logs_dir", property(lambda _self: tmp_path), raising=False
        )
        return agent, _Environment(), seen

    @pytest.mark.asyncio
    async def test_finish_stops_the_loop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        agent, environment, _ = self._prepare(
            monkeypatch,
            [
                _Message(tool_calls=[_shell("ls /app")]),
                _Message(tool_calls=[_finish()]),
                _Message(tool_calls=[_shell("echo SHOULD_NOT_RUN")]),
            ],
            tmp_path,
        )
        await agent.run("do the thing", environment, None)  # type: ignore[arg-type]
        assert environment.commands == ["ls /app"]

    @pytest.mark.asyncio
    async def test_reply_with_no_tool_call_costs_one_turn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        agent, environment, _ = self._prepare(
            monkeypatch,
            [
                _Message(content="I'll start by looking around."),
                _Message(tool_calls=[_shell("make")]),
                _Message(tool_calls=[_finish()]),
            ],
            tmp_path,
        )
        await agent.run("do the thing", environment, None)  # type: ignore[arg-type]
        assert environment.commands == ["make"]

    @pytest.mark.asyncio
    async def test_malformed_arguments_cost_one_turn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        agent, environment, _ = self._prepare(
            monkeypatch,
            [
                _Message(tool_calls=[_Call(_Function("run_shell", "{oops"))]),
                _Message(tool_calls=[_shell("true")]),
                _Message(tool_calls=[_finish()]),
            ],
            tmp_path,
        )
        await agent.run("do the thing", environment, None)  # type: ignore[arg-type]
        assert environment.commands == ["true"]

    @pytest.mark.asyncio
    async def test_every_tool_call_is_answered(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """An unanswered tool_call makes the provider reject the next request."""
        agent, environment, seen = self._prepare(
            monkeypatch,
            [
                _Message(
                    tool_calls=[
                        _shell("first", "a"),
                        _shell("second", "b"),
                        _shell("third", "c"),
                    ]
                ),
                _Message(tool_calls=[_finish()]),
            ],
            tmp_path,
        )
        await agent.run("do the thing", environment, None)  # type: ignore[arg-type]
        assert environment.commands == ["first"]
        answered = {m["tool_call_id"] for m in seen[-1] if m.get("role") == "tool"}
        assert answered == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_step_exhaustion_is_traced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Otherwise a zero is indistinguishable from a crash."""
        agent, environment, _ = self._prepare(
            monkeypatch, [_Message(tool_calls=[_shell("true")])] * 200, tmp_path
        )
        await agent.run("do the thing", environment, None)  # type: ignore[arg-type]
        trace = (tmp_path / "terminal-bench-trace.jsonl").read_text(encoding="utf-8")
        assert "exhausted_steps" in trace
        assert environment.commands


def test_agent_identity_is_stable() -> None:
    assert TerminalBenchAgent.name() == "terminal-bench-shell-baseline"
