"""Behaviour the seed agent must keep, so an optimizer cannot break it silently.

These are not a specification of a good agent -- the agent is deliberately
minimal. They pin the properties whose loss would make a candidate score zero for
reasons unrelated to its ideas: talking to the metered gateway, finding the command
in a reply, surviving a reply with no command, and stopping when told.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest

from terminal_bench_agent.agent import (
    FINISH_SENTINEL,
    MAX_OUTPUT_CHARS,
    TerminalBenchAgent,
)


@dataclass
class _Result:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class _Environment:
    """Records the commands it is asked to run."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, **_: Any) -> _Result:
        self.commands.append(command)
        return _Result(stdout="ok")


class TestCommandExtraction:
    def test_language_tagged_block_wins(self) -> None:
        reply = "I will look around.\n\n```bash\nls -la /app\n```"
        assert TerminalBenchAgent._extract_command(reply) == "ls -la /app"

    def test_untagged_fence_is_accepted(self) -> None:
        assert TerminalBenchAgent._extract_command("```\npwd\n```") == "pwd"

    def test_reply_without_a_block_yields_none(self) -> None:
        """Must be None, not "": the loop distinguishes these two cases."""
        assert TerminalBenchAgent._extract_command("I think we are done.") is None
        assert TerminalBenchAgent._extract_command("```bash\n\n```") is None
        assert TerminalBenchAgent._extract_command("") is None

    def test_multiline_command_survives_intact(self) -> None:
        reply = "```bash\ncd /app\nmake test\n```"
        assert TerminalBenchAgent._extract_command(reply) == "cd /app\nmake test"


class TestOutputTruncation:
    def test_short_output_is_untouched(self) -> None:
        assert TerminalBenchAgent._truncate("hello") == "hello"

    def test_long_output_keeps_both_ends(self) -> None:
        """Head and tail, because a traceback's cause and its message differ."""
        value = ("a" * MAX_OUTPUT_CHARS) + "MIDDLE" + ("z" * MAX_OUTPUT_CHARS)
        result = TerminalBenchAgent._truncate(value)
        assert len(result) < len(value)
        assert result.startswith("a")
        assert result.endswith("z")
        assert "characters omitted" in result
        assert "MIDDLE" not in result


class TestGatewayCredentials:
    """The target must never reach the upstream directly.

    OPENAI_* inside the eval container can point at the unmetered upstream, so a
    fallback would bypass both metering and the per-scope model allow-list without
    failing -- the worst kind of regression, because everything still appears to
    work.
    """

    @staticmethod
    def _construct(tmp_path: Any) -> TerminalBenchAgent:
        return TerminalBenchAgent(logs_dir=tmp_path, model_name="test-model")

    def test_missing_gateway_credentials_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.delenv("VERO_AGENT_INFERENCE_API_KEY", raising=False)
        monkeypatch.delenv("VERO_AGENT_INFERENCE_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "upstream-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://upstream.example/v1")
        with pytest.raises(RuntimeError, match="VERO_AGENT_INFERENCE"):
            self._construct(tmp_path)

    def test_error_names_the_reason_not_just_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.delenv("VERO_AGENT_INFERENCE_API_KEY", raising=False)
        monkeypatch.delenv("VERO_AGENT_INFERENCE_BASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="unmetered"):
            self._construct(tmp_path)

    def test_gateway_credentials_are_used_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The positive case, so the guard cannot be satisfied by always raising."""
        monkeypatch.setenv("VERO_AGENT_INFERENCE_API_KEY", "scoped-token")
        monkeypatch.setenv("VERO_AGENT_INFERENCE_BASE_URL", "https://gw.example/v1")
        agent = self._construct(tmp_path)
        assert "gw.example" in str(agent._client.base_url)


class TestRunLoop:
    """Drives run() with a scripted model, no network."""

    @staticmethod
    def _prepare(
        monkeypatch: pytest.MonkeyPatch, replies: list[str], tmp_path: Any
    ) -> tuple[TerminalBenchAgent, _Environment]:
        agent = TerminalBenchAgent.__new__(TerminalBenchAgent)
        remaining = list(replies)

        async def _complete(_messages: list[dict[str, str]]) -> str:
            return remaining.pop(0) if remaining else "done"

        monkeypatch.setattr(agent, "_complete", _complete, raising=False)
        monkeypatch.setattr(
            type(agent), "logs_dir", property(lambda _self: tmp_path), raising=False
        )
        return agent, _Environment()

    @pytest.mark.asyncio
    async def test_runs_commands_then_stops_on_the_sentinel(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        agent, environment = self._prepare(
            monkeypatch,
            [
                "looking\n```bash\nls /app\n```",
                f"done\n```bash\necho {FINISH_SENTINEL}\n```",
                "```bash\necho SHOULD_NOT_RUN\n```",
            ],
            tmp_path,
        )
        await agent.run("do the thing", environment, None)  # type: ignore[arg-type]
        assert environment.commands == ["ls /app"], (
            "the sentinel must end the loop without being executed, and nothing "
            "after it may run"
        )

    @pytest.mark.asyncio
    async def test_a_reply_without_a_command_is_recoverable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A malformed reply must cost one turn, not the whole trial."""
        agent, environment = self._prepare(
            monkeypatch,
            [
                "I am thinking about it.",
                "now\n```bash\nmake\n```",
                f"```bash\necho {FINISH_SENTINEL}\n```",
            ],
            tmp_path,
        )
        await agent.run("do the thing", environment, None)  # type: ignore[arg-type]
        assert environment.commands == ["make"]

    @pytest.mark.asyncio
    async def test_step_exhaustion_is_recorded_and_returns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Running out of steps ends the trial cleanly; the container is the answer.

        It must also leave a trace saying so, or a zero looks identical to a crash.
        """
        agent, environment = self._prepare(
            monkeypatch, ["```bash\ntrue\n```"] * 200, tmp_path
        )
        await agent.run("do the thing", environment, None)  # type: ignore[arg-type]
        trace = (tmp_path / "terminal-bench-trace.jsonl").read_text(encoding="utf-8")
        assert "exhausted_steps" in trace
        assert len(environment.commands) > 0


def test_agent_identity_is_stable() -> None:
    """The name is how the harness reports the target; keep it recognisable."""
    assert TerminalBenchAgent.name() == "terminal-bench-shell-baseline"
    assert os.sep not in TerminalBenchAgent.name()
