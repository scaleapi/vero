"""Smoke tests for the Terminal-Bench skeleton.

These assert only that the agent satisfies the Harbor interface and can be
constructed and run. They deliberately do not assert anything about how the task
is solved -- that is the thing being built, and a test that pinned a particular
approach would constrain it. Replace or extend these as the implementation grows.
"""

from types import SimpleNamespace

import pytest
from harbor.agents.base import BaseAgent

from terminal_bench_agent.agent import TerminalBenchAgent


class FakeEnvironment:
    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def _agent(tmp_path):
    return TerminalBenchAgent(logs_dir=tmp_path / "logs", model_name="openai/grok-build-0.1")


def test_agent_satisfies_the_harbor_interface(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = _agent(tmp_path)
    assert isinstance(agent, BaseAgent)
    assert isinstance(TerminalBenchAgent.name(), str)
    assert isinstance(agent.version(), str)
    # The provider prefix is stripped for the API call; the gateway allow-lists
    # the bare name.
    assert agent._api_model == "grok-build-0.1"


def test_agent_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        TerminalBenchAgent(logs_dir=tmp_path)


@pytest.mark.asyncio
async def test_setup_prepares_app_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    environment = FakeEnvironment()
    await _agent(tmp_path).setup(environment)
    assert environment.commands == ["mkdir -p /app"]


@pytest.mark.asyncio
async def test_run_completes_without_touching_the_container(tmp_path, monkeypatch):
    """The skeleton scores zero, but it must score -- not error."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = _agent(tmp_path)
    environment = FakeEnvironment()
    context = SimpleNamespace(metadata=None, n_input_tokens=None, n_output_tokens=None, n_cache_tokens=None)

    await agent.run("Create /app/hello.txt containing hello", environment, context)

    assert environment.commands == []
    assert context.n_input_tokens == 0
