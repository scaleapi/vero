"""Smoke tests for the GAIA skeleton.

These assert only that the agent satisfies the Harbor interface and can be
constructed and run. They deliberately do not assert anything about how the task
is solved -- that is the thing being built, and a test that pinned a particular
approach would constrain it. Replace or extend these as the implementation grows.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.agents.base import BaseAgent

from gaia_agent import GaiaAgent


class FakeEnvironment:
    def __init__(self):
        self.uploads: list[tuple[Path, str]] = []

    async def exec(self, command, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source_path, target_path):
        self.uploads.append((Path(source_path), target_path))


def _agent(tmp_path):
    return GaiaAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4-mini-2026-03-17",
    )


def test_agent_satisfies_the_harbor_interface(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = _agent(tmp_path)
    assert isinstance(agent, BaseAgent)
    assert isinstance(GaiaAgent.name(), str)
    assert isinstance(agent.version(), str)
    # The provider prefix is stripped for the API call; the gateway allow-lists
    # the bare name.
    assert agent._api_model == "gpt-5.4-mini-2026-03-17"


def test_agent_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires a Harbor model"):
        GaiaAgent(logs_dir=tmp_path)


@pytest.mark.asyncio
async def test_setup_prepares_app_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    await _agent(tmp_path).setup(FakeEnvironment())


@pytest.mark.asyncio
async def test_run_completes_and_writes_the_answer_file(tmp_path, monkeypatch):
    """The skeleton scores zero, but it must score -- not error."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = _agent(tmp_path)
    environment = FakeEnvironment()
    context = SimpleNamespace(
        metadata=None,
        n_input_tokens=None,
        n_output_tokens=None,
        n_cache_tokens=None,
    )

    await agent.run("What is six times seven?", environment, context)

    assert len(environment.uploads) == 1
    _, remote_path = environment.uploads[0]
    assert remote_path == "/app/answer.txt"
