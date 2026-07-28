from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("agents")

from agents import Agent, Runner

from vero.agents import AgentContext, CodingAgent
from vero.agents.vero import VeroAgent, default_tool_sets
from vero.candidate import Candidate
from vero.optimization import CandidateProposal
from vero.tools.evaluation import EvaluationTools


class StubSandbox:
    def host_path(self, path):
        return Path(path)


class StubWorkspace:
    def __init__(self, project_path: Path):
        self.project_path = str(project_path)
        self.sandbox = StubSandbox()
        self.accesses = []


class StubEvaluationGateway:
    async def evaluate(self, **kwargs):
        raise AssertionError("the fake run does not invoke tools")

    def budgets(self):
        return {}


class FakeRunResult:
    def __init__(self):
        self.context_wrapper = SimpleNamespace(
            usage={"requests": 1, "input_tokens": 8, "output_tokens": 2}
        )

    def stream_events(self):
        async def _gen():
            return
            yield  # pragma: no cover - marks this as an async generator

        return _gen()

    def to_input_list(self):
        return [
            {"role": "user", "content": "Try a tiled kernel"},
            {"role": "assistant", "content": "Done"},
        ]


def test_agent_credentials_fall_back_to_the_openai_pair(monkeypatch):
    """The native optimizer must work with the OPENAI_* env the rest of vero uses.

    It read only LITELLM_*, so an environment holding just OPENAI_BASE_URL left
    base_url unset and litellm's own resolution posted to a route the proxy does
    not serve -- answering 403 "This route is not publicly accessible", which
    reads like an auth failure rather than a misrouted request.
    """
    from vero.agents.vero import _default_oai_agent

    for name in ("LITELLM_API_KEY", "LITELLM_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1/")
    agent = _default_oai_agent(model="openai/some-model")
    # trailing slash stripped: litellm appends its own route
    assert agent.model.base_url == "https://proxy.example/v1"
    assert agent.model.api_key == "openai-key"

    # LITELLM_* still wins where both are present.
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example/v1")
    agent = _default_oai_agent(model="openai/some-model")
    assert agent.model.base_url == "https://gateway.example/v1"
    assert agent.model.api_key == "litellm-key"


def test_default_tools_use_canonical_evaluation_capability():
    names = {type(tool).__name__ for tool in default_tool_sets()}

    assert "EvaluationTools" in names
    assert not any("Experiment" in name or "Dataset" in name for name in names)


def agent_context(tmp_path: Path, gateway: StubEvaluationGateway) -> AgentContext:
    baseline = Candidate(id="baseline", version="baseline-version")
    proposal = CandidateProposal(
        id="proposal",
        parent_id=baseline.id,
        instruction="Optimize matrix multiplication",
    )
    return AgentContext(
        session_id="session-1",
        workspace=StubWorkspace(tmp_path),
        proposal=proposal,
        parent=baseline,
        evaluation=gateway,
    )


@pytest.mark.asyncio
async def test_vero_agent_implements_canonical_coding_agent_contract(
    tmp_path: Path,
    monkeypatch,
):
    gateway = StubEvaluationGateway()
    evaluation_tools = EvaluationTools()
    agent = VeroAgent(
        oai_agent=Agent(name="test-agent", model="gpt-4.1"),
        tool_sets=[evaluation_tools],
    )
    captured = {}

    def fake_run_streamed(agent, *, input, max_turns, run_config=None):
        captured["agent"] = agent
        captured["input"] = input
        captured["max_turns"] = max_turns
        captured["run_config"] = run_config
        return FakeRunResult()

    monkeypatch.setattr(Runner, "run_streamed", staticmethod(fake_run_streamed))
    context = agent_context(tmp_path, gateway)
    result = await agent.run(
        context=context,
        prompt="Try a tiled kernel",
        max_turns=7,
    )

    assert isinstance(agent, CodingAgent)
    assert captured["input"] == [{"role": "user", "content": "Try a tiled kernel"}]
    assert captured["max_turns"] == 7
    assert captured["run_config"].workflow_name == "vero::session-1"
    assert evaluation_tools.evaluation is gateway
    assert agent._context is context
    assert result.state[-1] == {"role": "assistant", "content": "Done"}
    assert result.metadata["model"] == "gpt-4.1"
    assert result.metadata["usage"]["orchestrator"]["input_tokens"] == 8
