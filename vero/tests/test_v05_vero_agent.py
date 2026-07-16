from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("agents")

from agents import Agent

from vero.agents import AgentContext, CodingAgent
from vero.agents.vero import VeroAgent, default_tool_sets
from vero.candidate import Candidate
from vero.optimization import CandidateProposal
from vero.tools.evaluation import EvaluationTools


class StubWorkspace:
    def __init__(self, project_path: Path):
        self.project_path = str(project_path)
        self.accesses = []


class StubEvaluationGateway:
    async def evaluate_current(self, *, description="Evaluate agent checkpoint"):
        raise AssertionError("the fake run does not invoke tools")

    def budget(self):
        return None


class FakeRunResult:
    def __init__(self):
        self.context_wrapper = SimpleNamespace(
            usage={"requests": 1, "input_tokens": 8, "output_tokens": 2}
        )

    def to_input_list(self):
        return [
            {"role": "user", "content": "Try a tiled kernel"},
            {"role": "assistant", "content": "Done"},
        ]


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

    async def fake_run_agent_with_json_sanitization(**kwargs):
        captured.update(kwargs)
        return FakeRunResult(), None

    monkeypatch.setattr(
        "vero.agents.vero.run_agent_with_json_sanitization",
        fake_run_agent_with_json_sanitization,
    )
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
