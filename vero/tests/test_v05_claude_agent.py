from __future__ import annotations

from pathlib import Path
import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import ResultMessage

from vero.agents import AgentContext, CodingAgent
from vero.agents.claude_code import ClaudeCodeAgent, default_tool_sets
from vero.candidate import Candidate
from vero.optimization import CandidateProposal
from vero.sandbox import LocalSandbox
from vero.tools.evaluation import EvaluationTools


class StubWorkspace:
    def __init__(self, project_path: Path):
        self.project_path = str(project_path)
        self.sandbox = LocalSandbox(project_path)
        self.accesses = []
        self.saved: list[str] = []

    async def save(self, description: str):
        self.saved.append(description)
        return "saved-version"


class StubEvaluationGateway:
    async def evaluate(self, **kwargs):
        raise AssertionError("the fake client does not invoke tools")

    def budgets(self):
        return {}


class FakeClaudeClient:
    def __init__(self, result: ResultMessage):
        self.result = result
        self.queries: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def query(self, prompt: str):
        self.queries.append(prompt)

    async def receive_response(self):
        yield self.result


def test_default_tools_use_canonical_evaluation_capability():
    assert [type(tool).__name__ for tool in default_tool_sets()] == ["EvaluationTools"]


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
async def test_claude_agent_implements_canonical_coding_agent_contract(
    tmp_path: Path,
):
    gateway = StubEvaluationGateway()
    evaluation_tools = EvaluationTools()
    agent = ClaudeCodeAgent(tool_sets=[evaluation_tools])
    result_message = ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="claude-session",
        usage={"input_tokens": 10, "output_tokens": 3},
        result="done",
    )
    client = FakeClaudeClient(result_message)
    agent._create_client = lambda max_turns=None: client
    events = []

    async def capture(event):
        events.append(event)

    context = agent_context(tmp_path, gateway)
    result = await agent.run(
        context=context,
        prompt="Try a tiled kernel",
        max_turns=7,
        on_event=capture,
    )

    assert isinstance(agent, CodingAgent)
    assert client.queries == ["Try a tiled kernel"]
    assert events == [result_message]
    assert evaluation_tools.evaluation is gateway
    assert agent._context is context
    assert agent.state == {"session_id": "claude-session"}
    assert result.state == {"session_id": "claude-session"}
    assert result.metadata["usage"]["input_tokens"] == 10
    assert context.project_path == tmp_path
    assert context.instructions.startswith("Optimize matrix multiplication\n\n")
    assert "read-only optimization context in `.vero/`" in context.instructions
    assert context.base_version == "baseline-version"
