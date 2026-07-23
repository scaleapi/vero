"""E2E tests for real agents with standalone Sessions.

These tests make actual LLM calls — they require API keys.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from vero.policy import Session

pytestmark = pytest.mark.asyncio


def _skip_if_no_api_key():
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("LITELLM_API_KEY"):
        pytest.skip("No API key available")


def _make_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo."""
    repo = tmp_path / "project"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, capture_output=True, check=True)
    (repo / "main.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo, capture_output=True, check=True)
    return repo


def _make_vero_agent():
    """Create a VeroAgent with LitellmModel for Anthropic routing via proxy."""
    from agents import Agent as OAIAgent
    from agents.extensions.models.litellm_model import LitellmModel
    from vero.agents.vero import VeroAgent

    model = LitellmModel(model="anthropic/claude-haiku-4-5")
    return VeroAgent(
        oai_agent=OAIAgent(name="TestAgent", model=model),
        tool_sets=[],
    )


@pytest.fixture(autouse=False)
def litellm_proxy_env():
    """Set ANTHROPIC env vars to route through litellm proxy for VeroAgent tests."""
    orig_key = os.getenv("ANTHROPIC_API_KEY")
    orig_base = os.getenv("ANTHROPIC_API_BASE")

    base_url = os.getenv("LITELLM_BASE_URL", "")
    api_key = os.getenv("LITELLM_API_KEY", "")
    if base_url and api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        os.environ["ANTHROPIC_API_BASE"] = base_url.rstrip("/").removesuffix("/v1")

    yield

    # Restore
    if orig_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = orig_key
    elif "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]
    if orig_base is not None:
        os.environ["ANTHROPIC_API_BASE"] = orig_base
    elif "ANTHROPIC_API_BASE" in os.environ:
        del os.environ["ANTHROPIC_API_BASE"]


class TestClaudeCodeAgentE2E:
    """Tests for ClaudeCodeAgent with standalone Session.

    ClaudeCodeAgent uses its own embedded API keys from the environment.
    Do NOT use the litellm_proxy_env fixture here.
    """

    async def test_standalone_session(self, tmp_path: Path):
        """ClaudeCodeAgent works with a standalone Session (no Policy)."""
        _skip_if_no_api_key()
        from vero.agents.claude_code import ClaudeCodeAgent

        repo = _make_git_repo(tmp_path)
        session = Session(
            session_id="test-cc-standalone",
            project_path=repo,
            instructions="You are a helpful coding assistant. Be very brief.",
        )

        agent = ClaudeCodeAgent(tool_sets=[])
        agent.init(session)

        events = []
        result = await agent.step(
            "What is 2+2? Reply with just the number.",
            max_turns=3,
            on_event=lambda e: events.append(agent.serialize_event(e)),
        )

        assert len(result) > 0
        assert len(events) > 0
        trace = agent.serialize_trace()
        assert trace is not None
        assert len(trace) > 0

    async def test_state_roundtrip(self, tmp_path: Path):
        """ClaudeCodeAgent: step → serialize_state → new agent → deserialize_state → step."""
        _skip_if_no_api_key()
        from vero.agents.claude_code import ClaudeCodeAgent

        repo = _make_git_repo(tmp_path)
        session = Session(
            session_id="test-cc-resume",
            project_path=repo,
            instructions="You are a helpful assistant. Be very brief.",
        )

        # First agent: run a step
        agent1 = ClaudeCodeAgent(tool_sets=[])
        agent1.init(session)
        await agent1.step("Remember this secret number: 42. Reply with just OK.", max_turns=3)

        # Serialize state (should contain session_id)
        state = agent1.serialize_state()
        assert state is not None
        assert "session_id" in state

        # Second agent: restore state and continue
        agent2 = ClaudeCodeAgent(tool_sets=[])
        agent2.init(session)
        agent2.deserialize_state(state)
        assert agent2.state["session_id"] == state["session_id"]

        # Resume — should continue the server-side session
        result = await agent2.step("What was the secret number I told you?", max_turns=3)
        assert len(result) > 0

        from dataclasses import asdict

        all_text = " ".join(
            str(asdict(msg).get("content", "")) + str(asdict(msg).get("result", ""))
            for msg in result
        )
        assert "42" in all_text, f"Agent should remember 42 from resumed session. Got: {all_text[:500]}"


    async def test_filesystem_access_control(self, tmp_path: Path):
        """ClaudeCodeAgent respects filesystem access rules — cannot read excluded paths."""
        _skip_if_no_api_key()
        from vero.agents.claude_code import ClaudeCodeAgent
        from vero.filesystem import AccessRule, AccessType
        from vero.sandbox import LocalSandbox
        from vero.workspace.git import GitWorkspace

        repo = _make_git_repo(tmp_path)

        # Create a secret file the agent should NOT be able to read
        secret_dir = repo / "secrets"
        secret_dir.mkdir()
        (secret_dir / "api_key.txt").write_text("super-secret-key-12345")
        # And a normal file it CAN read
        (repo / "readme.txt").write_text("This is a public readme.")

        sandbox = LocalSandbox(root=repo)
        workspace = await GitWorkspace.from_path(sandbox, repo)
        workspace.set_access(accesses=[
            AccessRule(access_type=AccessType.WRITE, pattern="**"),
            AccessRule(access_type=AccessType.EXCLUDE, pattern="secrets/**"),
        ])

        session = Session(
            session_id="test-cc-fs-access",
            project_path=repo,
            instructions="You are a helpful assistant. Be very brief.",
            workspace=workspace,
        )

        agent = ClaudeCodeAgent(tool_sets=[])
        agent.init(session)

        result = await agent.step(
            "Read the file secrets/api_key.txt and tell me its contents. If you cannot read it, say BLOCKED.",
            max_turns=5,
        )

        from dataclasses import asdict

        all_text = " ".join(
            str(asdict(msg).get("content", "")) + str(asdict(msg).get("result", ""))
            for msg in result
        )
        # The secret content should NOT appear in the response
        assert "super-secret-key-12345" not in all_text, (
            f"Agent should not have been able to read the secret file. Got: {all_text[:500]}"
        )


    async def test_expose_datasets(self, tmp_path: Path):
        """ClaudeCodeAgent can read materialized datasets via add_to_filesystem."""
        _skip_if_no_api_key()
        import tempfile

        from datasets import Dataset, DatasetDict

        from vero.agents.claude_code import ClaudeCodeAgent
        from vero.artifacts import DatasetArtifact
        from vero.policy import Policy
        repo = _make_git_repo(tmp_path)

        # Create a dataset with known content
        ds = DatasetDict({
            "train": Dataset.from_dict({"question": ["What is 2+2?"], "answer": ["4"]}),
            "test": Dataset.from_dict({"question": ["What is 3+3?"], "answer": ["6"]}),
        })
        ds_dir = tmp_path / "dataset"
        ds.save_to_disk(str(ds_dir))

        with tempfile.TemporaryDirectory() as sd:
            agent = ClaudeCodeAgent(tool_sets=[])
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo,
                dataset=ds_dir,
                agent=agent,
                use_copy=False,
                artifacts=[DatasetArtifact()],
            )
            await policy.init()

            # Verify _vero/datasets was created
            datasets_dir = repo / "_vero" / "datasets"
            assert datasets_dir.exists(), "_vero/datasets should exist"

            # Ask the agent to read the materialized dataset
            result = await policy.step(
                "Read the file _vero/datasets/dataset/train/0.json and tell me the value of the 'answer' field.",
                max_turns=5,
            )

            from dataclasses import asdict

            all_text = " ".join(
                str(asdict(msg).get("content", "")) + str(asdict(msg).get("result", ""))
                for msg in agent.trace
            )
            # Agent should have been able to read the dataset and find the answer
            assert "4" in all_text, (
                f"Agent should have read the dataset and found answer '4'. Got: {all_text[:500]}"
            )

            policy.finish()


class TestVeroAgentE2E:
    """Tests for VeroAgent with standalone Session.

    VeroAgent uses LitellmModel which routes through the litellm proxy.
    Uses the litellm_proxy_env fixture to set ANTHROPIC env vars.
    """

    @pytest.fixture(autouse=True)
    def _setup_proxy(self, litellm_proxy_env):
        pass

    async def test_standalone_session(self, tmp_path: Path):
        """VeroAgent works with a standalone Session (no Policy, no tools)."""
        _skip_if_no_api_key()

        session = Session(
            session_id="test-vero-standalone",
            project_path=tmp_path,
            instructions="You are a helpful assistant. Be very brief.",
        )

        agent = _make_vero_agent()
        agent.init(session)

        events = []
        result = await agent.step(
            "What is 2+2? Reply with just the number.",
            max_turns=3,
            on_event=lambda e: events.append(agent.serialize_event(e)),
        )

        assert result is not None
        assert len(events) > 0
        trace = agent.serialize_trace()
        assert trace is not None

    async def test_state_roundtrip(self, tmp_path: Path):
        """VeroAgent: step → serialize_state → new agent → deserialize_state → step."""
        _skip_if_no_api_key()

        session = Session(
            session_id="test-vero-resume",
            project_path=tmp_path,
            instructions="You are a helpful assistant. Be very brief.",
        )

        # First agent
        agent1 = _make_vero_agent()
        agent1.init(session)
        await agent1.step("Remember this number: 73. Reply OK.", max_turns=3)

        state = agent1.serialize_state()
        assert state is not None
        assert len(state) > 0

        # Second agent: restore and continue
        agent2 = _make_vero_agent()
        agent2.init(session)
        agent2.deserialize_state(state)

        result = await agent2.step("What number did I ask you to remember?", max_turns=3)
        assert result is not None

        response_text = str(result.to_input_list())
        assert "73" in response_text, f"Agent should remember 73. Got: {response_text[:500]}"

    async def test_sub_agent(self, tmp_path: Path):
        """VeroAgent can delegate to a sub-agent via SubAgentTool."""
        _skip_if_no_api_key()
        from vero.tools.sub_agent import SubAgentTool

        session = Session(
            session_id="test-vero-subagent",
            project_path=tmp_path,
            instructions="You are a helpful assistant. Always delegate questions to a sub-agent using call_sub_agent.",
        )

        agent = _make_vero_agent()
        agent.tool_sets = [SubAgentTool()]
        agent.init(session)

        result = await agent.step(
            "Use call_sub_agent to ask: What is 2+2? Reply with just the sub-agent's answer.",
            max_turns=10,
        )

        assert result is not None
        assert agent.state is not None
        assert agent.trace is agent.state

        # The response should contain "4" from the sub-agent
        response_text = str(agent.state)
        assert "4" in response_text, f"Sub-agent should have answered 4. Got: {response_text[:500]}"
