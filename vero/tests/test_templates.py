"""Tests that all Jinja2 templates render without errors."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from vero.agents.vero import VeroAgent
from vero.policy import Policy

pytestmark = pytest.mark.asyncio

PROMPT_TEMPLATES = [
    "prompts/simple_prompt",
    "prompts/agentic_prompt",
    "prompts/claude_code_prompt",
]

INSTRUCTION_TEMPLATES = [
    "instructions/simple_instructions",
    "instructions/agentic_instructions",
    "instructions/cookbook_instructions",
    "instructions/few_shot_instructions",
    "instructions/few_shot_simple_instructions",
    "instructions/few_shot_resources_only_instructions",
    "instructions/few_shot_orchestrator_instructions",
]


def _make_repo(tmp_path: Path) -> Path:
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


def _make_dataset(tmp_path: Path) -> Path:
    ds = DatasetDict({
        "train": Dataset.from_dict({"q": ["2+2"], "a": ["4"]}),
        "validation": Dataset.from_dict({"q": ["3+3"], "a": ["6"]}),
    })
    ds_dir = tmp_path / "dataset"
    ds.save_to_disk(str(ds_dir))
    return ds_dir


async def _make_policy(
    tmp_path: Path,
    monkeypatch,
    agent=None,
    prompt_template: str | None = None,
    instructions_template: str | None = None,
) -> Policy:
    with tempfile.TemporaryDirectory() as sd:
        repo = _make_repo(tmp_path)
        ds_dir = _make_dataset(tmp_path)

        if agent is None:
            agent = VeroAgent(tool_sets=[])
        policy = Policy(
            vero_home=Path(sd),
            project_path=repo,
            dataset=ds_dir,
            agent=agent,
            use_copy=False,
            train_budget=5,
            validation_budget=3,
            prompt_kwargs={"batch_size": 10, "score_threshold": 0.9},
            prompt_template=prompt_template,
            instructions_template=instructions_template,
        )
        await policy.init()
        return policy


AGENTS = [
    ("VeroAgent", lambda: VeroAgent(tool_sets=[])),
]

# Only include ClaudeCodeAgent if available
try:
    from vero.agents.claude_code import ClaudeCodeAgent
    AGENTS.append(("ClaudeCodeAgent", lambda: ClaudeCodeAgent(tool_sets=[])))
except ImportError:
    pass


class TestPromptTemplates:
    @pytest.mark.parametrize("template_name", PROMPT_TEMPLATES)
    @pytest.mark.parametrize("agent_name,agent_factory", AGENTS, ids=[a[0] for a in AGENTS])
    async def test_prompt_renders(self, tmp_path, monkeypatch, template_name, agent_name, agent_factory):
        """Each prompt template should render without errors for each agent type."""
        policy = await _make_policy(tmp_path, monkeypatch, agent=agent_factory(), prompt_template=template_name)
        assert policy.prompt is not None
        assert len(policy.prompt) > 0
        policy.finish()


class TestInstructionTemplates:
    @pytest.mark.parametrize("template_name", INSTRUCTION_TEMPLATES)
    @pytest.mark.parametrize("agent_name,agent_factory", AGENTS, ids=[a[0] for a in AGENTS])
    async def test_instructions_render(self, tmp_path, monkeypatch, template_name, agent_name, agent_factory):
        """Each instruction template should render without errors for each agent type."""
        policy = await _make_policy(tmp_path, monkeypatch, agent=agent_factory(), instructions_template=template_name)
        assert policy.session.instructions is not None
        assert len(policy.session.instructions) > 0
        policy.finish()
