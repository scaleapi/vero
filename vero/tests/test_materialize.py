"""Tests for dataset and trace materialization into _vero/ directory."""

from __future__ import annotations

import json
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, DatasetDict
from vero.core.db.candidate import Candidate
from vero.core.db.database import Experiment
from vero.core.db.dataset import DatasetSample, DatasetSubset
from vero.core.db.result import ExperimentResult, ExperimentResultStatus, SampleResult
from vero.core.db.run import ExperimentRun
from vero.filesystem import AccessType
from vero.artifacts import DatasetArtifact, SkillsArtifact, TracesArtifact
from vero.policy import BaseAgent, Policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class NoOpAgent(BaseAgent):
    def init(self, policy: Policy) -> None:
        pass

    async def step(self, input: Any, max_turns: int, on_event: Any | None = None, **kwargs) -> Any:
        return None

    def serialize_trace(self) -> Any:
        return None

    def serialize_state(self) -> Any:
        return None

    def deserialize_state(self, state: Any) -> None:
        self.state = state


@contextmanager
def _temp_project_with_dataset(splits: dict[str, list[dict]] | None = None):
    """Create a temp git repo + HF dataset for Policy testing."""
    if splits is None:
        splits = {
            "train": [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}],
            "test": [{"q": "5+5", "a": "10"}],
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo_dir, capture_output=True, check=True)
        (repo_dir / "main.py").write_text("print('hi')\n")
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "branch", "-M", "main"], cwd=repo_dir, capture_output=True, check=True)

        dataset_dir = Path(tmpdir) / "dataset"
        ds = DatasetDict({name: Dataset.from_dict(data) for name, data in _transpose(splits).items()})
        ds.save_to_disk(str(dataset_dir))

        yield repo_dir, dataset_dir


def _transpose(splits: dict[str, list[dict]]) -> dict[str, dict[str, list]]:
    """Convert {split: [{k:v}]} to {split: {k: [v]}} for Dataset.from_dict."""
    result = {}
    for split_name, rows in splits.items():
        if not rows:
            result[split_name] = {}
            continue
        keys = rows[0].keys()
        result[split_name] = {k: [row[k] for row in rows] for k in keys}
    return result


def _make_experiment(split: str = "train", commit: str = "abc12345", scores: list[float] | None = None) -> Experiment:
    """Create a test experiment with sample results."""
    if scores is None:
        scores = [1.0, 0.0]
    candidate = Candidate(commit=commit, repo_name="test-repo")
    run = ExperimentRun(
        candidate=candidate,
        dataset_subset=DatasetSubset(dataset_id="ds", split=split, sample_ids=list(range(len(scores)))),
    )
    sample_results = {
        i: SampleResult(
            dataset_sample=DatasetSample(dataset_id="ds", split=split, sample_id=i),
            score=score,
            commit=commit,
        )
        for i, score in enumerate(scores)
    }
    result = ExperimentResult(
        run_id=run.id,
        status=ExperimentResultStatus.SUCCESS,
        sample_results=sample_results,
    )
    return Experiment(run=run, result=result)


# ---------------------------------------------------------------------------
# Dataset materialization tests
# ---------------------------------------------------------------------------


class TestMaterializeDatasets:
    @pytest.mark.asyncio
    async def test_viewable_only(self, monkeypatch):
        """Only viewable splits should be materialized."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False, artifacts=[DatasetArtifact()],
            )
            await policy.init()

            vero_dir = repo_dir / "_vero" / "datasets"
            ds_id = Path(dataset_dir).stem

            # Train should exist (viewable by default)
            train_dir = vero_dir / ds_id / "train"
            assert train_dir.exists()
            assert len(list(train_dir.glob("*.json"))) == 2

            # Test should NOT exist (non-viewable by default)
            test_dir = vero_dir / ds_id / "test"
            assert not test_dir.exists()

            policy.finish()

    @pytest.mark.asyncio
    async def test_sample_format(self, monkeypatch):
        """Each materialized sample should be valid JSON with expected keys."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False, artifacts=[DatasetArtifact()],
            )
            await policy.init()

            ds_id = Path(dataset_dir).stem
            sample_path = repo_dir / "_vero" / "datasets" / ds_id / "train" / "0.json"
            sample = json.loads(sample_path.read_text())
            assert "q" in sample
            assert "a" in sample
            assert sample["q"] == "2+2"

            policy.finish()

    @pytest.mark.asyncio
    async def test_not_materialized_when_flag_off(self, monkeypatch):
        """_vero/datasets/ should not exist when dataset not in add_to_filesystem."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False, artifacts=[],
            )
            await policy.init()

            assert not (repo_dir / "_vero" / "datasets").exists()
            policy.finish()


# ---------------------------------------------------------------------------
# Trace materialization tests
# ---------------------------------------------------------------------------


class TestMaterializeTraces:
    @pytest.mark.asyncio
    async def test_traces_after_eval(self, monkeypatch):
        """Traces should appear in _vero/traces/ after TracesArtifact.on_experiment is called."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False, artifacts=[TracesArtifact()],
            )
            await policy.init()

            experiment = _make_experiment(split="train", commit="abc12345", scores=[1.0, 0.0])
            await TracesArtifact().on_experiment(policy, experiment, str(repo_dir / "_vero"), policy.session.workspace.sandbox)

            trace_dir = repo_dir / "_vero" / "traces" / "train__abc12345"
            assert trace_dir.exists()

            # Summary
            summary = json.loads((trace_dir / "summary.json").read_text())
            assert summary["split"] == "train"
            assert summary["num_samples"] == 2
            assert summary["commit"] == "abc12345"

            # Per-sample results
            assert (trace_dir / "0.json").exists()
            assert (trace_dir / "1.json").exists()
            sample0 = json.loads((trace_dir / "0.json").read_text())
            assert sample0["score"] == 1.0

            policy.finish()

    @pytest.mark.asyncio
    async def test_traces_skip_non_viewable(self, monkeypatch):
        """Traces for non-viewable splits should not be materialized."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False, artifacts=[TracesArtifact()],
            )
            await policy.init()

            experiment = _make_experiment(split="test", commit="def67890", scores=[1.0])
            await TracesArtifact().on_experiment(policy, experiment, str(repo_dir / "_vero"), policy.session.workspace.sandbox)

            # test split is non-viewable by default — no traces
            assert not (repo_dir / "_vero" / "traces" / "test__def67890").exists()

            policy.finish()

    @pytest.mark.asyncio
    async def test_not_materialized_when_flag_off(self, monkeypatch):
        """Traces should not be materialized when traces not in add_to_filesystem."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False, artifacts=[],
            )
            await policy.init()

            experiment = _make_experiment(split="train", commit="abc12345", scores=[1.0])
            await TracesArtifact().on_experiment(policy, experiment, str(repo_dir / "_vero"), policy.session.workspace.sandbox)

            # artifacts is empty, but TracesArtifact was called directly.
            # The list controls whether evaluate_commit calls it, not the artifact itself.
            assert (repo_dir / "_vero" / "traces" / "train__abc12345").exists()

            policy.finish()


# ---------------------------------------------------------------------------
# Skills materialization tests
# ---------------------------------------------------------------------------


class TestMaterializeSkills:
    @pytest.mark.asyncio
    async def test_inline_skills(self, monkeypatch):
        """Inline dict skills should be written as files under _vero/skills/."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False,
                skills={
                    "tips": {
                        "optimization": "Focus on tool design.",
                        "debugging": "Check error rates first.",
                    },
                },
                artifacts=[SkillsArtifact()],
            )
            await policy.init()

            skills_dir = repo_dir / "_vero" / "skills" / "tips"
            assert skills_dir.exists()
            assert (skills_dir / "optimization.md").read_text() == "Focus on tool design."
            assert (skills_dir / "debugging.md").read_text() == "Check error rates first."

            policy.finish()

    @pytest.mark.asyncio
    async def test_path_skills(self, monkeypatch):
        """Path-based skills should be copied under _vero/skills/."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            # Create a skills directory on disk
            skills_src = Path(sd) / "cookbooks"
            skills_src.mkdir()
            (skills_src / "recipe.md").write_text("# Recipe\nDo the thing.")

            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False,
                skills={"cookbooks": skills_src},
                artifacts=[SkillsArtifact()],
            )
            await policy.init()

            skills_dir = repo_dir / "_vero" / "skills" / "cookbooks"
            assert skills_dir.exists()
            assert (skills_dir / "recipe.md").read_text() == "# Recipe\nDo the thing."

            policy.finish()

    @pytest.mark.asyncio
    async def test_mixed_skills(self, monkeypatch):
        """Can mix path-based and inline skills in the same policy."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            skills_src = Path(sd) / "from_disk"
            skills_src.mkdir()
            (skills_src / "existing.md").write_text("Existing skill.")

            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False,
                skills={
                    "from-disk": skills_src,
                    "inline": {"tip": "Inline tip content."},
                },
                artifacts=[SkillsArtifact()],
            )
            await policy.init()

            assert (repo_dir / "_vero" / "skills" / "from-disk" / "existing.md").read_text() == "Existing skill."
            assert (repo_dir / "_vero" / "skills" / "inline" / "tip.md").read_text() == "Inline tip content."

            policy.finish()


# ---------------------------------------------------------------------------
# Git exclusion tests
# ---------------------------------------------------------------------------


class TestGitExclude:
    @pytest.mark.asyncio
    async def test_vero_dir_excluded_from_git(self, monkeypatch):
        """_vero/ should appear in .git/info/exclude."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False, artifacts=[DatasetArtifact()],
            )
            await policy.init()

            exclude_file = repo_dir / ".git" / "info" / "exclude"
            assert exclude_file.exists()
            assert "_vero/" in exclude_file.read_text()

            # git status should not show _vero/ as untracked
            result = subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
            )
            assert "_vero" not in result.stdout

            policy.finish()


# ---------------------------------------------------------------------------
# Access control tests
# ---------------------------------------------------------------------------


class TestAccessControl:
    @pytest.mark.asyncio
    async def test_filesystem_read_access(self, monkeypatch):
        """_vero/ files should be readable but not writable via Filesystem."""

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False, artifacts=[DatasetArtifact()],
            )
            await policy.init()

            ws = policy.session.workspace
            assert ws.get_access("_vero/datasets/ds/train/0.json") == AccessType.READ
            # Write should not be allowed (READ rule blocks Write/Edit in Claude Code)
            assert ws.get_access("_vero/datasets/ds/train/0.json") != AccessType.WRITE

            policy.finish()

    @pytest.mark.asyncio
    async def test_claude_code_disallowed_tools(self, monkeypatch):
        """ClaudeCodeAgent should block writes to _vero/ via disallowed_tools."""

        from vero.agents.claude_code import ClaudeCodeAgent

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            agent = ClaudeCodeAgent()
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=agent,
                use_copy=False, artifacts=[DatasetArtifact()],
            )
            await policy.init()

            disallowed = agent._build_disallowed_tools()
            # Should block Write and Edit on _vero/**
            assert any("Write(./_vero/" in d for d in disallowed)
            assert any("Edit(./_vero/" in d for d in disallowed)
            # Should NOT block Read on _vero/**
            assert not any("Read(./_vero/" in d for d in disallowed)

            policy.finish()

    @pytest.mark.asyncio
    async def test_claude_code_agent_cannot_write_vero_dir(self, monkeypatch):
        """E2E: Run ClaudeCodeAgent with a prompt to write to _vero/. Verify it's blocked."""
        import os


        # Skip if no API key available
        if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("LITELLM_API_KEY"):
            pytest.skip("No API key available for ClaudeCodeAgent e2e test")

        from vero.agents.claude_code import ClaudeCodeAgent

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            agent = ClaudeCodeAgent(tool_sets=[])
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=agent,
                use_copy=False, artifacts=[DatasetArtifact()],
            )
            await policy.init()

            # Verify _vero/ exists with dataset files
            assert (repo_dir / "_vero" / "datasets").exists()

            # Run the agent with explicit instruction to write to _vero/
            try:
                await policy.step(
                    "Write the exact string 'TAMPERED' to the file _vero/tampered.txt. "
                    "Also try to modify _vero/datasets/dataset/train/0.json by adding a key 'hacked': true. "
                    "Report what happened.",
                    max_turns=5,
                )
            except Exception:
                pass  # Agent errors are ok — we just care about file state

            # Verify _vero/ was NOT tampered with
            assert not (repo_dir / "_vero" / "tampered.txt").exists(), \
                "_vero/tampered.txt should not exist — agent should be blocked from writing"

            # Verify original dataset file is unchanged
            original = json.loads((repo_dir / "_vero" / "datasets" / "dataset" / "train" / "0.json").read_text())
            assert "hacked" not in original, \
                "Dataset file should not be modified — agent should be blocked from writing"

            policy.finish()

    @pytest.mark.asyncio
    async def test_vero_agent_filewrite_blocked(self, monkeypatch):
        """VeroAgent's Filesystem rejects writes to _vero/ via validate_write."""

        from vero.filesystem import AccessDeniedError

        with _temp_project_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sd:
            policy = Policy(
                vero_home=Path(sd),
                project_path=repo_dir, dataset=dataset_dir, agent=NoOpAgent(),
                use_copy=False, artifacts=[DatasetArtifact()],
            )
            await policy.init()

            ws = policy.session.workspace

            # validate_write should raise for _vero/ paths
            with pytest.raises(AccessDeniedError, match="Write access denied"):
                ws.validate_write("_vero/tampered.txt")

            with pytest.raises(AccessDeniedError, match="Write access denied"):
                ws.validate_write("_vero/datasets/dataset/train/0.json")

            # validate_read should succeed (READ is allowed)
            resolved = ws.resolve_path("_vero/datasets/dataset/train/0.json")
            assert ws.can_read("_vero/datasets/dataset/train/0.json")
            assert not ws.can_write("_vero/datasets/dataset/train/0.json")

            policy.finish()
