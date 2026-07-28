"""Tests for DB reconstruction from experiments/, Policy.fork(), and SessionLogger."""

from __future__ import annotations

import json
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, DatasetDict
from vero.core.constants import (
    evaluation_parameters_basename,
    result_metadata_basename,
    samples_dir_name,
)
from vero.core.db.candidate import Candidate
from vero.core.db.database import ExperimentDatabase
from vero.core.db.dataset import DatasetSample, DatasetSubset
from vero.core.db.result import ExperimentResultStatus, SampleResult
from vero.core.db.run import ExperimentRun
from vero.core.evaluation import EvaluationParameters
from vero.core.sessions import find_project_dir_in_session, get_session_experiments_dir
from vero.policy import BaseAgent, Policy, Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(commit: str = "abc123", repo_name: str = "test-repo") -> Candidate:
    return Candidate(commit=commit, repo_name=repo_name)


def _make_run(candidate: Candidate | None = None, split: str = "test") -> ExperimentRun:
    candidate = candidate or _make_candidate()
    return ExperimentRun(
        candidate=candidate,
        dataset_subset=DatasetSubset(dataset_id="ds1", split=split, sample_ids=[0, 1, 2]),
    )


def _make_sample_result(sample_id: int, score: float, split: str = "test") -> SampleResult:
    return SampleResult(
        dataset_sample=DatasetSample(dataset_id="ds1", split=split, sample_id=sample_id),
        score=score,
        commit="abc123",
        result_id="result-1",
    )


def _write_experiment_to_disk(
    experiments_dir: Path,
    result_id: str,
    run: ExperimentRun,
    sample_scores: list[float],
    status: str | None = "success",
) -> None:
    """Write a complete experiment to disk in the expected format."""
    result_dir = experiments_dir / result_id
    result_dir.mkdir(parents=True)

    # Write evaluation_parameters.json
    params = EvaluationParameters(
        result_id=result_id,
        run=run,
        session_id="test-session",
    )
    (result_dir / evaluation_parameters_basename).write_text(params.model_dump_json(indent=2))

    # Write samples
    samples_dir = result_dir / samples_dir_name
    samples_dir.mkdir()
    sample_ids = run.dataset_subset.sample_ids or list(range(len(sample_scores)))
    for sid, score in zip(sample_ids, sample_scores):
        sr = _make_sample_result(sid, score, split=run.dataset_subset.split)
        (samples_dir / f"{sid}.json").write_text(sr.model_dump_json(indent=2))

    # Write result_metadata.json
    if status is not None:
        metadata = {"id": result_id, "run_id": run.id, "status": status}
        (result_dir / result_metadata_basename).write_text(json.dumps(metadata))


# ---------------------------------------------------------------------------
# Feature A: DB reconstruction from experiments/
# ---------------------------------------------------------------------------


class TestDBReconstruction:
    def test_from_experiments_dir_basic(self, tmp_path: Path):
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        run = _make_run()
        _write_experiment_to_disk(experiments_dir, "result-1", run, [1.0, 0.0, 1.0])

        db = ExperimentDatabase.from_experiments_dir(experiments_dir, db_id="test")

        assert len(db.candidates) == 1
        assert len(db.runs) == 1
        assert len(db.results) == 1

        result = list(db.results.values())[0]
        assert result.status == ExperimentResultStatus.SUCCESS
        assert len(result.sample_results) == 3
        assert result.sample_results[0].score == 1.0
        assert result.sample_results[1].score == 0.0

    def test_from_experiments_dir_multiple_experiments(self, tmp_path: Path):
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        run1 = _make_run(_make_candidate("commit1"))
        run2 = _make_run(_make_candidate("commit2"))
        _write_experiment_to_disk(experiments_dir, "result-1", run1, [1.0, 1.0, 1.0])
        _write_experiment_to_disk(experiments_dir, "result-2", run2, [0.0, 0.0, 0.0])

        db = ExperimentDatabase.from_experiments_dir(experiments_dir, db_id="test")

        assert len(db.candidates) == 2
        assert len(db.results) == 2

    def test_from_experiments_dir_missing_metadata(self, tmp_path: Path):
        """Status should be computed from error rate when result_metadata.json is absent."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        run = _make_run()
        _write_experiment_to_disk(experiments_dir, "result-1", run, [1.0, 0.0, 1.0], status=None)

        db = ExperimentDatabase.from_experiments_dir(experiments_dir, db_id="test")

        result = list(db.results.values())[0]
        # With default error_rate_threshold=0.1, 0/3 errors → SUCCESS
        assert result.status == ExperimentResultStatus.SUCCESS

    def test_from_experiments_dir_corrupt_entry_skipped(self, tmp_path: Path):
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        # Good experiment
        run = _make_run()
        _write_experiment_to_disk(experiments_dir, "good-result", run, [1.0, 0.5, 0.0])

        # Corrupt experiment (missing evaluation_parameters.json)
        bad_dir = experiments_dir / "bad-result"
        bad_dir.mkdir()
        (bad_dir / "samples").mkdir()

        db = ExperimentDatabase.from_experiments_dir(experiments_dir, db_id="test")

        assert len(db.results) == 1
        assert "good-result" in db.results

    def test_from_experiments_dir_empty(self, tmp_path: Path):
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        db = ExperimentDatabase.from_experiments_dir(experiments_dir, db_id="test")
        assert len(db.results) == 0

    def test_from_experiments_dir_nonexistent(self, tmp_path: Path):
        db = ExperimentDatabase.from_experiments_dir(tmp_path / "nonexistent", db_id="test")
        assert len(db.results) == 0

    def test_reconstructed_db_matches_scores(self, tmp_path: Path):
        """Verify reconstructed DB produces correct experiment scores."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        run = _make_run()
        _write_experiment_to_disk(experiments_dir, "result-1", run, [1.0, 0.0, 1.0])

        db = ExperimentDatabase.from_experiments_dir(experiments_dir, db_id="test")
        result = list(db.results.values())[0]

        # 2/3 non-null scores of 1.0 and 0.0 → mean ~0.667
        score = result.score(fill_score=None)
        assert score is not None
        assert abs(score - 2.0 / 3) < 0.01


# ---------------------------------------------------------------------------
# Feature B: Policy.fork() (unit-level, no real Policy init)
# ---------------------------------------------------------------------------


class TestFork:
    def test_find_project_dir_in_session(self, tmp_path: Path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        session_dir = sessions_dir / "session-1"
        session_dir.mkdir()

        # No project yet
        assert find_project_dir_in_session(sessions_dir, "session-1") is None

        # Add a project with .git/
        project_dir = session_dir / "my-project"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()

        found = find_project_dir_in_session(sessions_dir, "session-1")
        assert found == project_dir

    def test_fork_copies_experiments(self, tmp_path: Path):
        """Test that Policy.fork copies experiments directory."""

        # Set up source session
        source_id = "source-session"
        source_dir = tmp_path / source_id
        source_dir.mkdir()

        # Add project with .git
        project_dir = source_dir / "my-project"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()
        (project_dir / "main.py").write_text("print('hello')")

        # Add experiments
        experiments_dir = source_dir / "experiments"
        experiments_dir.mkdir()
        run = _make_run()
        _write_experiment_to_disk(experiments_dir, "result-1", run, [1.0, 0.5])

        # Fork (we call the underlying logic directly, not Policy.fork, to avoid full Policy construction)
        import shutil

        new_session_id = "forked-session"
        new_session_dir = tmp_path / new_session_id
        new_session_dir.mkdir()

        # Copy project
        dest_project = new_session_dir / project_dir.name
        shutil.copytree(project_dir, dest_project)

        # Copy experiments
        dest_experiments = new_session_dir / "experiments"
        shutil.copytree(experiments_dir, dest_experiments)

        # Verify project copied (including .git)
        assert (dest_project / ".git").exists()
        assert (dest_project / "main.py").read_text() == "print('hello')"

        # Verify experiments copied
        assert (dest_experiments / "result-1" / evaluation_parameters_basename).exists()
        assert (dest_experiments / "result-1" / samples_dir_name / "0.json").exists()

        # Verify DB can be reconstructed from forked experiments
        db = ExperimentDatabase.from_experiments_dir(dest_experiments, db_id=new_session_id)
        assert len(db.results) == 1
        assert list(db.results.values())[0].sample_results[0].score == 1.0

    def test_fork_independence(self, tmp_path: Path):
        """Changes to forked experiments don't affect source."""
        import shutil

        source_experiments = tmp_path / "source" / "experiments"
        source_experiments.mkdir(parents=True)
        run = _make_run()
        _write_experiment_to_disk(source_experiments, "result-1", run, [1.0])

        dest_experiments = tmp_path / "dest" / "experiments"
        shutil.copytree(source_experiments, dest_experiments)

        # Modify forked experiment
        (dest_experiments / "result-1" / samples_dir_name / "0.json").write_text("{}")

        # Source should be unchanged
        source_sample = json.loads(
            (source_experiments / "result-1" / samples_dir_name / "0.json").read_text()
        )
        assert source_sample["score"] == 1.0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# E2E: Policy with mock agent
# ---------------------------------------------------------------------------


class MockAgent(BaseAgent):
    """Minimal agent that emits a few events and does nothing."""

    def __init__(self):
        super().__init__()
        self._session = None
        self._trace: list[dict] = []

    def init(self, session) -> None:
        self._session = session

    async def step(self, input: Any, max_turns: int, on_event: Any | None = None, **kwargs) -> Any:
        events = [
            {"role": "user", "content": str(input)},
            {"role": "assistant", "content": "I'll look at the code."},
            {"role": "tool", "name": "read_file", "content": "file contents here"},
            {"role": "assistant", "content": "Done. No changes needed."},
        ]
        for event in events:
            self._trace.append(event)
            if on_event is not None:
                on_event(event)
        return events

    def serialize_trace(self) -> Any:
        return self._trace

    def serialize_state(self) -> Any:
        return self._trace if self._trace else None

    def deserialize_state(self, state: Any) -> None:
        self.state = state

    def serialize_event(self, event: Any) -> dict:
        if isinstance(event, dict):
            return event
        return {"raw": str(event)}


class CrashingAgent(BaseAgent):
    """Agent that emits some events then raises an error."""

    def __init__(self, crash_after: int = 2):
        super().__init__()
        self._session = None
        self._trace: list[dict] = []
        self._crash_after = crash_after

    def init(self, session) -> None:
        self._session = session

    def serialize_trace(self) -> Any:
        return self._trace

    async def step(self, input: Any, max_turns: int, on_event: Any | None = None, **kwargs) -> Any:
        events = [
            {"role": "user", "content": str(input)},
            {"role": "assistant", "content": "Starting work..."},
            {"role": "tool", "name": "run_eval", "content": "running evaluation"},
            {"role": "assistant", "content": "Analyzing results..."},
        ]
        for i, event in enumerate(events):
            self._trace.append(event)
            if on_event is not None:
                on_event(event)
            if i + 1 >= self._crash_after:
                raise RuntimeError("Agent crashed mid-execution!")
        return events

    def serialize_trace(self) -> Any:
        return self._trace

    def serialize_state(self) -> Any:
        return self._trace if self._trace else None

    def deserialize_state(self, state: Any) -> None:
        self.state = state

    def serialize_event(self, event: Any) -> dict:
        if isinstance(event, dict):
            return event
        return {"raw": str(event)}


@contextmanager
def _temp_git_repo_with_dataset():
    """Create a temp git repo with a HuggingFace dataset for Policy testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "test-project"
        repo_dir.mkdir()

        # Init git repo
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.name", "test"], cwd=repo_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True, check=True
        )
        (repo_dir / "main.py").write_text("print('hello')\n")
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, capture_output=True, check=True)

        # Rename to main branch
        subprocess.run(["git", "branch", "-M", "main"], cwd=repo_dir, capture_output=True, check=True)

        # Create dataset
        dataset_dir = Path(tmpdir) / "dataset"
        ds = DatasetDict({"test": Dataset.from_dict({"task": ["a", "b", "c"]})})
        ds.save_to_disk(str(dataset_dir))

        yield repo_dir, dataset_dir


class TestPolicyE2E:
    @pytest.mark.asyncio
    async def test_policy_init_step_events(self, monkeypatch):
        """Full lifecycle: init → step (with events) → finish → verify trace on disk."""
        with _temp_git_repo_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sessions_dir:
            agent = MockAgent()
            policy = Policy(
                vero_home=Path(sessions_dir),
                project_path=repo_dir,
                dataset=dataset_dir,
                agent=agent,
                use_copy=False,
            )

            await policy.init()

            # Step — should fire on_event callbacks (including SessionLogger)
            await policy.step("optimize the code", max_turns=10)

            policy.finish()

            # Verify per-turn trace files were written
            from vero.core.sessions import get_session_dir

            trace_dir = get_session_dir(policy.sessions_dir, policy.session_id) / "agent_trace"
            assert trace_dir.exists()

            files = sorted(trace_dir.glob("turn_*.json"))
            assert len(files) == 4  # MockAgent emits 4 events

            events = [json.loads(f.read_text()) for f in files]
            assert events[0]["role"] == "user"
            assert events[1]["role"] == "assistant"
            assert events[2]["role"] == "tool"
            assert events[3]["role"] == "assistant"

            # Verify turn numbers increment
            assert [e["turn"] for e in events] == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_policy_fork_preserves_experiments(self, monkeypatch):
        """Fork a session with experiments, verify they're accessible in the fork."""
        with _temp_git_repo_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sessions_dir:
            # Create source policy and add a fake experiment
            agent = MockAgent()
            policy = Policy(
                vero_home=Path(sessions_dir),
                project_path=repo_dir,
                dataset=dataset_dir,
                agent=agent,
                use_copy=False,
            )
            await policy.init()

            # Manually write an experiment to the session's experiments/ dir
            experiments_dir = get_session_experiments_dir(policy.sessions_dir, policy.session_id)
            experiments_dir.mkdir(parents=True, exist_ok=True)
            run = _make_run()
            _write_experiment_to_disk(experiments_dir, "result-1", run, [1.0, 0.5, 0.0])

            policy.finish()
            source_session_id = policy.session_id

            # Fork
            forked_policy = Policy.fork(
                source_session_id,
                vero_home=Path(sessions_dir),
                project_path=repo_dir,
                dataset=dataset_dir,
                agent=MockAgent(),
                use_copy=False,
            )
            await forked_policy.init()

            # Verify DB was reconstructed from forked experiments
            assert forked_policy.session.db is not None
            assert len(forked_policy.session.db.results) == 1

            result = list(forked_policy.session.db.results.values())[0]
            assert len(result.sample_results) == 3
            assert result.sample_results[0].score == 1.0

            forked_policy.finish()

    @pytest.mark.asyncio
    async def test_policy_step_crash_preserves_trace(self, monkeypatch):
        """If agent crashes mid-step, events emitted before the crash are on disk."""
        with _temp_git_repo_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sessions_dir:
            agent = CrashingAgent(crash_after=2)
            policy = Policy(
                vero_home=Path(sessions_dir),
                project_path=repo_dir,
                dataset=dataset_dir,
                agent=agent,
                use_copy=False,
            )
            await policy.init()

            with pytest.raises(RuntimeError, match="Agent crashed mid-execution"):
                await policy.step("optimize", max_turns=10)

            # Even though step() crashed, events before the crash should be on disk
            from vero.core.sessions import get_session_dir

            trace_dir = get_session_dir(policy.sessions_dir, policy.session_id) / "agent_trace"
            assert trace_dir.exists()

            files = sorted(trace_dir.glob("turn_*.json"))
            assert len(files) == 2  # 2 events emitted before crash

            events = [json.loads(f.read_text()) for f in files]
            assert events[0]["role"] == "user"
            assert events[1]["role"] == "assistant"
            assert events[1]["content"] == "Starting work..."

            # finish() should still work (trace writer closes cleanly)
            policy.finish()

    @pytest.mark.asyncio
    async def test_policy_resume_rebuilds_db(self, monkeypatch):
        """Resume from an existing session, verify DB is reconstructed."""
        with _temp_git_repo_with_dataset() as (repo_dir, dataset_dir), tempfile.TemporaryDirectory() as sessions_dir:
            # Create a session with experiments
            agent = MockAgent()
            policy = Policy(
                vero_home=Path(sessions_dir),
                project_path=repo_dir,
                dataset=dataset_dir,
                agent=agent,
                use_copy=False,
                isolate=True,
            )
            await policy.init()
            session_id = policy.session_id

            # Write an experiment
            experiments_dir = get_session_experiments_dir(policy.sessions_dir, session_id)
            experiments_dir.mkdir(parents=True, exist_ok=True)
            run = _make_run()
            _write_experiment_to_disk(experiments_dir, "result-1", run, [1.0, 0.5, 0.0])

            policy.finish()

            # Resume from the same session
            resumed = Policy.resume(
                session_id,
                vero_home=Path(sessions_dir),
                agent=MockAgent(),
                dataset=dataset_dir,
                use_copy=False,
            )
            await resumed.init()

            # DB should be reconstructed from experiments on disk
            assert resumed.session.db is not None
            assert len(resumed.session.db.results) == 1
            result = list(resumed.session.db.results.values())[0]
            assert result.sample_results[0].score == 1.0

            resumed.finish()


# ---------------------------------------------------------------------------
# Session standalone tests
# ---------------------------------------------------------------------------


class TestSessionStandalone:
    @pytest.mark.asyncio
    async def test_minimal_session_with_mock_agent(self, tmp_path: Path):
        """Agent works with a minimal Session (no filesystem, no db, no evaluator)."""
        session = Session(session_id="test-123", project_path=tmp_path)
        agent = MockAgent()
        agent.init(session)

        events = []

        def capture(event):
            events.append(event)

        result = await agent.step("hello", max_turns=5, on_event=capture)

        assert len(result) == 4
        assert len(events) == 4
        assert events[0]["role"] == "user"
        assert agent._session.session_id == "test-123"

    @pytest.mark.asyncio
    async def test_session_with_filesystem(self, tmp_path: Path):
        """Session with just filesystem — FileWrite tool can bind and enforce access."""
        from vero.filesystem import AccessDeniedError, AccessRule, AccessType, Filesystem
        from vero.sandbox import LocalSandbox
        from vero.tools.file_write import FileWrite
        from vero.workspace.base import Workspace

        sandbox = LocalSandbox(root=tmp_path)

        class _TestWorkspace(Workspace):
            def __init__(self):
                self._fs = Filesystem(root=tmp_path, default_access=AccessType.WRITE)

            @property
            def sandbox(self):
                return sandbox

            @property
            def root(self):
                return str(tmp_path)

            @property
            def project_path(self):
                return str(tmp_path)

            @property
            def name(self):
                return "test"

            async def current_version(self):
                return ""

            async def save(self, message="Save"):
                return ""

            async def restore(self, version_id, message=None):
                return ""

            async def diff(self, from_version=None, to_version=None):
                return ""

            async def log(self, max_count=10, since_version=None):
                return ""

            async def is_ancestor(self, version_a, version_b):
                return False

            async def copy(self, name=None, from_version=None):
                return self

            async def is_dirty(self):
                return False

        workspace = _TestWorkspace()
        workspace.set_access(accesses=[
            AccessRule(access_type=AccessType.WRITE, pattern="**"),
            AccessRule(access_type=AccessType.READ, pattern="_vero/**"),
        ])

        session = Session(session_id="test", project_path=tmp_path, workspace=workspace)

        # Bind FileWrite — gets sandbox from workspace
        fw = FileWrite()
        fw.bind(session)

        assert fw.sandbox is not None
        assert fw.workspace is workspace

        # Workspace access checks work
        with pytest.raises(AccessDeniedError):
            fw.workspace.validate_write("_vero/test.txt")

        # Regular paths are writable
        fw.workspace.validate_write("src/main.py")

    @pytest.mark.asyncio
    async def test_agent_state_roundtrip(self):
        """MockAgent: step → serialize_state → new agent → deserialize_state → verify."""
        session = Session(session_id="test", project_path=Path("."))
        agent = MockAgent()
        agent.init(session)

        await agent.step("first prompt", max_turns=5)

        # Serialize state
        state = agent.serialize_state()
        trace = agent.serialize_trace()

        # Both should return the trace (MockAgent's state == trace)
        assert state == trace
        assert len(state) == 4

        # New agent, restore state
        agent2 = MockAgent()
        agent2.init(session)
        agent2.deserialize_state(state)

        assert agent2.state == state
