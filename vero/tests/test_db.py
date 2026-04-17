"""Tests for the experiment database."""

from datetime import datetime

import pytest
from vero.core.dataset import DatasetInfo, DefaultSplitNames
from vero.core.db.candidate import Candidate
from vero.core.db.database import ExperimentDatabase
from vero.core.db.dataset import DatasetSample, DatasetSubset
from vero.core.db.result import ExperimentResult, ExperimentResultStatus, SampleResult
from vero.core.db.run import ExperimentRun


# Helper functions to create dummy data
def create_dummy_candidate(
    commit_hash: str, repo_name: str = "test_repo", parent_commit: str | None = None
) -> Candidate:
    """Create a dummy candidate."""
    return Candidate(
        commit=commit_hash,
        repo_name=repo_name,
        parent_commit=parent_commit,
        created_at=datetime.now(),
    )


def create_dummy_dataset_info(dataset_id: str = "test_dataset") -> DatasetInfo:
    """Create a dummy dataset info."""
    return DatasetInfo(
        id=dataset_id,
        splits={DefaultSplitNames.train: 100, DefaultSplitNames.test: 50},
        description="Test dataset",
    )


def create_dummy_dataset_subset(
    dataset_id: str = "test_dataset",
    split: str = DefaultSplitNames.train,
    sample_ids: list[int] | None = None,
) -> DatasetSubset:
    """Create a dummy dataset subset."""
    return DatasetSubset(dataset_id=dataset_id, split=split, sample_ids=sample_ids)


def create_dummy_run(candidate: Candidate, dataset_subset: DatasetSubset) -> ExperimentRun:
    """Create a dummy experiment run."""
    return ExperimentRun(candidate=candidate, dataset_subset=dataset_subset)


def create_dummy_sample_result(
    dataset_sample: DatasetSample, score: float | None = None, error: str | None = None
) -> SampleResult:
    """Create a dummy sample result."""
    return SampleResult(
        dataset_sample=dataset_sample,
        score=score,
        error=error,
        feedback="Test feedback" if score is not None else None,
    )


def create_dummy_result(run_id: str, scores: list[float | None]) -> ExperimentResult:
    """Create a dummy experiment result with multiple sample results.

    Args:
        run_id: The run ID for this result
        scores: List of scores. None values indicate errors.

    Returns:
        ExperimentResult with sample results
    """
    sample_results = {}
    for i, score in enumerate(scores):
        dataset_sample = DatasetSample(dataset_id="test_dataset", split="train", sample_id=i)
        error = "error" if score is None else None
        sample_results[i] = create_dummy_sample_result(dataset_sample, score, error)

    return ExperimentResult(
        run_id=run_id, status=ExperimentResultStatus.SUCCESS, sample_results=sample_results
    )


def create_database_with_scores(
    scores: list[list[float | None]] | dict[str, list[list[float | None]]],
    db_id: str = "test_db",
    dataset_id: str = "test_dataset",
    repo_name: str = "test_repo",
) -> ExperimentDatabase:
    """Create a database with experiments for each list of scores.

    Args:
        scores: Either:
                - List of score lists (will be assigned to 'train' split), or
                - Dictionary mapping split names to lists of score lists
                Each inner list contains scores for individual sample results.
                None values indicate errors.
        db_id: Database ID
        dataset_id: Dataset ID
        repo_name: Repository name for candidates

    Returns:
        ExperimentDatabase with all experiments added
    """
    # Normalize to dict format
    if isinstance(scores, list):
        scores_by_split = {DefaultSplitNames.train: scores}
    else:
        scores_by_split = scores

    db = ExperimentDatabase(id=db_id)

    candidate_idx = 0
    for split, score_lists in scores_by_split.items():
        dataset_subset = create_dummy_dataset_subset(dataset_id=dataset_id, split=split)

        for score_list in score_lists:
            candidate = create_dummy_candidate(f"commit{candidate_idx}", repo_name=repo_name)
            db.add_candidate(candidate)

            run = create_dummy_run(candidate, dataset_subset)
            db.add_run(run)

            # Create result (errors inferred from None scores)
            result = create_dummy_result(run.id, score_list)
            db.add_result(result)

            candidate_idx += 1

    return db


class TestExperimentDatabase:
    """Tests for the ExperimentDatabase class."""

    def test_add_and_get_candidate(self):
        """Test adding and retrieving candidates."""
        db = ExperimentDatabase(id="test_db")

        candidate = create_dummy_candidate("abc123def456")
        db.add_candidate(candidate)

        assert len(db.candidates) == 1
        assert db.get_candidate(candidate) == candidate
        assert db.get_candidate(candidate.id) == candidate
        assert db.get_candidate("nonexistent") is None

    def test_add_duplicate_candidate(self):
        """Test that adding duplicate candidates doesn't create duplicates."""
        db = ExperimentDatabase(id="test_db")

        candidate = create_dummy_candidate("abc123def456")
        db.add_candidate(candidate)
        db.add_candidate(candidate)

        assert len(db.candidates) == 1

    def test_add_and_get_run(self):
        """Test adding and retrieving runs."""
        db = ExperimentDatabase(id="test_db")

        candidate = create_dummy_candidate("abc123def456")
        dataset_subset = create_dummy_dataset_subset()
        run = create_dummy_run(candidate, dataset_subset)

        db.add_candidate(candidate)
        db.add_run(run)

        assert len(db.runs) == 1
        assert db.get_run(run) == run
        assert db.get_run(run.id) == run

    def test_add_result_requires_run(self):
        """Test that adding a result requires the corresponding run to exist."""
        db = ExperimentDatabase(id="test_db")

        result = create_dummy_result("nonexistent_run_id", [0.9, 0.8])

        with pytest.raises(ValueError, match="ExperimentRun nonexistent_run_id does not exist"):
            db.add_result(result)

    def test_add_and_get_result(self):
        """Test adding and retrieving results."""
        db = ExperimentDatabase(id="test_db")

        candidate = create_dummy_candidate("abc123def456")
        dataset_subset = create_dummy_dataset_subset()
        run = create_dummy_run(candidate, dataset_subset)

        db.add_candidate(candidate)
        db.add_run(run)

        result = create_dummy_result(run.id, [0.9, 0.8, 0.7])
        db.add_result(result)

        assert len(db.results) == 1
        assert db.get_result(result) == result
        assert db.get_result(result.id) == result

    def test_get_experiments(self):
        """Test getting experiments (run + result pairs)."""
        db = ExperimentDatabase(id="test_db")

        candidate = create_dummy_candidate("abc123def456")
        dataset_subset = create_dummy_dataset_subset()
        run = create_dummy_run(candidate, dataset_subset)

        db.add_candidate(candidate)
        db.add_run(run)

        result = create_dummy_result(run.id, [0.9, 0.8, 0.7])
        db.add_result(result)

        experiments = db.get_experiments()
        assert len(experiments) == 1
        assert experiments[0].run == run
        assert experiments[0].result == result

    def test_serialization(self):
        """Test database serialization and deserialization."""
        db = ExperimentDatabase(id="test_db")

        candidate = create_dummy_candidate("abc123def456")
        dataset_subset = create_dummy_dataset_subset()
        run = create_dummy_run(candidate, dataset_subset)

        db.add_candidate(candidate)
        db.add_run(run)

        result = create_dummy_result(run.id, [0.9, 0.8])
        db.add_result(result)

        # Serialize to dict
        serialized = db.serialize()
        assert serialized["id"] == "test_db"
        assert len(serialized["candidates"]) == 1
        assert len(serialized["runs"]) == 1
        assert len(serialized["results"]) == 1

        # Deserialize
        db2 = ExperimentDatabase.deserialize(serialized)
        assert db2.id == db.id
        assert len(db2.candidates) == len(db.candidates)
        assert len(db2.runs) == len(db.runs)
        assert len(db2.results) == len(db.results)

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        db = ExperimentDatabase(id="test_db")

        candidate = create_dummy_candidate("abc123def456")
        dataset_subset = create_dummy_dataset_subset()
        run = create_dummy_run(candidate, dataset_subset)

        db.add_candidate(candidate)
        db.add_run(run)

        result = create_dummy_result(run.id, [0.9, 0.8])
        db.add_result(result)

        # Convert to JSON
        json_str = db.to_json()
        assert isinstance(json_str, str)

        # Convert back from JSON
        db2 = ExperimentDatabase.from_json(json_str)
        assert db2.id == db.id
        assert len(db2.candidates) == len(db.candidates)
        assert len(db2.runs) == len(db.runs)
        assert len(db2.results) == len(db.results)
