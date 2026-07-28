"""Tests for the cache module."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from vero.core.db.dataset import DatasetSample
from vero.core.db.result import SampleResult
from vero.core.sessions import (
    FileNotFoundInCacheError,
    clear_result_cache,
    get_experiment_dir,
    initialize_result_store,
    load_all_sample_results,
    load_json_from_cache,
    load_sample_result,
    save_json_to_cache,
    save_sample_result,
)


class DummyModel(BaseModel):
    """A simple model for testing."""

    name: str
    value: int


@pytest.fixture
def sessions_dir(tmp_path: Path):
    """Fixture that provides a temp sessions directory."""
    sd = tmp_path / "sessions"
    sd.mkdir()
    return sd


SESSION_ID = "test-session"
RESULT_ID = "test-result-id"


class TestGetExperimentDir:
    """Tests for get_experiment_dir."""

    def test_returns_correct_path(self, sessions_dir: Path):
        """Test that experiment dir is under sessions/{session_id}/experiments/{result_id}."""
        path = get_experiment_dir(sessions_dir, SESSION_ID, RESULT_ID)
        assert path == sessions_dir / SESSION_ID / "experiments" / RESULT_ID


class TestInitializeResultStore:
    """Tests for initialize_result_store."""

    def test_creates_directory(self, sessions_dir: Path):
        """Test that directory is created if it doesn't exist."""
        result_dir = initialize_result_store(sessions_dir, SESSION_ID, RESULT_ID)
        assert result_dir.exists()
        assert result_dir.is_dir()

    def test_returns_existing_directory(self, sessions_dir: Path):
        """Test that existing directory is returned without error."""
        expected = get_experiment_dir(sessions_dir, SESSION_ID, RESULT_ID)
        expected.mkdir(parents=True)
        (expected / "existing_file.txt").touch()

        result_dir = initialize_result_store(sessions_dir, SESSION_ID, RESULT_ID)
        assert result_dir == expected
        assert (result_dir / "existing_file.txt").exists()


class TestSaveAndLoadJson:
    """Tests for save_json_to_cache and load_json_from_cache."""

    def test_save_and_load_dict(self, sessions_dir: Path):
        """Test saving and loading a dictionary."""
        data = {"key": "value", "number": 42}
        save_json_to_cache(sessions_dir, SESSION_ID, RESULT_ID, basename="data.json", data=data)

        loaded = load_json_from_cache(sessions_dir, SESSION_ID, RESULT_ID, basename="data.json")
        assert loaded == data

    def test_save_and_load_model(self, sessions_dir: Path):
        """Test saving and loading a Pydantic model."""
        model = DummyModel(name="test", value=123)
        save_json_to_cache(sessions_dir, SESSION_ID, RESULT_ID, basename="model.json", data=model)

        # Load as dict
        loaded_dict = load_json_from_cache(sessions_dir, SESSION_ID, RESULT_ID, basename="model.json")
        assert loaded_dict == {"name": "test", "value": 123}

        # Load as model
        loaded_model = load_json_from_cache(
            sessions_dir, SESSION_ID, RESULT_ID, basename="model.json", model=DummyModel
        )
        assert loaded_model == model

    def test_load_nonexistent_raises(self, sessions_dir: Path):
        """Test that loading nonexistent file raises FileNotFoundInCacheError."""
        with pytest.raises(FileNotFoundInCacheError):
            load_json_from_cache(sessions_dir, SESSION_ID, "nonexistent", basename="data.json")


class TestClearResultCache:
    """Tests for clear_result_cache."""

    def test_clears_samples_directory(self, sessions_dir: Path):
        """Test that samples directory is cleared."""
        experiment_dir = get_experiment_dir(sessions_dir, SESSION_ID, RESULT_ID)
        samples_dir = experiment_dir / "samples"
        samples_dir.mkdir(parents=True)
        (samples_dir / "0.json").write_text("{}")
        (samples_dir / "1.json").write_text("{}")

        clear_result_cache(sessions_dir, SESSION_ID, RESULT_ID)

        assert not samples_dir.exists()

    def test_clears_specified_basenames(self, sessions_dir: Path):
        """Test that specified basenames are cleared."""
        experiment_dir = get_experiment_dir(sessions_dir, SESSION_ID, RESULT_ID)
        experiment_dir.mkdir(parents=True)
        (experiment_dir / "report.json").write_text("{}")
        (experiment_dir / "other.json").write_text("{}")

        clear_result_cache(sessions_dir, SESSION_ID, RESULT_ID, result_basenames=["report.json"])

        assert not (experiment_dir / "report.json").exists()
        assert (experiment_dir / "other.json").exists()

    def test_handles_nonexistent_gracefully(self, sessions_dir: Path):
        """Test that clearing nonexistent directory doesn't raise."""
        clear_result_cache(sessions_dir, SESSION_ID, "nonexistent")


class TestSampleResultIO:
    """Tests for sample result save/load functions."""

    def _create_sample_result(self, sample_id: int, score: float) -> SampleResult:
        """Helper to create a sample result."""
        return SampleResult(
            dataset_sample=DatasetSample(
                dataset_id="test_dataset",
                split="test",
                sample_id=sample_id,
            ),
            score=score,
            feedback="Test feedback",
        )

    def test_save_and_load_single(self, sessions_dir: Path):
        """Test saving and loading a single sample result."""
        result = self._create_sample_result(0, 0.95)

        save_sample_result(sessions_dir, SESSION_ID, RESULT_ID, sample_id=0, result=result)
        loaded = load_sample_result(sessions_dir, SESSION_ID, RESULT_ID, sample_id=0)

        assert loaded is not None
        assert loaded.score == 0.95
        assert loaded.dataset_sample.sample_id == 0

    def test_load_nonexistent_returns_none(self, sessions_dir: Path):
        """Test that loading nonexistent sample returns None."""
        loaded = load_sample_result(sessions_dir, SESSION_ID, RESULT_ID, sample_id=999)
        assert loaded is None

    def test_load_all_sample_results(self, sessions_dir: Path):
        """Test loading all sample results."""
        results = {
            0: self._create_sample_result(0, 0.9),
            1: self._create_sample_result(1, 0.8),
            2: self._create_sample_result(2, 0.7),
        }

        for sample_id, result in results.items():
            save_sample_result(
                sessions_dir, SESSION_ID, RESULT_ID, sample_id=sample_id, result=result
            )

        loaded = load_all_sample_results(sessions_dir, SESSION_ID, RESULT_ID)

        assert len(loaded) == 3
        assert loaded[0].score == 0.9
        assert loaded[1].score == 0.8
        assert loaded[2].score == 0.7

    def test_load_all_empty_returns_empty_dict(self, sessions_dir: Path):
        """Test that loading from empty/nonexistent directory returns empty dict."""
        loaded = load_all_sample_results(sessions_dir, SESSION_ID, "nonexistent")
        assert loaded == {}


class TestConcurrentResultIsolation:
    """Tests verifying that concurrent evaluations get isolated storage."""

    def test_different_result_ids_are_isolated(self, sessions_dir: Path):
        """Test that different result_ids create separate storage locations."""
        result1 = SampleResult(
            dataset_sample=DatasetSample(dataset_id="d", split="test", sample_id=0),
            score=0.9,
        )
        result2 = SampleResult(
            dataset_sample=DatasetSample(dataset_id="d", split="test", sample_id=0),
            score=0.5,
        )

        save_sample_result(sessions_dir, SESSION_ID, "result_id_1", sample_id=0, result=result1)
        save_sample_result(sessions_dir, SESSION_ID, "result_id_2", sample_id=0, result=result2)

        loaded1 = load_sample_result(sessions_dir, SESSION_ID, "result_id_1", sample_id=0)
        loaded2 = load_sample_result(sessions_dir, SESSION_ID, "result_id_2", sample_id=0)

        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1.score == 0.9
        assert loaded2.score == 0.5

    def test_clearing_one_result_doesnt_affect_other(self, sessions_dir: Path):
        """Test that clearing one result doesn't affect another."""
        result = SampleResult(
            dataset_sample=DatasetSample(dataset_id="d", split="test", sample_id=0),
            score=0.9,
        )

        save_sample_result(sessions_dir, SESSION_ID, "result_id_1", sample_id=0, result=result)
        save_sample_result(sessions_dir, SESSION_ID, "result_id_2", sample_id=0, result=result)

        clear_result_cache(sessions_dir, SESSION_ID, "result_id_1")

        assert load_sample_result(sessions_dir, SESSION_ID, "result_id_1", sample_id=0) is None
        assert load_sample_result(sessions_dir, SESSION_ID, "result_id_2", sample_id=0) is not None
