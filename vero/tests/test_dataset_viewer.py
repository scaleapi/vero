"""Tests for the dataset viewer tool."""

from __future__ import annotations

import json

import pytest
from datasets import Dataset, DatasetDict
from vero.core.dataset import (
    DefaultSplitNames,
    default_split_accesses,
)
from vero.core.dataset.store import save_dataset
from vero.policy import Session
from vero.tools.dataset_viewer import DatasetViewer


@pytest.fixture
def mock_dataset():
    """Create a test dataset."""
    train = Dataset.from_dict({
        "id": [1, 2, 3, 4, 5],
        "text": ["sample 1", "sample 2", "sample 3", "sample 4", "sample 5"],
        "label": [0, 1, 0, 1, 0],
        "score": [0.5, 0.8, 0.3, 0.9, 0.6],
    })
    val = Dataset.from_dict({
        "id": [6, 7],
        "text": ["val 1", "val 2"],
        "label": [1, 0],
        "score": [0.7, 0.4],
    })
    test = Dataset.from_dict({
        "id": [8, 9],
        "text": ["test 1", "test 2"],
        "label": [0, 1],
        "score": [0.2, 0.95],
    })
    return DatasetDict({
        DefaultSplitNames.train: train,
        DefaultSplitNames.validation: val,
        DefaultSplitNames.test: test,
    })


@pytest.fixture
def session_with_dataset(mock_dataset, tmp_path):
    """Create a session with a dataset saved to the store."""
    vero_home = tmp_path / "vero_home"
    sessions_dir = vero_home / "sessions"
    dataset_cache = vero_home / "datasets"
    sessions_dir.mkdir(parents=True)
    dataset_cache.mkdir(parents=True)

    session_id = "test-session"
    (sessions_dir / session_id).mkdir(parents=True)
    save_dataset(sessions_dir, dataset_cache, session_id, "test_dataset", mock_dataset)

    return Session(
        session_id=session_id,
        project_path=tmp_path,
        vero_home=vero_home,
        dataset_id="test_dataset",
        split_accesses=list(default_split_accesses),
    )


@pytest.fixture
def dataset_viewer(session_with_dataset):
    """Create a DatasetViewer bound to a session."""
    viewer = DatasetViewer()
    viewer.bind(session_with_dataset)
    return viewer


class TestDatasetViewerInit:
    def test_bind_sets_session(self, session_with_dataset):
        viewer = DatasetViewer()
        viewer.bind(session_with_dataset)
        assert viewer._session_id == "test-session"
        assert viewer._dataset_id == "test_dataset"

    def test_bind_sets_exclude_splits(self, session_with_dataset):
        viewer = DatasetViewer()
        viewer.bind(session_with_dataset)
        assert "test" in viewer.exclude_splits
        assert "validation" in viewer.exclude_splits


class TestDatasetInfo:
    def test_get_dataset_info(self, dataset_viewer):
        result = dataset_viewer.get_dataset_info()
        info = json.loads(result.strip("```json\n").strip("\n```"))
        assert len(info) == 1
        assert info[0]["id"] == "test_dataset"
        assert info[0]["splits"]["train"] == 5
        assert info[0]["splits"]["validation"] == 2
        assert info[0]["splits"]["test"] == 2


class TestGetDatasetStats:
    def test_stats_returns_json(self, dataset_viewer):
        result = dataset_viewer.get_dataset_stats("test_dataset", "train")
        assert "json" in result

    def test_stats_non_viewable_split_raises(self, dataset_viewer):
        with pytest.raises(ValueError, match="cannot view"):
            dataset_viewer.get_dataset_stats("test_dataset", "test")


class TestViewSamples:
    def test_view_default_samples(self, dataset_viewer):
        result = dataset_viewer.view_samples("test_dataset", "train")
        samples = json.loads(result.strip("```json\n").strip("\n```"))
        assert len(samples) == 5  # all 5 train samples (default is first 5)

    def test_view_specific_samples(self, dataset_viewer):
        result = dataset_viewer.view_samples("test_dataset", "train", sample_ids=[0, 2])
        samples = json.loads(result.strip("```json\n").strip("\n```"))
        assert len(samples) == 2
        assert samples[0]["id"] == 1
        assert samples[1]["id"] == 3

    def test_view_range(self, dataset_viewer):
        result = dataset_viewer.view_samples(
            "test_dataset", "train", sample_id_range_start=1, sample_id_range_end=3
        )
        samples = json.loads(result.strip("```json\n").strip("\n```"))
        assert len(samples) == 2

    def test_view_non_viewable_split_raises(self, dataset_viewer):
        with pytest.raises(ValueError, match="cannot view"):
            dataset_viewer.view_samples("test_dataset", "test")

    def test_view_yaml_format(self, dataset_viewer):
        result = dataset_viewer.view_samples("test_dataset", "train", format="yaml")
        assert "yaml" in result

    def test_mutual_exclusivity(self, dataset_viewer):
        with pytest.raises(ValueError, match="Cannot specify both"):
            dataset_viewer.view_samples(
                "test_dataset", "train", sample_ids=[0], sample_id_range_start=0
            )
