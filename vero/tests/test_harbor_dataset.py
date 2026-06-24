"""Tests for vero.harbor.dataset — partition compile + local task enumeration."""

import pytest

from vero.harbor.dataset import (
    build_harbor_dataset,
    enumerate_local_task_names,
    validate_partition,
)


def _make_task_dir(root, name):
    d = root / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text("[task]\nname='x'\n")
    return d


class TestBuildDataset:
    def test_partition_to_datasetdict(self):
        ds = build_harbor_dataset({"train": ["a", "b"], "test": ["c"]})
        assert set(ds.keys()) == {"train", "test"}
        assert ds["train"]["task_name"] == ["a", "b"]
        assert ds["test"]["task_name"] == ["c"]

    def test_empty_partition_raises(self):
        with pytest.raises(ValueError):
            build_harbor_dataset({})


class TestEnumerateLocal:
    def test_dataset_dir_of_tasks(self, tmp_path):
        _make_task_dir(tmp_path, "task_b")
        _make_task_dir(tmp_path, "task_a")
        (tmp_path / "not_a_task").mkdir()  # no task.toml -> excluded
        assert enumerate_local_task_names(tmp_path) == ["task_a", "task_b"]

    def test_single_task_dir(self, tmp_path):
        d = _make_task_dir(tmp_path, "solo")
        assert enumerate_local_task_names(d) == ["solo"]


class TestValidatePartition:
    def test_ok_when_subset(self):
        validate_partition({"train": ["a"], "test": ["b"]}, ["a", "b", "c"])

    def test_unknown_names_raise(self):
        with pytest.raises(ValueError, match="not found"):
            validate_partition({"test": ["a", "zzz"]}, ["a", "b"])
