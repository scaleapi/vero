from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from vero.core.dataset import DatasetInfo, get_non_viewable_splits
from vero.tools.utils import is_tool
from vero.utils import df_to_format


@dataclass
class DatasetViewer:
    """View samples and metadata of datasets."""

    exclude_tools: list[str] = field(default_factory=list)

    # Runtime fields — set during bind()
    _session_id: str | None = None
    _dataset_id: str | None = None
    _sessions_dir: Path | None = None
    _dataset_cache: Path | None = None
    exclude_splits: list[str] = field(default_factory=list)

    def bind(self, session) -> None:
        self._session_id = session.session_id
        self._dataset_id = session.dataset_id
        if session.vero_home:
            self._sessions_dir = session.vero_home / "sessions"
            self._dataset_cache = session.vero_home / "datasets"
        if session.split_accesses:
            self.exclude_splits = get_non_viewable_splits(session.split_accesses)

    def _load_dataset(self, dataset_id: str | None = None):
        """Load a DatasetDict from the store."""
        from vero.core.dataset.store import load_dataset

        ds_id = dataset_id or self._dataset_id
        return load_dataset(self._sessions_dir, self._dataset_cache, self._session_id, ds_id)

    def _validate_dataset_and_split(self, dataset_id: str, split: str) -> None:
        """Validate that a dataset and split exist and are viewable."""
        dataset_dict = self._load_dataset(dataset_id)

        if split not in dataset_dict:
            raise KeyError(f"Split {split} not found for dataset {dataset_id}.")

        viewable_splits = [s for s in dataset_dict.keys() if s not in self.exclude_splits]

        if split in self.exclude_splits:
            raise ValueError(
                f"You cannot view the data in {split} for dataset {dataset_id}. Viewable splits: {viewable_splits}"
            )

    @is_tool
    def get_dataset_info(self, dataset_ids: list[str] | None = None) -> str:
        """Get metadata about datasets, including the number of samples in each split.

        Args:
            dataset_ids: List of dataset ids. If None, uses the default dataset.

        Returns:
            JSON string containing the metadata
        """
        if dataset_ids is None:
            dataset_ids = [self._dataset_id]

        dataset_info = []
        for ds_id in dataset_ids:
            dataset = self._load_dataset(ds_id)
            info = DatasetInfo(
                id=ds_id,
                splits={split: len(dataset[split]) for split in dataset},
                features={split: list(dataset[split].features) for split in dataset},
            )
            dataset_info.append(info.model_dump())

        return f"```json\n{json.dumps(dataset_info, indent=2)}\n```"

    @is_tool
    def get_dataset_stats(self, dataset_id: str, split: str) -> str:
        """Get statistics about a dataset split.

        Args:
            dataset_id: The id of the dataset
            split: The split to get statistics about

        Returns:
            JSON string containing the statistics
        """
        self._validate_dataset_and_split(dataset_id, split)
        dataset = self._load_dataset(dataset_id)[split]
        df = dataset.to_pandas()
        stats = df.describe(include="all")
        return f"```json\n{df_to_format(stats, 'json', indent=2)}\n```"

    @is_tool
    def view_samples(
        self,
        dataset_id: str,
        split: str,
        sample_ids: list[int] | None = None,
        sample_id_range_start: int | None = None,
        sample_id_range_end: int | None = None,
        columns: list[str] | None = None,
        format: Literal["json", "yaml"] = "json",
    ) -> str:
        """View samples from a dataset and split.

        Use either sample_ids for specific samples, or sample_id_range_start/end for a range.
        Defaults to first 5 samples if neither is provided.

        Args:
            dataset_id: The dataset to view
            split: The split to view
            sample_ids: Specific sample ids
            sample_id_range_start: Start of range
            sample_id_range_end: End of range
            columns: Columns to include
            format: Output format (json or yaml)

        Returns:
            Formatted string with the samples
        """
        if sample_ids is not None and (
            sample_id_range_start is not None or sample_id_range_end is not None
        ):
            raise ValueError(
                "Cannot specify both sample_ids and sample_id_range_start/end."
            )

        self._validate_dataset_and_split(dataset_id, split)
        dataset = self._load_dataset(dataset_id)[split]

        if columns:
            dataset = dataset.select_columns(columns)

        if sample_ids is not None:
            selected_ids = sample_ids
        elif sample_id_range_start is not None or sample_id_range_end is not None:
            start = sample_id_range_start if sample_id_range_start is not None else 0
            end = sample_id_range_end
            if start >= len(dataset):
                raise IndexError(f"Start index {start} is beyond the dataset size {len(dataset)}")
            end = end if end is not None else len(dataset)
            end = min(end, len(dataset))
            selected_ids = list(range(start, end))
        else:
            selected_ids = list(range(min(5, len(dataset))))

        dataset = dataset.select(selected_ids)
        samples = list(dataset)

        if format == "json":
            return f"```json\n{json.dumps(samples, indent=2)}\n```"
        else:
            return f"```yaml\n{yaml.dump(samples, indent=2)}\n```"
