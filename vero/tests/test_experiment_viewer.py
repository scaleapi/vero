"""Tests for the experiment viewer tool."""

from pathlib import Path

import pytest
from vero.core.dataset import DefaultSplitNames
from vero.core.db.database import ExperimentDatabase
from vero.tools.experiment_viewer import ExperimentViewer


@pytest.fixture
def experiment_db_path(resources_path: Path) -> Path:
    """Path to the experiment database JSON file."""
    return resources_path / "experiment-db-1.json"


@pytest.fixture
def experiment_viewer(experiment_db_path: Path) -> ExperimentViewer:
    """Create an ExperimentViewer instance with test data."""
    return ExperimentViewer.load_from_file(experiment_db_path)


class TestExperimentViewerInit:
    """Tests for ExperimentViewer initialization."""

    def test_load_from_file(self, experiment_db_path: Path):
        """Test loading an ExperimentViewer from a file."""
        viewer = ExperimentViewer.load_from_file(experiment_db_path)
        assert viewer is not None
        assert viewer.db is not None
        assert len(viewer.experiments()) > 0

    def test_load_from_nonexistent_file(self):
        """Test loading from a non-existent file raises a FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ExperimentViewer.load_from_file("/nonexistent/path.json")

    def test_exclude_splits(self, experiment_db_path: Path):
        """Test that exclude_splits filters experiments correctly."""
        viewer = ExperimentViewer(
            db=ExperimentDatabase.load_from_file(experiment_db_path),
            exclude_splits=[DefaultSplitNames.test],
        )
        all_experiments = viewer.db.get_experiments()
        filtered_experiments = viewer.experiments()

        # Check that test split is excluded
        for exp in filtered_experiments:
            assert exp.run.dataset_subset.split != DefaultSplitNames.test

        # Check that we have fewer experiments after filtering
        assert len(filtered_experiments) <= len(all_experiments)


class TestExperimentTableMetadata:
    """Tests for experiment_table_metadata method."""

    def test_experiment_table_metadata(self, experiment_viewer: ExperimentViewer):
        """Test getting experiment table metadata."""
        metadata = experiment_viewer.get_experiment_table_metadata()

        assert isinstance(metadata, str)
        assert "rows" in metadata
        assert "columns" in metadata
        assert "column names" in metadata
        assert "experiment_idx" in metadata or "id" in metadata


class TestExperimentIdMethods:
    """Tests for experiment ID methods."""

    def test_get_experiment_by_id(self, experiment_viewer: ExperimentViewer):
        """Test getting experiment by ID."""
        experiments = experiment_viewer.experiments(splits=[DefaultSplitNames.train])
        experiment_id = experiments[0].id
        # Should not raise
        experiment = experiment_viewer._get_experiment(experiment_id)
        assert experiment.id == experiment_id

    def test_get_experiment_invalid_id(self, experiment_viewer: ExperimentViewer):
        """Test getting experiment with invalid ID raises error."""
        with pytest.raises(KeyError):
            experiment_viewer._get_experiment("invalid_id")


class TestViewExperimentTable:
    """Tests for view_experiment_table method."""

    def test_view_experiment_table_default(self, experiment_viewer: ExperimentViewer):
        """Test viewing experiment table with default parameters."""
        result = experiment_viewer.view_experiment_table(split=DefaultSplitNames.train)
        assert isinstance(result, str)
        assert "experiment" in result.lower()
        assert "viewing" in result.lower()

    def test_view_experiment_table_with_num_rows(self, experiment_viewer: ExperimentViewer):
        """Test viewing experiment table with specified number of rows."""
        result = experiment_viewer.view_experiment_table(num_rows=2, split=DefaultSplitNames.train)
        assert "Viewing 2 experiment" in result

    def test_view_experiment_table_with_offset(self, experiment_viewer: ExperimentViewer):
        """Test viewing experiment table with row offset."""
        result = experiment_viewer.view_experiment_table(
            num_rows=2, row_offset_idx=1, split=DefaultSplitNames.train
        )
        assert "starting at row 1" in result

    def test_view_experiment_table_all_rows(self, experiment_viewer: ExperimentViewer):
        """Test viewing all rows in experiment table."""
        result = experiment_viewer.view_experiment_table(num_rows=None, split=DefaultSplitNames.train)
        assert "experiment" in result

    def test_view_experiment_table_with_columns(self, experiment_viewer: ExperimentViewer):
        """Test viewing experiment table with specific columns."""
        # Get all column names first
        df = experiment_viewer.df()
        if len(df.columns) >= 2:
            columns = [df.columns[0], df.columns[1]]
            result = experiment_viewer.view_experiment_table(
                columns=columns, num_rows=1, split=DefaultSplitNames.train
            )
            assert isinstance(result, str)

    def test_view_experiment_table_sort_ascending(self, experiment_viewer: ExperimentViewer):
        """Test sorting experiment table in ascending order."""
        df = experiment_viewer.df()
        if len(df.columns) > 0:
            sort_column = df.columns[0]
            result = experiment_viewer.view_experiment_table(
                sort_values_by=sort_column, ascending=True, num_rows=2, split=DefaultSplitNames.train
            )
            assert isinstance(result, str)

    def test_view_experiment_table_sort_descending(self, experiment_viewer: ExperimentViewer):
        """Test sorting experiment table in descending order."""
        df = experiment_viewer.df()
        if len(df.columns) > 0:
            sort_column = df.columns[0]
            result = experiment_viewer.view_experiment_table(
                sort_values_by=sort_column, ascending=False, num_rows=2, split=DefaultSplitNames.train
            )
            assert isinstance(result, str)

    def test_view_experiment_table_csv_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing experiment table in CSV format."""
        result = experiment_viewer.view_experiment_table(
            num_rows=2, format="csv", split=DefaultSplitNames.train
        )
        assert "```csv" in result
        assert "```" in result

    def test_view_experiment_table_json_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing experiment table in JSON format."""
        result = experiment_viewer.view_experiment_table(
            num_rows=2, format="json", split=DefaultSplitNames.train
        )
        assert "```json" in result
        assert "```" in result

    def test_view_experiment_table_yaml_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing experiment table in YAML format."""
        result = experiment_viewer.view_experiment_table(
            num_rows=2, format="yaml", split=DefaultSplitNames.train
        )
        assert "```yaml" in result
        assert "```" in result

    def test_view_experiment_table_kv_markdown_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing experiment table in kv_markdown format."""
        result = experiment_viewer.view_experiment_table(
            num_rows=2, format="kv_markdown", split=DefaultSplitNames.train
        )
        assert "Experiment" in result
        assert isinstance(result, str)


class TestViewResultsTable:
    """Tests for view_results_table method."""

    def test_view_results_table_default(self, experiment_viewer: ExperimentViewer):
        """Test viewing results table with default parameters."""
        experiment_id = experiment_viewer.experiments(splits=[DefaultSplitNames.train])[0].id
        result = experiment_viewer.view_sample_results_table(experiment_id=experiment_id)
        assert isinstance(result, str)

    def test_view_results_table_with_num_rows(self, experiment_viewer: ExperimentViewer):
        """Test viewing results table with specified number of rows."""
        experiment_id = experiment_viewer.experiments(splits=[DefaultSplitNames.train])[0].id
        result = experiment_viewer.view_sample_results_table(
            experiment_id=experiment_id, num_rows=2
        )
        assert isinstance(result, str)

    def test_view_results_table_with_offset(self, experiment_viewer: ExperimentViewer):
        """Test viewing results table with row offset."""
        experiment_id = experiment_viewer.experiments(splits=[DefaultSplitNames.train])[0].id
        result = experiment_viewer.view_sample_results_table(
            experiment_id=experiment_id, num_rows=2, row_offset_idx=1
        )
        assert "starting at row 1" in result

    def test_view_results_table_all_rows(self, experiment_viewer: ExperimentViewer):
        """Test viewing all rows in results table."""
        experiment_id = experiment_viewer.experiments(splits=[DefaultSplitNames.train])[0].id
        result = experiment_viewer.view_sample_results_table(
            experiment_id=experiment_id, num_rows=None
        )
        assert isinstance(result, str)

    def test_view_results_table_with_columns(self, experiment_viewer: ExperimentViewer):
        """Test viewing results table with specific columns."""
        experiment = experiment_viewer.experiments()[0]
        df = experiment.result.sample_results_df(exclude=["execution_trace"])
        if len(df.columns) >= 2:
            columns = [df.columns[0], df.columns[1]]
            result = experiment_viewer.view_sample_results_table(
                experiment_id=experiment.id, columns=columns, num_rows=1
            )
            assert isinstance(result, str)

    def test_view_results_table_sort_ascending(self, experiment_viewer: ExperimentViewer):
        """Test sorting results table in ascending order."""
        experiment = experiment_viewer.experiments()[0]
        df = experiment.result.sample_results_df(exclude=["execution_trace"])
        if len(df.columns) > 0 and "score" in df.columns:
            result = experiment_viewer.view_sample_results_table(
                experiment_id=experiment.id, sort_values_by="score", ascending=True, num_rows=2
            )
            assert isinstance(result, str)

    def test_view_results_table_sort_descending(self, experiment_viewer: ExperimentViewer):
        """Test sorting results table in descending order."""
        experiment = experiment_viewer.experiments()[0]
        df = experiment.result.sample_results_df(exclude=["execution_trace"])
        if len(df.columns) > 0 and "score" in df.columns:
            result = experiment_viewer.view_sample_results_table(
                experiment_id=experiment.id, sort_values_by="score", ascending=False, num_rows=2
            )
            assert isinstance(result, str)

    def test_view_results_table_csv_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing results table in CSV format."""
        experiment_id = experiment_viewer.experiments(splits=[DefaultSplitNames.train])[0].id
        result = experiment_viewer.view_sample_results_table(
            experiment_id=experiment_id, num_rows=2, format="csv"
        )
        assert "```csv" in result
        assert "```" in result

    def test_view_results_table_json_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing results table in JSON format."""
        experiment_id = experiment_viewer.experiments(splits=[DefaultSplitNames.train])[0].id
        result = experiment_viewer.view_sample_results_table(
            experiment_id=experiment_id, num_rows=2, format="json"
        )
        assert "```json" in result
        assert "```" in result

    def test_view_results_table_yaml_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing results table in YAML format."""
        experiment_id = experiment_viewer.experiments(splits=[DefaultSplitNames.train])[0].id
        result = experiment_viewer.view_sample_results_table(
            experiment_id=experiment_id, num_rows=2, format="yaml"
        )
        assert "```yaml" in result
        assert "```" in result

    def test_view_results_table_kv_markdown_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing results table in kv_markdown format."""
        experiment_id = experiment_viewer.experiments(splits=[DefaultSplitNames.train])[0].id
        result = experiment_viewer.view_sample_results_table(
            experiment_id=experiment_id, num_rows=2, format="kv_markdown"
        )
        assert "Sample Result" in result
        assert isinstance(result, str)

    def test_view_results_table_invalid_experiment_id(self, experiment_viewer: ExperimentViewer):
        """Test viewing results table with invalid experiment ID."""
        with pytest.raises(KeyError):
            experiment_viewer.view_sample_results_table(experiment_id="invalid_id")


class TestViewResult:
    """Tests for view_result method."""

    def _get_first_experiment_and_sample(
        self, experiment_viewer: ExperimentViewer
    ) -> tuple[str, int]:
        """Get the first experiment_id and sample_id from the first experiment."""
        experiment = experiment_viewer.experiments()[0]
        return experiment.id, experiment.result.sample_ids[0]

    def test_view_result_default(self, experiment_viewer: ExperimentViewer):
        """Test viewing a single result with default parameters."""
        experiment_id, sample_id = self._get_first_experiment_and_sample(experiment_viewer)
        result = experiment_viewer.view_sample_result_trace(
            experiment_id=experiment_id, sample_id=sample_id
        )
        assert isinstance(result, str)
        assert f"sample_id={sample_id}" in result
        assert f"experiment '{experiment_id}'" in result

    def test_view_result_json_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing result in JSON format."""
        experiment_id, sample_id = self._get_first_experiment_and_sample(experiment_viewer)
        result = experiment_viewer.view_sample_result_trace(
            experiment_id=experiment_id, sample_id=sample_id, format="json"
        )
        assert "```json" in result
        assert isinstance(result, str)

    def test_view_result_yaml_format(self, experiment_viewer: ExperimentViewer):
        """Test viewing result in YAML format."""
        experiment_id, sample_id = self._get_first_experiment_and_sample(experiment_viewer)
        result = experiment_viewer.view_sample_result_trace(
            experiment_id=experiment_id, sample_id=sample_id, format="yaml"
        )
        assert "```yaml" in result
        assert isinstance(result, str)

    def test_view_result_with_num_spans(self, experiment_viewer: ExperimentViewer):
        """Test viewing result with specific number of spans."""
        experiment_id, sample_id = self._get_first_experiment_and_sample(experiment_viewer)
        result = experiment_viewer.view_sample_result_trace(
            experiment_id=experiment_id, sample_id=sample_id, num_spans=2
        )
        assert isinstance(result, str)
        assert f"sample_id={sample_id}" in result

    def test_view_result_with_span_offset(self, experiment_viewer: ExperimentViewer):
        """Test viewing result with span offset."""
        experiment_id, sample_id = self._get_first_experiment_and_sample(experiment_viewer)
        result = experiment_viewer.view_sample_result_trace(
            experiment_id=experiment_id, sample_id=sample_id, start_offset=1
        )
        assert isinstance(result, str)
        assert "spans 1" in result

    def test_view_result_with_num_spans_and_offset(self, experiment_viewer: ExperimentViewer):
        """Test viewing result with both num_spans and span_offset."""
        experiment_id, sample_id = self._get_first_experiment_and_sample(experiment_viewer)
        result = experiment_viewer.view_sample_result_trace(
            experiment_id=experiment_id, sample_id=sample_id, num_spans=3, start_offset=1
        )
        assert isinstance(result, str)
        assert f"sample_id={sample_id}" in result

    def test_view_result_invalid_experiment_id(self, experiment_viewer: ExperimentViewer):
        """Test viewing result with invalid experiment ID."""
        with pytest.raises(KeyError):
            experiment_viewer.view_sample_result_trace(experiment_id="invalid_id", sample_id=0)

    def test_view_result_invalid_sample_id(self, experiment_viewer: ExperimentViewer):
        """Test viewing result with invalid sample_id."""
        experiment_id = experiment_viewer.experiments()[0].id
        with pytest.raises(KeyError):
            experiment_viewer.view_sample_result_trace(experiment_id=experiment_id, sample_id=99999)


class TestDataFrameProperties:
    """Tests for DataFrame properties."""

    def test_experiments_property(self, experiment_viewer: ExperimentViewer):
        """Test experiments property returns list of experiments."""
        experiments = experiment_viewer.experiments()
        assert isinstance(experiments, list)
        assert len(experiments) > 0

    def test_df_property(self, experiment_viewer: ExperimentViewer):
        """Test df property returns a DataFrame."""
        df = experiment_viewer.df()
        assert df is not None
        assert len(df) > 0
        assert len(df.columns) > 0
