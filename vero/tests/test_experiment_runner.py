"""Tests for ExperimentRunnerTool and SplitBudget."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vero.core.dataset import DatasetInfo
from vero.core.db.candidate import Candidate
from vero.core.db.database import Experiment
from vero.core.db.dataset import DatasetSubset
from vero.core.db.result import ExperimentResult, ExperimentResultStatus
from vero.core.db.run import ExperimentRun
from vero.exceptions import ExperimentBudgetExceeded, InvalidSplitError
from vero.tools.experiment_runner import ExperimentRunnerTool, SplitBudget
from vero.tools.utils import get_tools_from_class
from vero.tools.utils.openai_agents import tool_set_instance_to_oai_tools

# Patch _get_dataset_info on all ExperimentRunnerTool instances to avoid store dependency
_DEFAULT_DATASET_INFO = DatasetInfo(
    id="ds1", splits={"dev": 100, "test": 50}, features={"dev": [], "test": []}
)


@pytest.fixture(autouse=True)
def mock_dataset_info(monkeypatch):
    """Mock _get_dataset_info to avoid dataset store dependency in tests."""
    original = ExperimentRunnerTool._get_dataset_info

    def patched_get_dataset_info(self, dataset_id):
        return _DEFAULT_DATASET_INFO

    monkeypatch.setattr(
        ExperimentRunnerTool, "_get_dataset_info", patched_get_dataset_info
    )


def make_mock_result(run_id: str = "test_run") -> ExperimentResult:
    """Create a mock ExperimentResult for testing."""
    return ExperimentResult(
        run_id=run_id,
        status=ExperimentResultStatus.SUCCESS,
        sample_results={},
    )


def make_mock_experiment() -> Experiment:
    """Create a mock Experiment for testing."""
    run = ExperimentRun(
        candidate=Candidate(commit="abc123def456", repo_name="test_repo"),
        dataset_subset=DatasetSubset(split="dev", dataset_id="ds1"),
    )
    return Experiment(run=run, result=make_mock_result(run_id=run.id))


class TestSplitBudget:
    """Tests for SplitBudget dataclass."""

    def test_init_with_sample_budget(self):
        """Test initialization with sample budget."""
        budget = SplitBudget(
            split="dev", dataset_id="test_dataset", total_sample_budget=100
        )
        assert budget.remaining_sample_budget == 100
        assert budget.remaining_run_budget is None

    def test_init_with_run_budget(self):
        """Test initialization with run budget."""
        budget = SplitBudget(split="dev", dataset_id="test_dataset", total_run_budget=5)
        assert budget.remaining_run_budget == 5
        assert budget.remaining_sample_budget is None

    def test_init_with_both_budgets(self):
        """Test initialization with both budgets."""
        budget = SplitBudget(
            split="dev",
            dataset_id="test_dataset",
            total_sample_budget=100,
            total_run_budget=5,
        )
        assert budget.remaining_sample_budget == 100
        assert budget.remaining_run_budget == 5

    def test_init_requires_at_least_one_budget(self):
        """Test that at least one budget is required."""
        with pytest.raises(AssertionError):
            SplitBudget(split="dev", dataset_id="test_dataset")

    def test_has_run_budget_when_unlimited(self):
        """Test has_run_budget returns True when no limit set."""
        budget = SplitBudget(
            split="dev", dataset_id="test_dataset", total_sample_budget=100
        )
        assert budget.has_run_budget() is True

    def test_has_run_budget_when_remaining(self):
        """Test has_run_budget when runs remain."""
        budget = SplitBudget(split="dev", dataset_id="test_dataset", total_run_budget=2)
        assert budget.has_run_budget() is True

    def test_has_run_budget_when_exhausted(self):
        """Test has_run_budget when no runs remain."""
        budget = SplitBudget(split="dev", dataset_id="test_dataset", total_run_budget=1)
        budget.decrement_run_budget()
        assert budget.has_run_budget() is False

    def test_decrement_run_budget(self):
        """Test decrementing run budget."""
        budget = SplitBudget(split="dev", dataset_id="test_dataset", total_run_budget=3)
        budget.decrement_run_budget()
        assert budget.remaining_run_budget == 2
        budget.decrement_run_budget()
        assert budget.remaining_run_budget == 1

    def test_decrement_run_budget_when_unlimited(self):
        """Test decrementing run budget when unlimited does nothing."""
        budget = SplitBudget(
            split="dev", dataset_id="test_dataset", total_sample_budget=100
        )
        budget.decrement_run_budget()
        assert budget.remaining_run_budget is None

    def test_has_sample_budget_when_unlimited(self):
        """Test has_sample_budget returns True when no limit set."""
        budget = SplitBudget(split="dev", dataset_id="test_dataset", total_run_budget=5)
        assert budget.has_sample_budget(1000) is True

    def test_has_sample_budget_sufficient(self):
        """Test has_sample_budget when sufficient samples remain."""
        budget = SplitBudget(
            split="dev", dataset_id="test_dataset", total_sample_budget=100
        )
        assert budget.has_sample_budget(50) is True
        assert budget.has_sample_budget(100) is True

    def test_has_sample_budget_insufficient(self):
        """Test has_sample_budget when insufficient samples remain."""
        budget = SplitBudget(
            split="dev", dataset_id="test_dataset", total_sample_budget=100
        )
        assert budget.has_sample_budget(101) is False

    def test_decrement_sample_budget(self):
        """Test decrementing sample budget."""
        budget = SplitBudget(
            split="dev", dataset_id="test_dataset", total_sample_budget=100
        )
        budget.decrement_sample_budget(30)
        assert budget.remaining_sample_budget == 70
        budget.decrement_sample_budget(20)
        assert budget.remaining_sample_budget == 50

    def test_decrement_sample_budget_when_unlimited(self):
        """Test decrementing sample budget when unlimited does nothing."""
        budget = SplitBudget(split="dev", dataset_id="test_dataset", total_run_budget=5)
        budget.decrement_sample_budget(50)
        assert budget.remaining_sample_budget is None

    def test_exceeds_per_run_budget_when_unlimited(self):
        """Test exceeds_per_run_budget when no limit set."""
        budget = SplitBudget(
            split="dev", dataset_id="test_dataset", total_sample_budget=100
        )
        assert budget.exceeds_per_run_budget(1000) is False

    def test_exceeds_per_run_budget_within_limit(self):
        """Test exceeds_per_run_budget within limit."""
        budget = SplitBudget(
            split="dev",
            dataset_id="test_dataset",
            total_sample_budget=100,
            max_samples_per_run=50,
        )
        assert budget.exceeds_per_run_budget(50) is False
        assert budget.exceeds_per_run_budget(30) is False

    def test_exceeds_per_run_budget_over_limit(self):
        """Test exceeds_per_run_budget over limit."""
        budget = SplitBudget(
            split="dev",
            dataset_id="test_dataset",
            total_sample_budget=100,
            max_samples_per_run=50,
        )
        assert budget.exceeds_per_run_budget(51) is True
        assert budget.exceeds_per_run_budget(100) is True

    def test_repr(self):
        """Test string representation."""
        budget = SplitBudget(
            split="dev",
            dataset_id="test_dataset",
            total_sample_budget=100,
            total_run_budget=5,
        )
        repr_str = repr(budget)
        assert "dev" in repr_str
        assert "test_dataset" in repr_str
        assert "100" in repr_str
        assert "5" in repr_str


class TestExperimentRunnerToolInit:
    """Tests for ExperimentRunnerTool initialization."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        evaluator = MagicMock()
        evaluator.git_worktree = MagicMock()
        evaluator.session_id = "test-session"
        return evaluator

    def test_init_with_list_budget(self, mock_evaluator):
        """Test initialization converts list budget to dict."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100),
            SplitBudget(split="test", dataset_id="ds1", total_sample_budget=50),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        assert isinstance(tool._budget_map, dict)
        assert ("dev", "ds1") in tool._budget_map
        assert ("test", "ds1") in tool._budget_map

    def test_init_creates_budget_map(self, mock_evaluator):
        """Test initialization creates budget map from list."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        assert ("dev", "ds1") in tool._budget_map


class TestExperimentRunnerToolValidation:
    """Tests for validation methods."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator with dataset info."""
        evaluator = MagicMock()

        # Mock dataset manager
        dataset_info = MagicMock()
        dataset_info.splits = {"dev": 100, "test": 50}
        # dataset_info provided by mock_dataset_info fixture

        evaluator.session_id = "test-session"
        return evaluator

    @pytest.fixture
    def tool(self, mock_evaluator):
        """Create tool with standard budget."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100),
            SplitBudget(split="test", dataset_id="ds1", total_sample_budget=50),
        ]
        return ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

    def test_validate_split_access_valid(self, tool):
        """Test split access validation for valid split."""
        # Should not raise
        tool._validate_split_access("ds1", "dev")
        tool._validate_split_access("ds1", "test")

    def test_validate_split_access_invalid_split(self, tool):
        """Test split access validation for invalid split."""
        with pytest.raises(InvalidSplitError) as exc_info:
            tool._validate_split_access("ds1", "invalid")
        assert "invalid" in str(exc_info.value)

    def test_validate_split_access_invalid_dataset(self, tool):
        """Test split access validation for invalid dataset."""
        with pytest.raises(InvalidSplitError) as exc_info:
            tool._validate_split_access("invalid_ds", "dev")
        assert "invalid_ds" in str(exc_info.value)

    def test_validate_and_count_samples_full_split(self, tool):
        """Test counting samples for full split."""
        count = tool._validate_and_count_samples("ds1", "dev", sample_ids=None)
        assert count == 100

    def test_validate_and_count_samples_subset(self, tool):
        """Test counting samples for subset."""
        count = tool._validate_and_count_samples("ds1", "dev", sample_ids=[0, 1, 2])
        assert count == 3

    def test_validate_and_count_samples_invalid_ids(self, tool):
        """Test validation fails for invalid sample IDs."""
        with pytest.raises(ValueError) as exc_info:
            tool._validate_and_count_samples("ds1", "dev", sample_ids=[0, 100, 150])
        assert "100" in str(exc_info.value)
        assert "150" in str(exc_info.value)

    def test_validate_and_count_samples_negative_ids(self, tool):
        """Test validation fails for negative sample IDs."""
        with pytest.raises(ValueError) as exc_info:
            tool._validate_and_count_samples("ds1", "dev", sample_ids=[-1, 0, 1])
        assert "-1" in str(exc_info.value)


class TestExperimentRunnerToolBudgetChecks:
    """Tests for budget checking methods."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        evaluator = MagicMock()
        dataset_info = MagicMock()
        dataset_info.splits = {"dev": 100, "test": 50}
        # dataset_info provided by mock_dataset_info fixture
        evaluator.session_id = "test-session"
        return evaluator

    def test_check_budget_valid(self, mock_evaluator):
        """Test budget check passes for valid request."""
        budgets = [
            SplitBudget(
                split="dev",
                dataset_id="ds1",
                total_sample_budget=100,
                total_run_budget=5,
            ),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        # Should not raise
        tool._check_budget("ds1", "dev", 50)

    def test_check_budget_no_runs_left(self, mock_evaluator):
        """Test budget check fails when no runs left."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_run_budget=1),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        # Exhaust the run budget
        tool._budget_map[("dev", "ds1")].decrement_run_budget()

        with pytest.raises(ExperimentBudgetExceeded) as exc_info:
            tool._check_budget("ds1", "dev", 10)
        assert "No runs left" in str(exc_info.value)

    def test_check_budget_insufficient_samples(self, mock_evaluator):
        """Test budget check fails when insufficient samples."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=30),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        with pytest.raises(ExperimentBudgetExceeded) as exc_info:
            tool._check_budget("ds1", "dev", 50)
        assert "50" in str(exc_info.value)
        assert "30" in str(exc_info.value)

    def test_check_budget_exceeds_per_run_limit(self, mock_evaluator):
        """Test budget check fails when exceeding per-run limit."""
        budgets = [
            SplitBudget(
                split="dev",
                dataset_id="ds1",
                total_sample_budget=100,
                max_samples_per_run=20,
            ),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        with pytest.raises(ExperimentBudgetExceeded) as exc_info:
            tool._check_budget("ds1", "dev", 30)
        assert "30" in str(exc_info.value)
        assert "20" in str(exc_info.value)

    def test_check_budget_invalid_split(self, mock_evaluator):
        """Test budget check fails for invalid split."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        with pytest.raises(InvalidSplitError):
            tool._check_budget("ds1", "invalid", 10)


class TestExperimentRunnerToolUpdateBudget:
    """Tests for budget update methods."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        evaluator = MagicMock()
        dataset_info = MagicMock()
        dataset_info.splits = {"dev": 100}
        # dataset_info provided by mock_dataset_info fixture
        evaluator.session_id = "test-session"
        return evaluator

    def test_update_budget_decrements_samples(self, mock_evaluator):
        """Test update budget decrements sample count."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        tool._update_budget("ds1", "dev", 30)

        assert tool._budget_map[("dev", "ds1")].remaining_sample_budget == 70

    def test_update_budget_decrements_runs(self, mock_evaluator):
        """Test update budget decrements run count."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_run_budget=5),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        tool._update_budget("ds1", "dev", 10)

        assert tool._budget_map[("dev", "ds1")].remaining_run_budget == 4

    def test_update_budget_returns_info(self, mock_evaluator):
        """Test update budget returns informative message."""
        budgets = [
            SplitBudget(
                split="dev",
                dataset_id="ds1",
                total_sample_budget=100,
                total_run_budget=5,
            ),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        info = tool._update_budget("ds1", "dev", 30)

        assert "30" in info
        assert "70" in info  # remaining samples
        assert "4" in info  # remaining runs


class TestExperimentRunnerToolGetSamples:
    """Tests for sample selection methods."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        evaluator = MagicMock()
        dataset_info = MagicMock()
        dataset_info.splits = {"dev": 100, "test": 50}
        # dataset_info provided by mock_dataset_info fixture
        evaluator.session_id = "test-session"
        return evaluator

    @pytest.fixture
    def tool(self, mock_evaluator):
        """Create tool with standard budget."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100),
        ]
        return ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

    def test_get_samples_from_split_subset(self, tool):
        """Test getting subset of samples."""
        samples = tool._get_samples_from_split("ds1", "dev", num_samples=10)
        assert samples == list(range(10))

    def test_get_samples_from_split_full(self, tool):
        """Test getting all samples returns None."""
        samples = tool._get_samples_from_split("ds1", "dev", num_samples=100)
        assert samples is None

    def test_get_samples_from_split_exceeds_size(self, tool):
        """Test requesting more samples than split size returns None."""
        samples = tool._get_samples_from_split("ds1", "dev", num_samples=200)
        assert samples is None


class TestExperimentRunnerToolEvaluate:
    """Tests for evaluate_commit method."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator with all required methods."""
        evaluator = MagicMock()

        # Dataset info
        dataset_info = MagicMock()
        dataset_info.splits = {"dev": 100, "test": 50}
        # dataset_info provided by mock_dataset_info fixture

        # Git worktree
        evaluator.git_worktree.repo_name = "test_repo"
        mock_commit = MagicMock()
        mock_commit.hexsha = "abc123def456"
        evaluator.git_worktree.repo.commit.return_value = mock_commit

        # Async run method - return proper ExperimentResult
        evaluator.evaluate = AsyncMock(return_value=make_mock_experiment())
        evaluator.session_id = "test-session"
        return evaluator

    @pytest.fixture
    def tool(self, mock_evaluator):
        """Create tool with standard budget."""
        budgets = [
            SplitBudget(
                split="dev",
                dataset_id="ds1",
                total_sample_budget=100,
                total_run_budget=5,
            ),
        ]
        return ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

    @pytest.mark.asyncio
    async def test_evaluate_commit_with_sample_ids(self, tool, mock_evaluator):
        """Test evaluate_commit with specific sample IDs."""
        result = await tool.evaluate_commit(
            commit="abc123",
            dataset_id="ds1",
            split="dev",
            sample_ids=[0, 1, 2],
        )

        mock_evaluator.evaluate.assert_called_once()
        assert "completed" in result
        assert tool._budget_map[("dev", "ds1")].remaining_sample_budget == 97
        assert tool._budget_map[("dev", "ds1")].remaining_run_budget == 4

    @pytest.mark.asyncio
    async def test_evaluate_commit_with_num_samples(self, tool, mock_evaluator):
        """Test evaluate_commit with num_samples."""
        _ = await tool.evaluate_commit(
            commit="abc123",
            dataset_id="ds1",
            split="dev",
            num_samples=10,
        )

        mock_evaluator.evaluate.assert_called_once()
        assert tool._budget_map[("dev", "ds1")].remaining_sample_budget == 90

    @pytest.mark.asyncio
    async def test_evaluate_commit_both_sample_ids_and_num_samples_raises(self, tool):
        """Test evaluate_commit raises when both sample_ids and num_samples provided."""
        with pytest.raises(ValueError) as exc_info:
            await tool.evaluate_commit(
                commit="abc123",
                dataset_id="ds1",
                split="dev",
                sample_ids=[0, 1],
                num_samples=10,
            )
        assert "Cannot specify both" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluate_commit_budget_exceeded(self, mock_evaluator):
        """Test evaluate_commit fails when budget exceeded."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=5),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        with pytest.raises(ExperimentBudgetExceeded):
            await tool.evaluate_commit(
                commit="abc123",
                dataset_id="ds1",
                split="dev",
                num_samples=10,
            )

    @pytest.mark.asyncio
    async def test_evaluate_commit_updates_budget_on_failure(self, mock_evaluator):
        """Test budget is updated even when evaluation fails."""
        mock_evaluator.evaluate.side_effect = Exception("Evaluation failed")

        budgets = [
            SplitBudget(
                split="dev",
                dataset_id="ds1",
                total_sample_budget=100,
                total_run_budget=5,
            ),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        with pytest.raises(Exception, match="Evaluation failed"):
            await tool.evaluate_commit(
                commit="abc123",
                dataset_id="ds1",
                split="dev",
                num_samples=10,
            )

        # Budget should still be decremented
        assert tool._budget_map[("dev", "ds1")].remaining_sample_budget == 90
        assert tool._budget_map[("dev", "ds1")].remaining_run_budget == 4


class TestExperimentRunnerToolCheckBudget:
    """Tests for check_remaining_experiment_budget method."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        evaluator = MagicMock()
        dataset_info = MagicMock()
        dataset_info.splits = {"dev": 100}
        # dataset_info provided by mock_dataset_info fixture
        evaluator.session_id = "test-session"
        return evaluator

    @pytest.mark.asyncio
    async def test_check_remaining_budget(self, mock_evaluator):
        """Test checking remaining budget."""
        budgets = [
            SplitBudget(
                split="dev",
                dataset_id="ds1",
                total_sample_budget=100,
                total_run_budget=5,
            ),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        result = await tool.check_remaining_experiment_budget("ds1", "dev")

        assert "100" in result
        assert "5" in result

    @pytest.mark.asyncio
    async def test_check_remaining_budget_invalid_split(self, mock_evaluator):
        """Test checking budget for invalid split raises error."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        with pytest.raises(InvalidSplitError):
            await tool.check_remaining_experiment_budget("ds1", "invalid")


class TestEvaluateCommitOnAllSplits:
    """Tests for evaluate_commit_on_all_splits method."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator with multiple splits."""
        evaluator = MagicMock()

        # Dataset info with multiple splits
        dataset_info = MagicMock()
        dataset_info.splits = {"dev": 100, "test": 50, "train": 200}
        # dataset_info provided by mock_dataset_info fixture

        # Git worktree
        evaluator.git_worktree.repo_name = "test_repo"
        mock_commit = MagicMock()
        mock_commit.hexsha = "abc123def456"
        evaluator.git_worktree.repo.commit.return_value = mock_commit

        # Async run method
        evaluator.evaluate = AsyncMock(return_value=make_mock_experiment())

        evaluator.session_id = "test-session"
        return evaluator

    @pytest.mark.asyncio
    async def test_evaluate_on_all_splits_success(self, mock_evaluator):
        """Test evaluating on all accessible splits."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100),
            SplitBudget(split="test", dataset_id="ds1", total_sample_budget=50),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        result = await tool.evaluate_commit_on_all_splits(
            commit="abc123",
            dataset_id="ds1",
        )

        # Should have results for both splits
        assert "dev" in result
        assert "test" in result
        assert mock_evaluator.evaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_on_all_splits_updates_budgets(self, mock_evaluator):
        """Test that budgets are updated for each split."""
        budgets = [
            SplitBudget(
                split="dev",
                dataset_id="ds1",
                total_sample_budget=200,
                total_run_budget=5,
            ),
            SplitBudget(
                split="test",
                dataset_id="ds1",
                total_sample_budget=100,
                total_run_budget=3,
            ),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        await tool.evaluate_commit_on_all_splits(commit="abc123", dataset_id="ds1")

        # Check budgets were decremented
        assert (
            tool._budget_map[("dev", "ds1")].remaining_sample_budget == 100
        )  # 200 - 100
        assert tool._budget_map[("dev", "ds1")].remaining_run_budget == 4
        assert (
            tool._budget_map[("test", "ds1")].remaining_sample_budget == 50
        )  # 100 - 50
        assert tool._budget_map[("test", "ds1")].remaining_run_budget == 2

    @pytest.mark.asyncio
    async def test_evaluate_on_all_splits_budget_capped(self, mock_evaluator):
        """Test that splits with limited budget are capped to remaining samples."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=200),
            SplitBudget(
                split="test", dataset_id="ds1", total_sample_budget=10
            ),  # Smaller than split
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        result = await tool.evaluate_commit_on_all_splits(
            commit="abc123", dataset_id="ds1"
        )

        # Both splits should succeed (test capped to 10 samples)
        assert "dev" in result
        assert "test" in result
        assert mock_evaluator.evaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_on_all_splits_all_fail_raises(self, mock_evaluator):
        """Test that ValueError is raised when all splits fail."""
        budgets = [
            SplitBudget(
                split="dev", dataset_id="ds1", total_run_budget=0
            ),  # No runs left
            SplitBudget(
                split="test", dataset_id="ds1", total_run_budget=0
            ),  # No runs left
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        with pytest.raises(ValueError, match="Failed to evaluate commit"):
            await tool.evaluate_commit_on_all_splits(commit="abc123", dataset_id="ds1")

    @pytest.mark.asyncio
    async def test_evaluate_on_all_splits_no_splits_found(self, mock_evaluator):
        """Test error when no splits found for dataset."""
        budgets = [
            SplitBudget(split="dev", dataset_id="other_ds", total_sample_budget=100),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        with pytest.raises(ValueError, match="No splits found"):
            await tool.evaluate_commit_on_all_splits(commit="abc123", dataset_id="ds1")

    @pytest.mark.asyncio
    async def test_evaluate_on_all_splits_evaluation_error(self, mock_evaluator):
        """Test handling when evaluation fails with exception."""
        # Make first call succeed, second fail
        mock_evaluator.evaluate = AsyncMock(
            side_effect=[make_mock_experiment(), Exception("Evaluation crashed")]
        )

        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=200),
            SplitBudget(split="test", dataset_id="ds1", total_sample_budget=100),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        result = await tool.evaluate_commit_on_all_splits(
            commit="abc123", dataset_id="ds1"
        )

        # Should still return results (one success, one error)
        assert "dev" in result or "test" in result
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_evaluate_on_all_splits_updates_budget_on_failure(
        self, mock_evaluator
    ):
        """Test that budget is updated even when evaluation fails."""
        mock_evaluator.evaluate = AsyncMock(side_effect=Exception("Evaluation failed"))

        budgets = [
            SplitBudget(
                split="dev",
                dataset_id="ds1",
                total_sample_budget=200,
                total_run_budget=5,
            ),
            SplitBudget(
                split="test",
                dataset_id="ds1",
                total_sample_budget=100,
                total_run_budget=3,
            ),
        ]
        tool = ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

        with pytest.raises(ValueError, match="Failed to evaluate"):
            await tool.evaluate_commit_on_all_splits(commit="abc123", dataset_id="ds1")

        # Budgets should still be decremented
        assert tool._budget_map[("dev", "ds1")].remaining_sample_budget == 100
        assert tool._budget_map[("dev", "ds1")].remaining_run_budget == 4
        assert tool._budget_map[("test", "ds1")].remaining_sample_budget == 50
        assert tool._budget_map[("test", "ds1")].remaining_run_budget == 2


class TestToolExtraction:
    """Tests for extracting tools from ExperimentRunnerTool."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        evaluator = MagicMock()
        dataset_info = MagicMock()
        dataset_info.splits = {"dev": 100}
        # dataset_info provided by mock_dataset_info fixture
        evaluator.session_id = "test-session"
        return evaluator

    @pytest.fixture
    def tool_instance(self, mock_evaluator):
        """Create an ExperimentRunnerTool instance."""
        budgets = [
            SplitBudget(split="dev", dataset_id="ds1", total_sample_budget=100),
        ]
        return ExperimentRunnerTool(evaluator=mock_evaluator, split_budgets=budgets)

    def test_get_all_tools_from_class(self, tool_instance):
        """Test that all @is_tool decorated methods are found."""
        tools = get_tools_from_class(ExperimentRunnerTool)
        tool_names = [t.__name__ for t in tools]

        assert "evaluate_commit" in tool_names
        assert "evaluate_commit_on_all_splits" in tool_names
        assert "check_remaining_experiment_budget" in tool_names
        assert len(tool_names) == 3

    def test_get_tools_from_instance_with_exclude_tools(self):
        """Test that exclude_tools on the instance filters out specified methods."""
        instance = ExperimentRunnerTool(exclude_tools=["evaluate_commit"])
        tools = get_tools_from_class(instance)
        tool_names = [t.__name__ for t in tools]

        assert "evaluate_commit" not in tool_names
        assert "evaluate_commit_on_all_splits" in tool_names
        assert "check_remaining_experiment_budget" in tool_names
        assert len(tool_names) == 2

    def test_tool_set_to_oai_tools_all(self, tool_instance):
        """Test converting all tools to OpenAI format."""
        oai_tools = tool_set_instance_to_oai_tools(tool_instance)

        tool_names = [t.name for t in oai_tools]
        assert len(tool_names) == 3
        assert "ExperimentRunnerTool_evaluate_commit" in tool_names
        assert "ExperimentRunnerTool_evaluate_commit_on_all_splits" in tool_names
        assert "ExperimentRunnerTool_check_remaining_experiment_budget" in tool_names

    def test_tool_set_to_oai_tools_exclude_evaluate_commit(self, tool_instance):
        """Test excluding evaluate_commit from tools."""
        oai_tools = tool_set_instance_to_oai_tools(
            tool_instance,
            exclude_methods=["evaluate_commit"],
        )

        tool_names = [t.name for t in oai_tools]
        assert len(tool_names) == 2
        assert "ExperimentRunnerTool_evaluate_commit" not in tool_names
        assert "ExperimentRunnerTool_evaluate_commit_on_all_splits" in tool_names
        assert "ExperimentRunnerTool_check_remaining_experiment_budget" in tool_names

    def test_tool_set_to_oai_tools_exclude_multiple(self, tool_instance):
        """Test excluding multiple methods."""
        oai_tools = tool_set_instance_to_oai_tools(
            tool_instance,
            exclude_methods=["evaluate_commit", "evaluate_commit_on_all_splits"],
        )

        tool_names = [t.name for t in oai_tools]
        assert len(tool_names) == 1
        assert "ExperimentRunnerTool_check_remaining_experiment_budget" in tool_names

    def test_tool_set_to_oai_tools_without_prefix(self, tool_instance):
        """Test tools without class name prefix."""
        oai_tools = tool_set_instance_to_oai_tools(
            tool_instance,
            prefix_class_name=False,
            exclude_methods=["evaluate_commit"],
        )

        tool_names = [t.name for t in oai_tools]
        # Without prefix, names should be just the method names
        assert "evaluate_commit_on_all_splits" in tool_names
        assert "check_remaining_experiment_budget" in tool_names
