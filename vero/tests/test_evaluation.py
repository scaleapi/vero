"""Tests for TaskParameters and EvaluationParameters.parse_task_params."""

import pytest
from pydantic import ValidationError

from vero.core.evaluation import TaskParameters


class TestTaskParameters:
    """Tests for TaskParameters base class."""

    def test_empty_params(self):
        """Empty dict produces default instance."""
        params = TaskParameters.model_validate({})
        assert params is not None

    def test_forbids_extra_keys(self):
        """Unknown keys raise ValidationError."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TaskParameters.model_validate({"unknown_key": "value"})

    def test_subclass_with_defaults(self):
        """Subclass with defaults works from empty dict."""

        class MyParams(TaskParameters):
            model: str = "gpt-4.1-mini"
            temperature: float = 0.0

        params = MyParams.model_validate({})
        assert params.model == "gpt-4.1-mini"
        assert params.temperature == 0.0

    def test_subclass_with_values(self):
        """Subclass populated from dict."""

        class MyParams(TaskParameters):
            model: str = "gpt-4.1-mini"
            num_trials: int = 1

        params = MyParams.model_validate({"model": "claude-sonnet", "num_trials": 5})
        assert params.model == "claude-sonnet"
        assert params.num_trials == 5

    def test_subclass_rejects_unknown_keys(self):
        """Subclass with extra="forbid" catches typos."""

        class MyParams(TaskParameters):
            model: str = "gpt-4.1-mini"

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MyParams.model_validate({"model": "gpt-4.1-mini", "modle": "typo"})

    def test_subclass_type_coercion(self):
        """Pydantic coerces compatible types."""

        class MyParams(TaskParameters):
            temperature: float = 0.0

        params = MyParams.model_validate({"temperature": 1})
        assert params.temperature == 1.0
        assert isinstance(params.temperature, float)

    def test_subclass_type_error(self):
        """Incompatible types raise ValidationError."""

        class MyParams(TaskParameters):
            num_trials: int = 1

        with pytest.raises(ValidationError):
            MyParams.model_validate({"num_trials": "not_a_number"})


class TestParseTaskParams:
    """Tests for EvaluationParameters.parse_task_params."""

    @pytest.fixture
    def make_eval_params(self):
        """Factory for EvaluationParameters with given task_params."""
        from vero.core.db.dataset import DatasetSubset
        from vero.core.db.run import ExperimentRun
        from vero.core.db.candidate import Candidate
        from vero.core.evaluation import EvaluationParameters

        def _make(task_params: dict | None = None):
            return EvaluationParameters(
                run=ExperimentRun(
                    candidate=Candidate(commit="abc123", repo_name="test"),
                    dataset_subset=DatasetSubset(split="test", dataset_id="test"),
                ),
                task="test_task",
                task_params=task_params or {},
                session_id="test-session",
            )

        return _make

    def test_parse_empty(self, make_eval_params):
        """parse_task_params with empty dict returns defaults."""

        class MyParams(TaskParameters):
            model: str = "default"

        ep = make_eval_params()
        params = ep.parse_task_params(MyParams)
        assert params.model == "default"

    def test_parse_populated(self, make_eval_params):
        """parse_task_params with values returns populated model."""

        class MyParams(TaskParameters):
            model: str = "default"
            temperature: float = 0.0

        ep = make_eval_params({"model": "gpt-4", "temperature": 0.7})
        params = ep.parse_task_params(MyParams)
        assert params.model == "gpt-4"
        assert params.temperature == 0.7

    def test_parse_rejects_typo(self, make_eval_params):
        """parse_task_params raises on unknown keys."""

        class MyParams(TaskParameters):
            model: str = "default"

        ep = make_eval_params({"modle": "typo"})
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ep.parse_task_params(MyParams)
