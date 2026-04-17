"""Tests for VeroTask registration, validation, and pipeline."""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from vero.core.db.result import TaskOutput, TaskResult
from vero.core.evaluation import TaskParameters
from vero.core.task import VeroTask, create_task



# ---------------------------------------------------------------------------
# Registration via new methods
# ---------------------------------------------------------------------------


class TestDecoratorMethods:
    def test_inference_registers(self):
        t = create_task("t_inf", register=False)

        @t.inference()
        async def run_inference(task, evaluation_parameters): ...

        assert t.get("run_inference") is run_inference

    def test_inference_batch_registers(self):
        t = create_task("t_inf_b", register=False)

        @t.inference(batch=True)
        async def run_inference(tasks, evaluation_parameters): ...

        assert t.get("run_inference", batch=True) is run_inference

    def test_evaluation_registers(self):
        t = create_task("t_eval", register=False)

        @t.evaluation()
        async def run_evaluation(task, output, evaluation_parameters): ...

        assert t.get("run_evaluation") is run_evaluation

    def test_evaluation_batch_registers(self):
        t = create_task("t_eval_b", register=False)

        @t.evaluation(batch=True)
        async def run_evaluation(tasks, outputs, evaluation_parameters): ...

        assert t.get("run_evaluation", batch=True) is run_evaluation

    def test_load_data_registers(self):
        t = create_task("t_ld", register=False)

        @t.load_data()
        def load(evaluation_parameters): ...

        assert t.get("load_data") is load

    def test_duplicate_inference_raises(self):
        t = create_task("t_dup", register=False)

        @t.inference()
        async def fn1(task, evaluation_parameters): ...

        with pytest.raises(ValueError, match="already registered"):

            @t.inference()
            async def fn2(task, evaluation_parameters): ...

    def test_duplicate_evaluation_raises(self):
        t = create_task("t_dup2", register=False)

        @t.evaluation()
        async def fn1(task, output, evaluation_parameters): ...

        with pytest.raises(ValueError, match="already registered"):

            @t.evaluation()
            async def fn2(task, output, evaluation_parameters): ...


# ---------------------------------------------------------------------------
# Signature validation
# ---------------------------------------------------------------------------


class TestSignatureValidation:
    def test_inference_wrong_param_count_raises(self):
        t = create_task("t_sig1", register=False)
        with pytest.raises(TypeError, match="2 parameters"):

            @t.inference()
            async def bad(a, b, c): ...

    def test_evaluation_wrong_param_count_raises(self):
        t = create_task("t_sig2", register=False)
        with pytest.raises(TypeError, match="3 parameters"):

            @t.evaluation()
            async def bad(a, b): ...

    def test_load_data_wrong_param_count_raises(self):
        t = create_task("t_sig3", register=False)
        with pytest.raises(TypeError, match="1 parameters"):

            @t.load_data()
            def bad(a, b): ...

    def test_param_name_mismatch_does_not_error(self):
        """Mismatched param names produce a warning, not an error."""
        t = create_task("t_sig4", register=False)

        @t.inference()
        async def fn(x, y): ...

        assert t.get("run_inference") is fn


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_call_with_string_emits_deprecation(self):
        t = create_task("t_bc1", register=False)
        with pytest.warns(DeprecationWarning, match="deprecated"):

            @t("run_inference")
            async def fn(task, evaluation_parameters): ...

        assert t.get("run_inference") is fn

    def test_call_maps_load_task_data_to_load_data(self):
        t = create_task("t_bc2", register=False)
        with pytest.warns(DeprecationWarning):

            @t("load_task_data")
            def fn(evaluation_parameters): ...

        assert t.get("load_data") is fn

    def test_no_local_vero_task_export(self):
        assert VeroTask.__name__ == "VeroTask"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def setup_method(self):
        VeroTask.clear_registry()

    def teardown_method(self):
        VeroTask.clear_registry()

    def test_register_and_get(self):
        t = create_task("my_task")
        assert VeroTask.get_task("my_task") is t

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            VeroTask.get_task("nope")

    def test_clear_registry(self):
        create_task("temp")
        VeroTask.clear_registry()
        with pytest.raises(KeyError):
            VeroTask.get_task("temp")

    def test_duplicate_name_raises(self):
        create_task("dup")
        with pytest.raises(ValueError, match="already registered"):
            create_task("dup")


# ---------------------------------------------------------------------------
# task_parameters early validation
# ---------------------------------------------------------------------------


def _make_eval_params(task_params=None, num_samples=1):
    """Helper to build EvaluationParameters for testing."""
    from vero.core.db.candidate import Candidate
    from vero.core.db.dataset import DatasetSubset
    from vero.core.db.run import ExperimentRun
    from vero.core.evaluation import EvaluationParameters

    return EvaluationParameters(
        run=ExperimentRun(
            candidate=Candidate(commit="abc123", repo_name="test"),
            dataset_subset=DatasetSubset(
                split="test", dataset_id="test", sample_ids=list(range(num_samples))
            ),
        ),
        task_params=task_params or {},
        session_id="test-session",
    )


class TestTaskParametersValidation:
    @pytest.mark.asyncio
    async def test_valid_params_pass(self):
        class MyParams(TaskParameters):
            model: str = "default"

        t = create_task("v1", register=False, task_parameters=MyParams)

        @t.inference()
        async def infer(task, evaluation_parameters):
            return TaskOutput(output="ok")

        @t.evaluation()
        async def evaluate(task, output, evaluation_parameters):
            return TaskResult(score=1.0)

        @t.load_data()
        def load(evaluation_parameters):
            return [{"id": 0}]

        params = _make_eval_params(task_params={"model": "gpt-4"})
        metrics = await t.run(params)
        assert metrics["num_samples"] == 1

    @pytest.mark.asyncio
    async def test_unknown_key_raises_before_inference(self):
        class MyParams(TaskParameters):
            model: str = "default"

        t = create_task("v2", register=False, task_parameters=MyParams)
        inference_called = False

        @t.inference()
        async def infer(task, evaluation_parameters):
            nonlocal inference_called
            inference_called = True
            return TaskOutput(output="ok")

        @t.evaluation()
        async def evaluate(task, output, evaluation_parameters):
            return TaskResult(score=1.0)

        params = _make_eval_params(task_params={"modle": "typo"})
        with pytest.raises(ValidationError, match="Extra inputs"):
            await t.run(params)
        assert not inference_called

    @pytest.mark.asyncio
    async def test_no_task_parameters_skips_validation(self):
        t = create_task("v3", register=False)

        @t.inference()
        async def infer(task, evaluation_parameters):
            return TaskOutput(output="ok")

        @t.evaluation()
        async def evaluate(task, output, evaluation_parameters):
            return TaskResult(score=1.0)

        @t.load_data()
        def load(evaluation_parameters):
            return [{"id": 0}]

        params = _make_eval_params(task_params={"anything": "goes"})
        metrics = await t.run(params)
        assert metrics["num_samples"] == 1


# ---------------------------------------------------------------------------
# Load data pipeline
# ---------------------------------------------------------------------------


class TestLoadData:
    @pytest.mark.asyncio
    async def test_custom_load_data_replaces_default(self):
        t = create_task("ld1", register=False)

        @t.load_data()
        def load(evaluation_parameters):
            return [{"custom": True, "id": 0}, {"custom": True, "id": 1}]

        @t.inference()
        async def infer(task, evaluation_parameters):
            assert task["custom"] is True
            return TaskOutput(output="ok")

        @t.evaluation()
        async def evaluate(task, output, evaluation_parameters):
            return TaskResult(score=1.0)

        params = _make_eval_params(num_samples=2)
        metrics = await t.run(params)
        assert metrics["num_samples"] == 2
        assert metrics["num_errors"] == 0

    @pytest.mark.asyncio
    async def test_no_load_data_requires_dataset_id(self):
        """Without @task.load_data(), run() needs a dataset_id."""
        t = create_task("ld2", register=False)

        @t.inference()
        async def infer(task, evaluation_parameters):
            return TaskOutput(output="ok")

        @t.evaluation()
        async def evaluate(task, output, evaluation_parameters):
            return TaskResult(score=1.0)

        params = _make_eval_params()
        with pytest.raises(ValueError, match="dataset_id"):
            await t.run(params)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        t = create_task("pipe1", register=False)

        @t.load_data()
        def load(evaluation_parameters):
            return [{"q": "2+2"}, {"q": "3+3"}]

        @t.inference()
        async def infer(task, evaluation_parameters):
            return TaskOutput(output=str(eval(task["q"])))

        @t.evaluation()
        async def evaluate(task, output, evaluation_parameters):
            expected = str(eval(task["q"]))
            score = 1.0 if output.output == expected else 0.0
            return TaskResult(score=score)

        params = _make_eval_params(num_samples=2)
        metrics = await t.run(params)
        assert metrics["num_samples"] == 2
        assert metrics["avg_score"] == 1.0

    @pytest.mark.asyncio
    async def test_missing_inference_raises(self):
        t = create_task("pipe2", register=False)

        @t.evaluation()
        async def evaluate(task, output, evaluation_parameters):
            return TaskResult(score=1.0)

        params = _make_eval_params()
        with pytest.raises(RuntimeError, match="No inference function"):
            await t.run(params)

    @pytest.mark.asyncio
    async def test_missing_evaluation_raises(self):
        t = create_task("pipe3", register=False)

        @t.inference()
        async def infer(task, evaluation_parameters):
            return TaskOutput(output="ok")

        params = _make_eval_params()
        with pytest.raises(RuntimeError, match="No evaluation function"):
            await t.run(params)
