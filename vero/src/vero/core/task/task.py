"""VeroTask: decorator-based task registration and evaluation pipeline."""

from __future__ import annotations

import inspect
import logging
import traceback
import warnings
from typing import Any, Callable, NamedTuple, Sequence, TypeVar

from datasets import Dataset, DatasetDict
from pydantic import JsonValue

from vero.core.db.dataset import DatasetSample
from vero.core.db.result import SampleResult, TaskOutput, TaskResult
from vero.core.evaluation import EvaluationParameters
from vero.core.sessions import (
    get_vero_home_dir,
    load_all_sample_results,
    load_sample_result,
    save_sample_result,
)
from vero.core.utils import limited_gather, maybe_await

logger = logging.getLogger(__name__)

TaskT = TypeVar("TaskT")


class TaskFunctionSpec(NamedTuple):
    """Definition of expected signature for a task function."""

    name: str
    batch: bool
    params: list[str]


class VeroTask:
    """Decorator-based task registration with signature validation and evaluation pipeline.

    Usage:
        task = VeroTask("my_task")

        @task.inference()
        async def run_inference(task, evaluation_parameters):
            ...

        @task.evaluation(batch=True)
        async def run_evaluation(tasks, outputs, evaluation_parameters):
            ...

    Required functions (provide single OR batch version):

        Inference (choose one):
            @task.inference() - (task, evaluation_parameters) -> TaskOutput
            @task.inference(batch=True) - (tasks, evaluation_parameters) -> Sequence[TaskOutput]

        Evaluation (choose one):
            @task.evaluation() - (task, output, evaluation_parameters) -> TaskResult
            @task.evaluation(batch=True) - (tasks, outputs, evaluation_parameters) -> Sequence[TaskResult]

    Optional:
        @task.load_data() - (evaluation_parameters) -> Sequence[dict]
            Custom data loading. If not provided, defaults to HuggingFace dataset loading.
    """

    _registry: dict[str, VeroTask] = {}

    FUNCTION_SPECS = [
        TaskFunctionSpec("run_inference", False, ["task", "evaluation_parameters"]),
        TaskFunctionSpec("run_inference", True, ["tasks", "evaluation_parameters"]),
        TaskFunctionSpec(
            "run_evaluation", False, ["task", "output", "evaluation_parameters"]
        ),
        TaskFunctionSpec(
            "run_evaluation", True, ["tasks", "outputs", "evaluation_parameters"]
        ),
        TaskFunctionSpec("load_data", False, ["evaluation_parameters"]),
    ]

    def __init__(
        self,
        name: str,
        register: bool = True,
        task_parameters_type: type | None = None,
        required_env_vars: list[str] | None = None,
        label_fields: list[str] | None = None,
    ):
        """Initialize a VeroTask.

        Args:
            name: Task name for registry lookup.
            register: Whether to register in the global registry.
            task_parameters_type: Optional TaskParameters subclass for early validation
                of evaluation_parameters.task_params in run().
            required_env_vars: Environment variables that must be set for this task
                to run (e.g. ``["LITELLM_BASE_URL", "LITELLM_API_KEY"]``).
                Checked before the evaluation subprocess starts.
            label_fields: Dataset columns holding labels/ground truth. Stripped from
                each sample before inference (so inference never sees them); scoring
                receives the full row. Static, immutable task property.
        """
        self.name = name
        self._functions: dict[str, Callable] = {}
        self._batch_functions: dict[str, Callable] = {}
        self._task_parameters_type = task_parameters_type
        self.required_env_vars: list[str] = required_env_vars or []
        self.label_fields: list[str] = label_fields or []

        if register:
            if name in VeroTask._registry:
                raise ValueError(f"VeroTask '{name}' already registered")
            VeroTask._registry[name] = self

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def _register_function(self, name: str, batch: bool = False) -> Callable:
        """Register a function with signature validation.

        Args:
            name: Internal function tag (e.g., "run_inference", "run_evaluation").
            batch: Whether this is a batch function.

        Returns:
            Decorator function.
        """

        def decorator(fn: Callable) -> Callable:
            # Check for duplicate registration
            target_dict = self._batch_functions if batch else self._functions
            if name in target_dict:
                existing_fn = target_dict[name]
                raise ValueError(
                    f"Tag '{name}' is already registered to function '{existing_fn.__name__}'. "
                    f"Cannot register it again to '{fn.__name__}'. "
                    f"Each decorator (batch={batch}) can only be used once."
                )

            # Find expected signature
            expected_sig = None
            for spec in self.FUNCTION_SPECS:
                if spec.name == name and spec.batch == batch:
                    expected_sig = spec.params
                    break

            # Validate signature if we have an expected signature
            if expected_sig is not None:
                self._validate_signature(fn, expected_sig, name)
            else:
                valid_names = {spec.name for spec in self.FUNCTION_SPECS}
                logger.warning(
                    f"Unrecognized tag '{name}' for function {fn.__name__}. "
                    f"Valid tags: {', '.join(sorted(valid_names))}. "
                    f"This function will be stored but may not be called by run()."
                )

            # Store the function
            if batch:
                self._batch_functions[name] = fn
            else:
                self._functions[name] = fn
            return fn

        return decorator

    def inference(self, batch: bool = False) -> Callable:
        """Register an inference function.

        Args:
            batch: If True, register as batch inference.

        Returns:
            Decorator function.
        """
        return self._register_function("run_inference", batch=batch)

    def evaluation(self, batch: bool = False) -> Callable:
        """Register an evaluation function.

        Args:
            batch: If True, register as batch evaluation.

        Returns:
            Decorator function.
        """
        return self._register_function("run_evaluation", batch=batch)

    def load_data(self) -> Callable:
        """Register a custom data loading function.

        The function should accept (evaluation_parameters) and return a sequence
        of task objects (dicts or typed objects).

        Returns:
            Decorator function.
        """
        return self._register_function("load_data", batch=False)

    def __call__(self, name: str, batch: bool = False) -> Callable:
        """Register a function by tag string (deprecated).

        Use .inference(), .evaluation(), or .load_data() instead.
        """
        warnings.warn(
            f'@task("{name}") is deprecated. '
            f"Use @task.inference(), @task.evaluation(), or @task.load_data() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Map old names to new internal names
        internal_name = name
        if name == "load_task_data" or name == "create_task":
            internal_name = "load_data"
        return self._register_function(internal_name, batch=batch)

    # -------------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------------

    def get(self, tag: str, batch: bool = False) -> Callable | None:
        """Get a registered function by tag.

        Args:
            tag: Function tag to retrieve.
            batch: Whether to get batch or single-sample function.

        Returns:
            Registered function or None.
        """
        return self._batch_functions.get(tag) if batch else self._functions.get(tag)

    def has(self, tag: str, batch: bool = False) -> bool:
        """Check if a function is registered for a tag.

        Args:
            tag: Function tag to check.
            batch: Whether to check batch or single-sample.

        Returns:
            True if function is registered.
        """
        return self.get(tag, batch) is not None

    @classmethod
    def get_task(cls, name: str) -> VeroTask:
        """Get a registered task by name.

        Args:
            name: Task name.

        Returns:
            VeroTask instance.

        Raises:
            KeyError: If task not found.
        """
        if name not in cls._registry:
            registered = list(cls._registry.keys())
            raise KeyError(f"VeroTask '{name}' not found. Registered: {registered}")
        return cls._registry[name]

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the global registry."""
        cls._registry.clear()

    # -------------------------------------------------------------------------
    # Signature validation
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_signature(
        fn: Callable, expected_params: list[str], name: str
    ) -> None:
        """Validate that a function has the expected parameter names.

        Args:
            fn: The function to validate.
            expected_params: List of expected parameter names.
            name: Function name for error messages.

        Raises:
            TypeError: If the signature doesn't match.
        """
        sig = inspect.signature(fn)
        actual_params = list(sig.parameters.keys())

        if len(actual_params) != len(expected_params):
            raise TypeError(
                f"@task.{name}() expects a function with {len(expected_params)} parameters "
                f"({', '.join(expected_params)}), but got {len(actual_params)} parameters "
                f"({', '.join(actual_params)}). "
                f"\n\nExpected signature: def {fn.__name__}({', '.join(expected_params)}) -> ..."
            )

        # Warn if parameter names differ but don't error
        for expected, actual in zip(expected_params, actual_params):
            if expected != actual:
                logger.warning(
                    f"@task.{name}(): Parameter '{actual}' should be named '{expected}' "
                    f"for consistency. Function: {fn.__name__}"
                )

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    @staticmethod
    def _default_load_task_data(
        evaluation_parameters: EvaluationParameters,
    ) -> Sequence[dict[str, JsonValue]]:
        """Default implementation for loading task data from HuggingFace datasets.

        Args:
            evaluation_parameters: Evaluation parameters with dataset config.

        Returns:
            Filtered dataset samples.
        """
        if not evaluation_parameters.dataset_id:
            raise ValueError("Evaluation parameters do not have a dataset_id!")

        from vero.core.dataset.store import load_dataset

        vero_home = get_vero_home_dir()
        dataset_dict: DatasetDict = load_dataset(
            vero_home / "sessions", vero_home / "datasets",
            evaluation_parameters.session_id, evaluation_parameters.dataset_id
        )

        split = evaluation_parameters.run.dataset_subset.split
        if split is None:
            assert len(dataset_dict) == 1, (
                "DatasetDict has multiple splits, so split must be provided!"
            )
            split = list(dataset_dict.keys())[0]

        dataset: Dataset = dataset_dict[split]

        if evaluation_parameters.run.dataset_subset.sample_ids is not None:
            dataset = dataset.select(
                evaluation_parameters.run.dataset_subset.sample_ids
            )

        return dataset

    def _load_and_prepare_data(
        self, evaluation_parameters: EvaluationParameters
    ) -> tuple[Sequence, Sequence[dict[str, JsonValue]] | None]:
        """Load and prepare task data.

        If a custom @task.load_data() is registered, uses it exclusively.
        Otherwise, falls back to default HuggingFace dataset loading.

        Args:
            evaluation_parameters: Evaluation parameters with dataset config.

        Returns:
            Tuple of (tasks, task_data) where task_data is the raw dicts
            for result saving (None if custom loader is used).
        """
        # Check for custom load_data function
        load_data_fn = self.get("load_data", batch=False)
        if load_data_fn is not None:
            tasks = load_data_fn(evaluation_parameters)
            return tasks, None

        # Default: load from HuggingFace
        task_data = self._default_load_task_data(evaluation_parameters)
        return task_data, task_data

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    @staticmethod
    def cast_to_task_output(obj: Any) -> TaskOutput:
        """Cast an object to a TaskOutput."""
        if isinstance(obj, TaskOutput):
            return obj
        if isinstance(obj, Exception):
            return TaskOutput(error=obj)
        return TaskOutput(output=obj)

    async def run_batch_inference(
        self, tasks: Sequence[TaskT], evaluation_parameters: EvaluationParameters
    ) -> list[TaskOutput]:
        """Run inference on a batch of tasks.

        Checks for batch inference function first, then falls back to
        per-sample inference with concurrency.

        Args:
            tasks: Batch of task objects.
            evaluation_parameters: Evaluation parameters.

        Returns:
            List of task outputs.
        """
        # Check for batch inference function
        batch_inference_fn = self.get("run_inference", batch=True)
        if batch_inference_fn:
            result = await maybe_await(batch_inference_fn(tasks, evaluation_parameters))
            return [self.cast_to_task_output(result) for result in result]

        # Fall back to per-sample inference
        inference_fn = self.get("run_inference", batch=False)
        if not inference_fn:
            raise RuntimeError(
                "No inference function registered. "
                "Use @task.inference() or @task.inference(batch=True) to register one."
            )

        results = await limited_gather(
            coro_factories=[
                lambda t=task: inference_fn(t, evaluation_parameters) for task in tasks
            ],
            limit=evaluation_parameters.max_concurrency,
            retry_config=evaluation_parameters.retry_config,
            desc="Running inference",
            return_exceptions=True,
            timeout=evaluation_parameters.sample_timeout,
            run_in_thread=evaluation_parameters.use_threading,
        )
        return [self.cast_to_task_output(result) for result in results]

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    @staticmethod
    def cast_to_task_result(task_output: TaskOutput, obj: Any) -> TaskResult:
        """Cast an object to a TaskResult."""
        if isinstance(obj, TaskResult):
            return obj
        elif isinstance(obj, Exception):
            error_traceback = "".join(
                traceback.format_exception(type(obj), obj, obj.__traceback__)
            )
            return TaskResult.from_task_output(
                task_output=task_output,
                eval_error=str(obj),
                error_traceback=error_traceback,
            )
        else:
            raise ValueError(
                f"Expected TaskResult or Exception, got {type(obj).__name__}."
            )

    async def run_batch_evaluation(
        self,
        tasks: Sequence[TaskT],
        outputs: Sequence[TaskOutput],
        evaluation_parameters: EvaluationParameters,
    ) -> list[TaskResult]:
        """Run evaluation on a batch of tasks and outputs.

        Checks for batch evaluation function first, then falls back to
        per-sample evaluation with concurrency.

        Args:
            tasks: Batch of task objects.
            outputs: Batch of task outputs.
            evaluation_parameters: Evaluation parameters.

        Returns:
            List of evaluation results or exceptions.
        """
        # Check for batch evaluation function
        batch_eval_fn = self.get("run_evaluation", batch=True)
        if batch_eval_fn:
            result = await maybe_await(
                batch_eval_fn(tasks, outputs, evaluation_parameters)
            )
            return [
                self.cast_to_task_result(output, result)
                for output, result in zip(outputs, result)
            ]

        # Fall back to per-sample evaluation
        eval_fn = self.get("run_evaluation", batch=False)
        if not eval_fn:
            raise RuntimeError(
                "No evaluation function registered. "
                "Use @task.evaluation() or @task.evaluation(batch=True) to register one."
            )

        async def evaluate_safely(task: TaskT, output: TaskOutput) -> TaskResult:
            try:
                result = await maybe_await(eval_fn(task, output, evaluation_parameters))
                return self.cast_to_task_result(output, result)
            except Exception as e:
                return self.cast_to_task_result(output, e)

        results = await limited_gather(
            coro_factories=[
                lambda t=task, o=output: evaluate_safely(t, o)
                for task, output in zip(tasks, outputs)
            ],
            limit=evaluation_parameters.max_concurrency,
            retry_config=evaluation_parameters.retry_config,
            desc="Evaluating samples",
            return_exceptions=True,
            timeout=evaluation_parameters.sample_timeout,
            run_in_thread=evaluation_parameters.use_threading,
        )
        return results

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Per-stage persistence
    #
    # Each sample is persisted to its own ``samples/{id}.json`` file as it
    # completes: a partial SampleResult after inference (score=None), then the
    # same file updated with scoring fields. This makes every stage independently
    # runnable and resumable from any partial state.
    # -------------------------------------------------------------------------

    def _sessions_dir(self) -> Path:
        return get_vero_home_dir() / "sessions"

    def _scrub_inputs(self, row: Any) -> Any:
        """Strip ``label_fields`` from a sample before it reaches inference.

        Only applies to mapping rows; non-mapping rows pass through unchanged.
        """
        if not self.label_fields:
            return row
        try:
            return {k: v for k, v in dict(row).items() if k not in self.label_fields}
        except (TypeError, ValueError):
            return row

    def _dataset_sample(
        self, params: EvaluationParameters, sample_id: int
    ) -> DatasetSample:
        return DatasetSample(
            sample_id=sample_id,
            split=params.run.dataset_subset.split,
            dataset_id=params.run.dataset_subset.dataset_id,
        )

    def _save_inference(
        self,
        params: EvaluationParameters,
        sample_id: int,
        task_data: Sequence[dict[str, JsonValue]] | None,
        pos: int,
        output: TaskOutput,
    ) -> None:
        """Persist a partial SampleResult holding only the inference output."""
        sample_input = (
            self._scrub_inputs(dict(task_data[pos]))
            if task_data is not None and pos < len(task_data)
            else None
        )
        sample_result = SampleResult.from_task_result(
            dataset_sample=self._dataset_sample(params, sample_id),
            task_result=TaskResult.from_task_output(output),
            commit=params.run.candidate.commit,
            result_id=params.result_id,
            input=sample_input,
        )
        save_sample_result(
            self._sessions_dir(),
            params.session_id,
            params.result_id,
            sample_id=sample_id,
            result=sample_result,
        )

    def _save_score(
        self,
        params: EvaluationParameters,
        sample_result: SampleResult,
        task_result: TaskResult,
    ) -> None:
        """Update a persisted SampleResult with scoring-stage fields and re-save."""
        sample_result.score = task_result.score
        sample_result.feedback = task_result.feedback
        sample_result.metrics = task_result.metrics
        sample_result.eval_error = task_result.eval_error
        sample_result.eval_trace = task_result.eval_trace
        if task_result.error_traceback and sample_result.error_traceback is None:
            sample_result.error_traceback = task_result.error_traceback
        save_sample_result(
            self._sessions_dir(),
            params.session_id,
            params.result_id,
            sample_id=sample_result.dataset_sample.sample_id,
            result=sample_result,
        )

    def compute_metrics(
        self, params: EvaluationParameters
    ) -> dict[str, int | float | None]:
        """Compute metrics from the SampleResults persisted on disk."""
        from vero.core.constants import default_minimum_score

        sample_results = load_all_sample_results(
            self._sessions_dir(), params.session_id, params.result_id
        )

        metrics: dict[str, int | float | None] = {
            "num_samples": len(sample_results),
            "num_errors": 0,
            "avg_score": 0,
            "avg_filled_score": None,
        }
        for sr in sample_results.values():
            if sr.error is not None or sr.eval_error is not None:
                metrics["num_errors"] += 1
            elif sr.score is not None:
                metrics["avg_score"] += sr.score

        metrics["num_successes"] = metrics["num_samples"] - metrics["num_errors"]

        if metrics["num_successes"] > 0:
            metrics["avg_score"] /= metrics["num_successes"]
        else:
            metrics["avg_score"] = None

        metrics["avg_filled_score"] = metrics["avg_score"]
        if metrics["avg_score"] is None:
            metrics["avg_filled_score"] = default_minimum_score
        elif metrics["num_errors"] > 0:
            metrics["avg_filled_score"] = (
                metrics["num_successes"] * metrics["avg_score"]
                + metrics["num_errors"] * default_minimum_score
            ) / metrics["num_samples"]

        return metrics

    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------

    def _validate_required_functions(self) -> None:
        """Validate that all required functions are registered.

        Raises:
            RuntimeError: If required functions are missing.
        """
        errors = []

        has_single_inference = self.has("run_inference", batch=False)
        has_batch_inference = self.has("run_inference", batch=True)
        if not has_single_inference and not has_batch_inference:
            errors.append(
                "No inference function registered. "
                "Use @task.inference() or @task.inference(batch=True)"
            )

        has_single_eval = self.has("run_evaluation", batch=False)
        has_batch_eval = self.has("run_evaluation", batch=True)
        if not has_single_eval and not has_batch_eval:
            errors.append(
                "No evaluation function registered. "
                "Use @task.evaluation() or @task.evaluation(batch=True)"
            )

        if errors:
            raise RuntimeError(
                f"Task '{self.name}' is missing required functions:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    async def run_inference_stage(self, params: EvaluationParameters) -> None:
        """Run (or resume) inference, persisting each sample as it completes.

        Resume: samples whose ``samples/{id}.json`` already exists are skipped.
        Per-sample inference persists incrementally; a batch inference function
        persists after the batch returns.
        """
        tasks, task_data = self._load_and_prepare_data(params)
        sample_ids = params.run.dataset_subset.sample_ids
        if sample_ids is None:
            sample_ids = list(range(len(tasks)))

        sessions_dir = self._sessions_dir()
        pending = [
            (pos, sid)
            for pos, sid in enumerate(sample_ids)
            if load_sample_result(sessions_dir, params.session_id, params.result_id, sid)
            is None
        ]
        if not pending:
            logger.info("Inference stage: all samples already persisted; skipping")
            return

        single_fn = self.get("run_inference", batch=False)
        batch_fn = self.get("run_inference", batch=True)
        if single_fn is None and batch_fn is None:
            raise RuntimeError(
                "No inference function registered. "
                "Use @task.inference() or @task.inference(batch=True) to register one."
            )

        if single_fn is not None:

            async def infer_and_save(pos: int, sid: int) -> TaskOutput:
                output = self.cast_to_task_output(
                    await maybe_await(single_fn(self._scrub_inputs(tasks[pos]), params))
                )
                self._save_inference(params, sid, task_data, pos, output)
                return output

            results = await limited_gather(
                coro_factories=[
                    (lambda p=pos, s=sid: infer_and_save(p, s)) for pos, sid in pending
                ],
                limit=params.max_concurrency,
                retry_config=params.retry_config,
                desc="Running inference",
                return_exceptions=True,
                timeout=params.sample_timeout,
                run_in_thread=params.use_threading,
            )
            # Persist an error record for samples that exhausted retries.
            for (pos, sid), res in zip(pending, results):
                if isinstance(res, Exception):
                    self._save_inference(
                        params, sid, task_data, pos, TaskOutput(error=res)
                    )
        else:
            outputs = await self.run_batch_inference(
                [self._scrub_inputs(tasks[pos]) for pos, _ in pending], params
            )
            for (pos, sid), output in zip(pending, outputs):
                self._save_inference(params, sid, task_data, pos, output)

        logger.info(f"Inference stage complete: {len(pending)} samples")

    async def run_scoring_stage(self, params: EvaluationParameters) -> None:
        """Run (or resume) scoring over persisted inference outputs.

        Skips samples that errored during inference (terminal) or are already
        scored. Reads inference outputs from disk and re-persists with scores.
        """
        tasks, _ = self._load_and_prepare_data(params)
        sample_ids = params.run.dataset_subset.sample_ids
        if sample_ids is None:
            sample_ids = list(range(len(tasks)))

        existing = load_all_sample_results(
            self._sessions_dir(), params.session_id, params.result_id
        )
        pending: list[tuple[int, SampleResult]] = []
        for pos, sid in enumerate(sample_ids):
            sr = existing.get(sid)
            if sr is None:
                logger.warning(
                    f"Scoring stage: no inference result for sample {sid}; skipping"
                )
                continue
            if sr.error is not None:  # inference error is terminal
                continue
            if sr.is_scored():
                continue
            pending.append((pos, sr))
        if not pending:
            logger.info("Scoring stage: nothing to score; skipping")
            return

        single_fn = self.get("run_evaluation", batch=False)
        batch_fn = self.get("run_evaluation", batch=True)
        if single_fn is None and batch_fn is None:
            raise RuntimeError(
                "No evaluation function registered. "
                "Use @task.evaluation() or @task.evaluation(batch=True) to register one."
            )

        def _output(sr: SampleResult) -> TaskOutput:
            return TaskOutput(
                output=sr.output, error=sr.error, execution_trace=sr.execution_trace
            )

        if single_fn is not None:

            async def score_and_save(pos: int, sr: SampleResult) -> None:
                result = await maybe_await(single_fn(tasks[pos], _output(sr), params))
                self._save_score(params, sr, self.cast_to_task_result(_output(sr), result))

            results = await limited_gather(
                coro_factories=[
                    (lambda p=pos, r=sr: score_and_save(p, r)) for pos, sr in pending
                ],
                limit=params.max_concurrency,
                retry_config=params.retry_config,
                desc="Evaluating samples",
                return_exceptions=True,
                timeout=params.sample_timeout,
                run_in_thread=params.use_threading,
            )
            for (pos, sr), res in zip(pending, results):
                if isinstance(res, Exception):
                    self._save_score(
                        params, sr, self.cast_to_task_result(_output(sr), res)
                    )
        else:
            eval_results = await self.run_batch_evaluation(
                [tasks[pos] for pos, _ in pending],
                [_output(sr) for _, sr in pending],
                params,
            )
            for (pos, sr), task_result in zip(pending, eval_results):
                self._save_score(params, sr, task_result)

        logger.info(f"Scoring stage complete: {len(pending)} samples")

    async def run(self, params: EvaluationParameters) -> dict[str, Any]:
        """Run the full evaluation pipeline as two resumable stages.

        Inference and scoring each persist per-sample as they complete and skip
        already-done samples, so a crashed run resumes from its partial state.

        Args:
            params: Evaluation parameters.

        Returns:
            Metrics dictionary (computed from the persisted sample results).

        Raises:
            RuntimeError: If required functions are not registered.
            pydantic.ValidationError: If task_params fail validation against
                the registered task_parameters type.
        """
        # Validate task_params against registered type (fail-fast)
        if self._task_parameters_type is not None:
            params.parse_task_params(self._task_parameters_type)

        # Validate required functions are registered
        self._validate_required_functions()

        await self.run_inference_stage(params)
        await self.run_scoring_stage(params)

        metrics = self.compute_metrics(params)
        logger.info(f"Logged results: {metrics}")
        return metrics

    def __repr__(self) -> str:
        tags = list(self._functions.keys())
        batch_tags = list(self._batch_functions.keys())
        return f"VeroTask(name={self.name!r}, tags={tags}, batch_tags={batch_tags})"
