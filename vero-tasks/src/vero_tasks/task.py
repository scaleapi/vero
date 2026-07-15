"""Decorator-based task definition without evaluation persistence concerns."""

from __future__ import annotations

import asyncio
import inspect
import os
import traceback
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from vero_tasks.models import TaskContext, TaskOutput, TaskResult, TaskT


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class TaskDefinition:
    """Inference and evaluation functions registered under one task name."""

    _registry: dict[str, TaskDefinition] = {}

    def __init__(
        self,
        name: str,
        *,
        register: bool = True,
        task_parameters_type: type | None = None,
        required_env_vars: Sequence[str] | None = None,
    ):
        if not name.strip():
            raise ValueError("task name must not be empty")
        if register and name in self._registry:
            raise ValueError(f"task {name!r} is already registered")
        self.name = name
        self.task_parameters_type = task_parameters_type
        self.required_env_vars = tuple(required_env_vars or ())
        self._single: dict[str, Callable[..., Any]] = {}
        self._batch: dict[str, Callable[..., Any]] = {}
        if register:
            self._registry[name] = self

    def _decorator(self, kind: str, *, batch: bool) -> Callable:
        expected = {
            ("inference", False): 2,
            ("inference", True): 2,
            ("evaluation", False): 3,
            ("evaluation", True): 3,
            ("load_data", False): 1,
        }[(kind, batch)]

        def register(function: Callable[..., Any]) -> Callable[..., Any]:
            parameters = inspect.signature(function).parameters
            if len(parameters) != expected:
                raise TypeError(
                    f"{kind} function {function.__name__!r} must accept "
                    f"{expected} parameters, got {len(parameters)}"
                )
            functions = self._batch if batch else self._single
            if kind in functions:
                raise ValueError(f"{kind} function is already registered")
            functions[kind] = function
            return function

        return register

    def inference(self, *, batch: bool = False) -> Callable:
        return self._decorator("inference", batch=batch)

    def evaluation(self, *, batch: bool = False) -> Callable:
        return self._decorator("evaluation", batch=batch)

    def load_data(self) -> Callable:
        return self._decorator("load_data", batch=False)

    def __call__(self, name: str, *, batch: bool = False) -> Callable:
        aliases = {
            "run_inference": "inference",
            "run_evaluation": "evaluation",
            "load_task_data": "load_data",
            "create_task": "load_data",
        }
        try:
            kind = aliases[name]
        except KeyError as error:
            raise ValueError(f"unknown task function kind: {name!r}") from error
        return self._decorator(kind, batch=batch)

    def get(self, kind: str, *, batch: bool = False) -> Callable[..., Any] | None:
        return (self._batch if batch else self._single).get(kind)

    @classmethod
    def resolve(cls, name: str) -> TaskDefinition:
        try:
            return cls._registry[name]
        except KeyError as error:
            raise KeyError(
                f"task {name!r} is not registered; available: {sorted(cls._registry)}"
            ) from error

    @classmethod
    def clear_registry(cls) -> None:
        cls._registry.clear()

    def _validate(self, context: TaskContext) -> None:
        missing = [name for name in self.required_env_vars if not os.environ.get(name)]
        if missing:
            raise ValueError(
                "missing required task environment variables: " + ", ".join(missing)
            )
        if self.task_parameters_type is not None:
            context.parse_task_params(self.task_parameters_type)
        if self.get("inference") is None and self.get("inference", batch=True) is None:
            raise RuntimeError("task has no inference function")
        if (
            self.get("evaluation") is None
            and self.get("evaluation", batch=True) is None
        ):
            raise RuntimeError("task has no evaluation function")

    @staticmethod
    def _output(value: Any) -> TaskOutput:
        if isinstance(value, TaskOutput):
            return value
        if isinstance(value, BaseException):
            error = value if isinstance(value, Exception) else Exception(str(value))
            return TaskOutput(error=error)
        return TaskOutput(output=value)

    @staticmethod
    def _result(output: TaskOutput, value: Any) -> TaskResult:
        if isinstance(value, TaskResult):
            updates: dict[str, Any] = {}
            if output.error is not None and value.error is None:
                updates["error"] = str(output.error)
                updates["error_traceback"] = "".join(
                    traceback.format_exception(
                        type(output.error),
                        output.error,
                        output.error.__traceback__,
                    )
                )
            if output.execution_trace is not None and value.execution_trace is None:
                updates["execution_trace"] = output.execution_trace
            return value.model_copy(update=updates) if updates else value
        if isinstance(value, BaseException):
            return TaskResult.from_task_output(
                output,
                eval_error=str(value) or type(value).__name__,
                evaluation_error_traceback="".join(
                    traceback.format_exception(type(value), value, value.__traceback__)
                ),
            )
        raise TypeError(
            f"evaluation returned {type(value).__name__}, expected TaskResult"
        )

    async def _map(
        self,
        factories: Sequence[Callable[[], Awaitable[Any] | Any]],
        context: TaskContext,
    ) -> list[Any]:
        semaphore = asyncio.Semaphore(context.max_concurrency)

        async def run(factory: Callable[[], Awaitable[Any] | Any]) -> Any:
            async with semaphore:
                try:
                    async with asyncio.timeout(context.case_timeout_seconds):
                        return await _resolve(factory())
                except Exception as error:
                    return error

        return list(await asyncio.gather(*(run(factory) for factory in factories)))

    async def run(
        self,
        cases: Sequence[TaskT] | None,
        context: TaskContext,
    ) -> list[TaskResult]:
        self._validate(context)
        if cases is None:
            loader = self.get("load_data")
            if loader is None:
                raise ValueError(
                    "cases are required when the task has no load_data function"
                )
            cases = list(await _resolve(loader(context)))
        else:
            cases = list(cases)

        batch_inference = self.get("inference", batch=True)
        if batch_inference is not None:
            raw_outputs = list(await _resolve(batch_inference(cases, context)))
        else:
            inference = self.get("inference")
            assert inference is not None
            raw_outputs = await self._map(
                [lambda case=case: inference(case, context) for case in cases],
                context,
            )
        if len(raw_outputs) != len(cases):
            raise ValueError("inference result count does not match case count")
        outputs = [self._output(value) for value in raw_outputs]

        batch_evaluation = self.get("evaluation", batch=True)
        if batch_evaluation is not None:
            raw_results = list(
                await _resolve(batch_evaluation(cases, outputs, context))
            )
        else:
            evaluation = self.get("evaluation")
            assert evaluation is not None
            raw_results = await self._map(
                [
                    lambda case=case, output=output: evaluation(case, output, context)
                    for case, output in zip(cases, outputs)
                ],
                context,
            )
        if len(raw_results) != len(cases):
            raise ValueError("evaluation result count does not match case count")
        return [
            self._result(output, result)
            for output, result in zip(outputs, raw_results)
        ]


def create_task(
    name: str,
    *,
    register: bool = True,
    task_parameters_type: type | None = None,
    required_env_vars: Sequence[str] | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        name,
        register=register,
        task_parameters_type=task_parameters_type,
        required_env_vars=required_env_vars,
    )
