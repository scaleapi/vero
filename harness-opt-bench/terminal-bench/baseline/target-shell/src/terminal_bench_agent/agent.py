"""Skeleton Harbor agent for Terminal-Bench.

This module defines the agent class the benchmark loads and nothing else. The
Harbor interface is satisfied, the model is resolved, and an OpenAI client is
constructed, so the process starts and every case runs to completion. `run` does
not attempt the task: it issues no commands and returns.

Grading runs each task's own tests against the container's final state, so an
untouched container scores whatever those tests give an unmodified environment,
which is zero for every task in the pinned partitions.
"""

from __future__ import annotations

from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI


class TerminalBenchAgent(BaseAgent):
    """Terminal-Bench agent whose source is the editable optimization target."""

    @staticmethod
    @override
    def name() -> str:
        return "terminal-bench-skeleton"

    @override
    def version(self) -> str:
        return "0.0.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("Terminal-Bench agent requires a Harbor model")
        # Harbor hands the model through with a provider prefix; the API takes
        # the bare name. The gateway allow-lists exactly the bare form, so a
        # request that keeps the prefix is denied.
        self._api_model = self.model_name.removeprefix("openai/")
        # Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment, which
        # point at the metered inference gateway rather than the vendor.
        # max_retries absorbs transient 429s in-client: an unretried rate limit
        # inside a trial scores at the failure value.
        self._client = AsyncOpenAI(max_retries=8)

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec("mkdir -p /app", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "could not prepare /app")

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Not implemented. Nothing is run in the container, so the task's tests
        # grade an untouched environment; the run returns without consulting
        # the model.
        context.n_input_tokens = 0
        context.n_output_tokens = 0
        context.n_cache_tokens = 0
