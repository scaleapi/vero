"""Seed target harness for the conformance example.

Deliberately incomplete: it answers addition tasks and gives up on multiplication
ones, so the seed scores about 0.5 and a one-line fix takes it to 1.0. That gap is
the point -- it makes the optimizer's edit -> evaluate -> submit loop observable in
the score rather than something we have to infer.

The model call is not optional decoration either: the answer has to come back from
the model, so an unreachable or mis-allow-listed inference gateway shows up as a
zero score instead of passing silently.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI

ANSWER_PATH = "/app/answer.txt"

INSTRUCTIONS = """You answer a single arithmetic question.
Reply with the numeric result and nothing else: no words, units, or punctuation.
"""


# A model told to answer with a bare number may still reason in prose first, so
# take the last number in the completion rather than trusting the raw text.
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class ConformanceAgent(BaseAgent):
    """One model call, one file written. Nothing else."""

    # BaseAgent declares these abstract; omitting them fails at instantiation with
    # "Can't instantiate abstract class" and every case reports an infrastructure
    # failure, not a low score.
    @staticmethod
    @override
    def name() -> str:
        return "conformance-agent"

    @override
    def version(self) -> str | None:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("conformance agent requires a model name")
        # The gateway matches the requested model as an exact string, and the
        # evaluation scope is allow-listed with the unprefixed name.
        self._api_model = self.model_name.removeprefix("openai/")
        # base_url and api_key come from OPENAI_BASE_URL / OPENAI_API_KEY, which
        # vero points at the metered evaluation scope of the inference gateway.
        self._client = AsyncOpenAI(max_retries=4)

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec("mkdir -p /app", timeout_sec=30)

    async def _write_answer(self, environment: BaseEnvironment, answer: str) -> None:
        quoted = shlex.quote(answer.strip())
        await environment.exec(
            f"printf '%s' {quoted} > {ANSWER_PATH}", cwd="/app", timeout_sec=30
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # THE DELIBERATE GAP: multiplication tasks are abandoned unanswered.
        # Removing this early return is the fix the optimizer is meant to find.
        if "*" in instruction:
            await self._write_answer(environment, "")
            return

        completion = await self._client.chat.completions.create(
            model=self._api_model,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": instruction},
            ],
            max_tokens=256,
        )
        content = (completion.choices[0].message.content or "").strip()
        numbers = NUMBER_RE.findall(content)
        await self._write_answer(environment, numbers[-1] if numbers else content)
