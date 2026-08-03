"""Bounded-concurrency async client for structured labelling.

Deliberately thin. Labelling is thousands of small independent calls against a cheap
model, so what matters is that concurrency is capped, transient failures are retried,
and a malformed response is retried rather than parsed leniently — a label that
silently degrades to a default is worse than one that is missing, because it looks
like evidence.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from vero.interpret.config import Settings


class LLMError(RuntimeError):
    pass


class AsyncLLM:
    def __init__(self, settings: Settings) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError(
                "the interpret extra is required: uv sync --extra interpret"
            ) from exc

        self.settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout,
            max_retries=0,          # retried here, so backoff is visible and uniform
        )
        self._sem = asyncio.Semaphore(settings.concurrency)
        self.calls = 0
        self.retries = 0

    async def json_call(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str = "label",
    ) -> dict[str, Any]:
        """One structured call, retried on transport error and on schema violation."""
        last: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                async with self._sem:
                    self.calls += 1
                    response = await self._client.chat.completions.create(
                        model=self.settings.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "schema": schema,
                                "strict": True,
                            },
                        },
                    )
                content = response.choices[0].message.content or ""
                return json.loads(content)
            except Exception as exc:  # noqa: BLE001 - retry policy is uniform
                last = exc
                self.retries += 1
                if attempt == self.settings.max_retries - 1:
                    break
                # Jittered backoff: these run thousands-wide, and synchronised
                # retries after a rate-limit burst just reproduce the burst.
                await asyncio.sleep((2**attempt) + random.random())
        raise LLMError(f"call failed after {self.settings.max_retries} attempts: {last}")

    async def close(self) -> None:
        await self._client.close()
