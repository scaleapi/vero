from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncGenerator

from vero.exceptions import StreamEventTimeout
from vero.utils import anext_with_timeout

if TYPE_CHECKING:
    from agents import OpenAIChatCompletionsModel, RunResultStreaming, StreamEvent
    from agents.extensions.models.litellm_model import LitellmModel


async def stream_events(
    result: RunResultStreaming,
    event_timeout: int | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream events from an OpenAI agent run.

    Args:
        result: The RunResultStreaming object to stream events from.
        event_timeout: Timeout in seconds for each event. None means no timeout.
    """

    event_iter = aiter(result.stream_events())

    while True:
        try:
            if event_timeout is not None:
                event = await anext_with_timeout(event_iter, event_timeout)
            else:
                event = await anext(event_iter)
            yield event
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            result.cancel(mode="immediate")
            await asyncio.sleep(0.1)
            raise StreamEventTimeout(
                f"Timeout of {event_timeout} seconds reached for an event on iterator {event_iter}!"
            )


def strict_mode_from_model(
    model: str | OpenAIChatCompletionsModel | LitellmModel,
) -> bool:
    """Gets the strict mode of the tool schema from the model name."""

    if not isinstance(model, str):
        model = model.model

    return (
        "/" not in model or model.startswith("openai") or model.startswith("anthropic")
    )
