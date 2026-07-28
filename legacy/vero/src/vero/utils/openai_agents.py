from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, AsyncGenerator, Callable

from vero.exceptions import StreamEventTimeout
from vero.utils import anext_with_timeout

if TYPE_CHECKING:
    from agents import (
        Agent,
        OpenAIChatCompletionsModel,
        RunConfig,
        RunResultStreaming,
        StreamEvent,
        TResponseInputItem,
    )
    from agents.extensions.models.litellm_model import LitellmModel


logger = logging.getLogger(__name__)


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


DEFAULT_USER_MSG = "It seems like your last tool call got corrupted. Please think carefully and try again."


async def run_agent_with_json_sanitization(
    agent: Agent,
    input: list[TResponseInputItem] | str,
    max_turns: int = 10,
    event_timeout: int | None = None,
    run_config: RunConfig | None = None,
    sanitize_invalid_json: bool = True,
    max_retries: int = 3,
    add_user_reminder: bool = True,
    retry_delay: float = 1.0,
    retry_delay_multiplier: float = 2.0,
    user_msg: str = DEFAULT_USER_MSG,
    on_event: "Callable[[StreamEvent], None] | None" = None,
) -> tuple[RunResultStreaming | None, Exception | None]:
    """Run an agent with JSON sanitization for corrupt tool call arguments.

    Rate limit retries are handled by ModelSettings.retry on the agent —
    this function only retries on invalid JSON in tool calls.

    Args:
        agent: The agent to run.
        input: The input to the agent.
        max_turns: The maximum number of turns.
        event_timeout: Timeout in seconds for each event.
        run_config: The run config.
        sanitize_invalid_json: Whether to sanitize corrupt tool call JSON.
        max_retries: Max retries for JSON sanitization.
        add_user_reminder: Whether to add a user message after sanitization.
        retry_delay: Initial delay between retries.
        retry_delay_multiplier: Multiplier for delay.
        user_msg: Message to add after sanitization.
        on_event: Callback for each event.

    Returns:
        A tuple of (result, error).
    """

    from agents import Runner
    from litellm.exceptions import BadRequestError

    def is_invalid_json_error(e: Exception) -> bool:
        return isinstance(
            e, BadRequestError
        ) and "AnthropicException - Expecting ',' delimiter" in str(e)

    def sanitize_input_list(
        input: list[dict | TResponseInputItem],
    ) -> list[dict | TResponseInputItem]:
        counter = 0
        for item in input:
            if "arguments" in item:
                try:
                    json.loads(item["arguments"])  # type: ignore
                except json.JSONDecodeError:
                    item["arguments"] = "{}"
                    counter += 1
        logger.info(f"Sanitized {counter} function call arguments in the input list.")

        if add_user_reminder and counter > 0:
            input.append({"role": "user", "content": user_msg})

        return input

    current_delay = retry_delay
    result = None
    error = None

    for _ in range(max_retries):
        result = Runner.run_streamed(
            agent, max_turns=max_turns, input=input, run_config=run_config
        )
        error = None

        try:
            async for event in stream_events(
                result,
                event_timeout=event_timeout,
            ):
                if on_event is not None:
                    on_event(event)
                await asyncio.sleep(0)
        except Exception as e:
            if sanitize_invalid_json and is_invalid_json_error(e):
                logger.warning(
                    f"Model produced invalid JSON for tool call. Sanitizing and retrying after {current_delay}s..."
                )
                input = sanitize_input_list(result.to_input_list())  # type: ignore
                await asyncio.sleep(current_delay)
                current_delay *= retry_delay_multiplier
                error = e
                continue

            error = e
            break

        return result, error

    # Retries exhausted
    return result, error


def strict_mode_from_model(
    model: str | OpenAIChatCompletionsModel | LitellmModel,
) -> bool:
    """Gets the strict mode of the tool schema from the model name."""

    if not isinstance(model, str):
        model = model.model

    return (
        "/" not in model or model.startswith("openai") or model.startswith("anthropic")
    )
