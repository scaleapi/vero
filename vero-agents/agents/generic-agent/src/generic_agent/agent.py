import asyncio
import os
from typing import Any, TypeVar

from pydantic import BaseModel

from agents import Agent, ModelSettings, Runner, RunResultStreaming, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from generic_agent.prompts import format_prompt

set_tracing_disabled(True)

T = TypeVar("T", bound=BaseModel)
default_model = LitellmModel(
    model="openai/gpt-4.1-mini-2025-04-14",
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
)


# More info at: https://openai.github.io/openai-agents-python/tools/
@function_tool(strict_mode=True)
async def example_async_tool(x: str) -> str:
    """
    Example async tool.

    Args:
        x (str): The input string.

    Returns:
        str: The input string.
    """
    await asyncio.sleep(2)
    return str(x)


@function_tool(strict_mode=True)
async def example_blocking_sync_tool(x: str) -> str:
    """
    Example of how to run sync code to avoid blocking the event loop.

    Args:
        x (str): The input string.

    Returns:
        str: The input string.
    """
    import time

    await asyncio.to_thread(time.sleep, 2)
    return str(x)


def get_temperature(model: str | LitellmModel) -> float | None:
    """Get the temperature for the given model."""
    if isinstance(model, LitellmModel):
        model = model.model

    # reasoning models don't support temperature == 0.0
    if "gpt-5" in model or "o3" in model:
        return None
    return 0.0


async def run_agent(
    task_inputs: dict[str, Any],
    task_name: str | None = None,
    output_type: type[T] | None = None,
    model: str | LitellmModel = default_model,
    max_turns: int = 20,
) -> RunResultStreaming:
    """Run the LLM on a given prompt or prompt components after formatting.

    Args:
        task_inputs: Dictionary of inputs that can be leveraged in the agent inference logic.
        task_name: The name of the task
        output_type: The structured output type to parse the agent's response into.
        model: The model to use for inference.
        max_turns: Maximum number of turns for the agent.

    Returns:
        A RunResultStreaming object.
    """
    prompt = format_prompt(task_inputs, task_name)

    agent = Agent(
        name=task_name.title() + "Agent",
        instructions=None,
        tools=[],
        model=model,
        model_settings=ModelSettings(temperature=get_temperature(model)),
        output_type=output_type,
    )

    result = Runner.run_streamed(agent, input=prompt, max_turns=max_turns)
    return result
