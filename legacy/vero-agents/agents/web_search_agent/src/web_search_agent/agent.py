import os

from vero.core.resource import resource

from agents import Agent, ModelSettings, Runner, RunResult, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from web_search_agent.search import get_page, search

# Required model for evaluation
default_model = LitellmModel(
    model="openai/gpt-4.1-mini-2025-04-14",
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
)


@resource(namespace="web_search_agent", name="search_tool_description")
def search_tool_description() -> str | None:
    return None


@resource(namespace="web_search_agent", name="wikipedia_page_tool_description")
def wikipedia_page_tool_description() -> str | None:
    return None


@resource(namespace="web_search_agent", name="wikipedia_search_tool")
@function_tool(strict_mode=True, description_override=search_tool_description())
async def wikipedia_search_tool(query: str, results: int = 10) -> list[str]:
    """
    Searches Wikipedia based on the given query and returns the search results.

    Args:
        query (str): The search query for Wikipedia.
        results (int): The number of results to return.

    Returns:
        list[str]: The search results.
    """
    search_results = await search(query=query, results=results)
    return search_results


@resource(namespace="web_search_agent", name="get_wikipedia_page_tool")
@function_tool(strict_mode=True, description_override=wikipedia_page_tool_description())
async def get_wikipedia_page_tool(page_title: str) -> str:
    """
    Gets the content of a Wikipedia page.

    Args:
        page_title (str): The title of the Wikipedia page.

    Returns:
        str: The content of the Wikipedia page.
    """
    return await get_page(title=page_title, auto_suggest=False, redirect=True)


def get_temperature(model: str | LitellmModel) -> float | None:
    """Get the temperature for the given model."""
    if isinstance(model, LitellmModel):
        model = model.model

    # reasoning models don't support temperature == 0.0
    if "gpt-5" in model or "o3" in model:
        return None
    return 0.0


@resource(namespace="web_search_agent", name="get_system_prompt")
def get_system_prompt(input: str) -> str:
    """Get the system prompt for the web search agent."""

    return """You are a search assistant designed to find accurate information through multi-step searches.
                Guidelines:
                1. You can perform multiple search queries across multiple hops/turns to gather comprehensive information
                2. Each hop allows multiple parallel searches, with the hop ending once you receive all search results
                3. Your final answer should be:
                   - Concise, short and accurate
                   - Based only on facts from the search results
                   - Answer with only one short sentence
                4. Do not provide partial or intermediate answers during the search process
                5. If information is incomplete, continue searching in the next hop"""


async def run_agent(input: str, model: str | LitellmModel = default_model) -> RunResult:
    """Run the wiki agent with the given prompt and model using the Wikipedia search and page retrieval tools."""

    agent = Agent(
        name="WebSearchAgent",
        instructions=get_system_prompt(input),
        tools=[wikipedia_search_tool, get_wikipedia_page_tool],
        model=model,
        model_settings=ModelSettings(temperature=get_temperature(model)),
    )

    result = await Runner.run(agent, input=input, max_turns=20)
    return result
