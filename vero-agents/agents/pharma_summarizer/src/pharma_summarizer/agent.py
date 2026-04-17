from openai.types.shared.reasoning import Reasoning

from agents import Agent, ModelSettings, Runner, RunResult

# Do not change the reasoning effort! Required for latency constraints.
REQUIRED_REASONING_EFFORT = "low"


async def run_agent(prompt: str) -> RunResult:
    agent = Agent(
        name="SummarizationAgent",
        instructions="You are a summarization assistant. Given a section of text, produce a summary.",
        tools=[],
        model="gpt-5",
        model_settings=ModelSettings(reasoning=Reasoning(effort=REQUIRED_REASONING_EFFORT, summary="detailed")),
    )
    result = await Runner.run(agent, input=prompt)
    return result
