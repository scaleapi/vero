from openai import AsyncOpenAI

client = AsyncOpenAI()

SYSTEM_PROMPT = """Think step by step and answer the question. Your solution should end with "Final answer: X". """


async def gsm8k_agent(
    messages: list[dict[str, str]] | str,
    model: str = "gpt-4o",
    instructions: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    system_message = [{"role": "system", "content": instructions}]
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    assert messages, "input cannot be empty"
    response = await client.chat.completions.create(
        model=model, messages=system_message + messages, temperature=0.0
    )
    reply = {"role": "assistant", "content": response.choices[0].message.content}
    return messages + [reply]
