from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

from litellm import acompletion


@dataclass
class EvalResult:
    """Result of evaluating an agent's answer on the GAIA dataset."""

    question: str
    expected_answer: str
    agent_answer: str
    is_correct: bool
    reasoning: str
    extracted_answer: str | None = None
    elapsed_time: float = 0.0
    error: str | None = None


def _truncate_to_last_chars(text: str, max_chars: int) -> str:
    """Truncate text to approximately the last N chars (1 chars ≈ 1 token)."""
    if not max_chars or max_chars <= 0:
        return text

    if len(text) > max_chars:
        return "..." + text[-max_chars:]
    return text


def _build_judge_prompt(question: str, expected_answer: str, agent_response: str) -> str:
    """Build the judge prompt for evaluating an agent's answer."""
    return f"""You are an expert evaluator. Your task is to determine if the agent's answer is correct by following a two-step process.

Question: {question}

Expected Answer: {expected_answer}

Agent's Response: {agent_response}

## Step 1: Extract the Answer
First, carefully read the agent's response and extract the specific answer to the question. The agent's response may contain reasoning, context, or other information - identify and extract only the core answer that directly addresses the question.

If the agent explicitly states it cannot answer, doesn't know, or fails to provide an answer, set extracted_answer to null.

## Step 2: Compare with Expected Answer
Compare your extracted answer with the expected answer:
- The extracted answer is CORRECT if it is factually equivalent to the expected answer
- Minor formatting differences are OK (e.g., "1945" vs "In 1945", "42" vs "42.0")
- Semantic equivalence matters, not exact string matching
- If extracted_answer is null (agent failed to answer), mark as INCORRECT

Respond with JSON:
{{"extracted_answer": "the specific answer extracted from agent's response (or null if none)", "is_correct": true/false, "reasoning": "brief explanation of extraction and comparison"}}"""


async def judge_answer_async(
    question: str,
    expected_answer: str,
    agent_answer: str,
    model: str = "openai/gpt-4.1-2025-04-14",
    max_eval_chars: int = 1024,
    semaphore: asyncio.Semaphore | None = None,
) -> EvalResult:
    """
    Async version of judge_answer for concurrent evaluation.

    Args:
        question: The original question
        expected_answer: The ground truth answer
        agent_answer: The agent's response
        model: LiteLLM model string for the judge
        max_eval_chars: Maximum chars from agent_answer to evaluate (uses last N chars).
                         Set to None or 0 to disable truncation. Default: 1024
        semaphore: Optional semaphore for concurrency control

    Returns:
        EvalResult with is_correct, reasoning, and extracted_answer
    """
    eval_answer = _truncate_to_last_chars(agent_answer, max_eval_chars)
    judge_prompt = _build_judge_prompt(question, expected_answer, eval_answer)

    async def _call_llm():
        return await acompletion(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            api_base=os.getenv("LITELLM_BASE_URL"),
            api_key=os.getenv("LITELLM_API_KEY", os.getenv("OPENAI_API_KEY")),
        )

    try:
        if semaphore:
            async with semaphore:
                response = await _call_llm()
        else:
            response = await _call_llm()

        result = json.loads(response.choices[0].message.content)

        return EvalResult(
            question=question,
            expected_answer=expected_answer,
            agent_answer=agent_answer,
            is_correct=result.get("is_correct", False),
            reasoning=result.get("reasoning", ""),
            extracted_answer=result.get("extracted_answer"),
        )
    except Exception as e:
        return EvalResult(
            question=question,
            expected_answer=expected_answer,
            agent_answer=agent_answer,
            is_correct=False,
            reasoning=f"Evaluation error: {e!s}",
            error=str(e),
        )
