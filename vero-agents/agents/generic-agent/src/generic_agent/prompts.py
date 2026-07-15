from collections.abc import Callable
from typing import Any


PromptFormatter = Callable[..., str]


def get_gsm8k_prompt(question: str, **kwargs: Any) -> str:
    """Format prompt for GSM8K benchmark."""
    GSM8K_DEFAULT_PROMPT = """{question}
Generate an answer to this question. At the end, provide the final answer in the format "Answer is <number>", where <number> is a single number."""
    return GSM8K_DEFAULT_PROMPT.format(question=question)


def get_math_prompt(question: str, **kwargs: Any) -> str:
    """Format prompt for MATH benchmark."""
    MATH_DEFAULT_PROMPT = """{question}
Please generate a solution for the problem. At the end, provide the final answer in the format "\\boxed{{<number>}}", where <number> is a math answer(an expression or number), without any additional information or explanation."""
    return MATH_DEFAULT_PROMPT.format(question=question)


def get_hotpotqa_prompt(question: str, context: str, **kwargs: Any) -> str:
    """Format prompt for HotpotQA benchmark."""
    HOTPOTQA_DEFAULT_PROMPT = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    return HOTPOTQA_DEFAULT_PROMPT.format(question=question, context=context)


def get_drop_prompt(question: str, passage: str, **kwargs: Any) -> str:
    """Format prompt for DROP benchmark."""
    DROP_DEFAULT_PROMPT = """Given a question and a passage, please answer the question.
1. In the "thought" field, explain your thinking process.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Question: {question}
The relevant passage: {passage}"""
    return DROP_DEFAULT_PROMPT.format(question=question, passage=passage)


def get_humaneval_prompt(question: str, **kwargs: Any) -> str:
    """Format prompt for HumanEval benchmark."""
    HUMANEVAL_DEFAULT_PROMPT = """{question}
Generate an answer to this question, without any additional test cases."""
    return HUMANEVAL_DEFAULT_PROMPT.format(question=question)


def get_mbpp_prompt(question: str, test_list: str, **kwargs: Any) -> str:
    """Format prompt for MBPP benchmark."""
    MBPP_DEFAULT_PROMPT = """You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:\n\n{test_list}\n"""
    return MBPP_DEFAULT_PROMPT.format(question=question, test_list=test_list)


def get_gpqa_prompt(question: str, options: str, **kwargs: Any) -> str:
    """Format prompt for GPQA benchmark."""
    GPQA_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: ‘Answer: $LETTER’ (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""
    LETTERS = "ABCD"
    kwargs = dict(zip(LETTERS, options, strict=False))
    kwargs["question"] = question
    return GPQA_TEMPLATE.format(**kwargs)


def get_gaia_prompt(question: str, file_path: str | None = None, **kwargs: Any) -> str:
    """
    Format prompt for GAIA benchmark.

    Args:
        question: The question to answer
        file_path: Optional file path

    Returns:
        Prompt string
    """
    prompt = f"Question: {question}\n"

    if file_path:
        prompt += "\nAttached files:\n"
        prompt += f"- {file_path}\n"

    prompt += "\nProvide ONLY the answer, no explanation."
    return prompt


def get_default_prompt(**kwargs: Any) -> str:
    """
    Default formatting function.

    If a 'prompt' key exists, return it directly.
    Otherwise, format as "key: value" pairs separated by newlines.
    """
    if "prompt" in kwargs:
        return str(kwargs["prompt"])
    return "\n".join(f"{key}: {value}" for key, value in kwargs.items())


PROMPT_REGISTRY: dict[str, PromptFormatter] = {
    "gsm8k": get_gsm8k_prompt,
    "math": get_math_prompt,
    "hotpotqa": get_hotpotqa_prompt,
    "drop": get_drop_prompt,
    "humaneval": get_humaneval_prompt,
    "mbpp": get_mbpp_prompt,
    "gpqa": get_gpqa_prompt,
    "default": get_default_prompt,
    "gaia": get_gaia_prompt,
}


def format_prompt(task_inputs: dict[str, Any], prompt_name: str | None = None) -> str:
    """
    Format prompt components using the specified template.

    Args:
        task_inputs: Dictionary of inputs that can be leveraged in the agent inference logic.
        prompt_name: The name of the prompt to use.

    Returns:
        The formatted prompt string.
    """
    prompt_name = prompt_name or "default"
    formatter = PROMPT_REGISTRY.get(prompt_name, get_default_prompt)
    return formatter(**task_inputs)
