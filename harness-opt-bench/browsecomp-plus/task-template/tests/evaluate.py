"""BrowseComp-Plus semantic-answer verifier.

The judge prompt and decision rule follow the pinned upstream OpenAI evaluator:
scripts_evaluation/evaluate_with_openai.py at commit 0469490.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from openai import OpenAI, OpenAIError

GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
"""


def _write_reward(value: float) -> None:
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(str(value), encoding="utf-8")


def main() -> int:
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    answer_path = Path("/app/answer.txt")
    if not answer_path.is_file():
        _write_reward(0.0)
        print("FAIL: /app/answer.txt is missing")
        return 0
    response = answer_path.read_text(encoding="utf-8").strip()
    if not response:
        _write_reward(0.0)
        print("FAIL: /app/answer.txt is empty")
        return 0

    config = json.loads(Path("/tests/config.json").read_text(encoding="utf-8"))
    prompt = GRADER_TEMPLATE.format(
        question=config["question"],
        response=response[:100_000],
        correct_answer=config["expected_answer"],
    )
    model = os.environ.get("BROWSECOMP_JUDGE_MODEL", "gpt-4.1")
    try:
        judged = OpenAI().responses.create(
            model=model,
            input=prompt,
            max_output_tokens=1024,
        )
        judge_text = judged.output_text or ""
        match = None
        for pattern in (
            r"\*\*correct:\*\*\s*(yes|no)",
            r"\*\*correct\*\*:\s*(yes|no)",
            r"(?im)^correct:\s*(yes|no)\b",
        ):
            match = re.search(pattern, judge_text, re.IGNORECASE)
            if match is not None:
                break
        correct = match is not None and match.group(1).lower() == "yes"
        (logs / "judge.json").write_text(
            json.dumps(
                {
                    "query_id": config["query_id"],
                    "judge_model": model,
                    "parse_error": match is None,
                    "correct": correct,
                    "judge_response": judge_text,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except (OpenAIError, OSError, TypeError, ValueError) as error:
        _write_reward(0.0)
        print(f"FAIL: judge request failed: {error}")
        return 0

    _write_reward(1.0 if correct else 0.0)
    print("PASS" if correct else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
