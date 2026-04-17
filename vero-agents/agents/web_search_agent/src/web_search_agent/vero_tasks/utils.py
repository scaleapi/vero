import json
import os
import re
from collections.abc import Callable
from enum import Enum
from typing import Literal, NamedTuple

from pydantic import BaseModel
from vero.core.db.result import TaskOutput
from vero.core.evaluation import EvaluationParameters, TaskParameters

from agents import Agent, Runner, RunResult
from agents.exceptions import MaxTurnsExceeded
from agents.extensions.models.litellm_model import LitellmModel
from agents.items import RunItemBase
from web_search_agent.agent import run_agent


class WebSearchParameters(TaskParameters):
    """Typed parameters for web search agent evaluation."""

    model: str = "gpt-4.1-mini-2025-04-14"


class TaskType(Enum):
    """Task types for grading."""

    SIMPLE_QA = "simple_qa"
    FACTS_SEARCH = "facts_search"
    HLE = "hle"


class GraderModel(Enum):
    """Grader models available for evaluation."""

    GPT41 = "gpt-4.1-2025-04-14"
    GEMINI_FLASH = "gemini/gemini-2.0-flash"
    O3 = "o3-mini-2025-01-31"


QA_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


HLE_GRADER_TEMPLATE = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {target}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {predicted_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]  # 100% reliability


class GraderConfig(NamedTuple):
    model: GraderModel
    prompt_template: str
    final_output_parser: Callable[[RunResult], tuple[float, str]]
    output_type: type[BaseModel] | None = None


def qa_final_output_parser(result: RunResult) -> tuple[float, str]:
    match = re.search(r"(A|B|C)", result.final_output)
    grade = match.group(0) if match else "C"
    if grade == "A":
        return 1.0, "CORRECT"
    elif grade == "B":
        return 0.0, "INCORRECT"
    else:
        return 0.0, "NOT_ATTEMPTED"


def hle_final_output_parser(result: RunResult) -> tuple[float, str]:
    output: ExtractedAnswer = result.final_output
    return float(output.correct == "yes"), output.reasoning


DEFAULT_GRADER_CONFIG = {
    TaskType.SIMPLE_QA: GraderConfig(
        model=GraderModel.GPT41,
        prompt_template=QA_GRADER_TEMPLATE,
        output_type=None,
        final_output_parser=qa_final_output_parser,
    ),
    TaskType.FACTS_SEARCH: GraderConfig(
        model=GraderModel.GEMINI_FLASH,
        prompt_template=HLE_GRADER_TEMPLATE,
        output_type=None,
        final_output_parser=qa_final_output_parser,
    ),
    TaskType.HLE: GraderConfig(
        model=GraderModel.O3,
        prompt_template=HLE_GRADER_TEMPLATE,
        output_type=ExtractedAnswer,
        final_output_parser=hle_final_output_parser,
    ),
}


async def grade_answer(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    task_type: TaskType = TaskType.FACTS_SEARCH,
) -> tuple[float, str]:
    """
    Grades the predicted answer using an LLM via LiteLLM.

    Args:
        question: The question being answered
        gold_answer: The ground truth answer
        predicted_answer: The model's predicted answer
        task_type: The task type (determines default grader config)

    Returns:
        Tuple of (score, feedback) where score is 1.0 for correct, 0.0 otherwise
    """
    grader_config = DEFAULT_GRADER_CONFIG[task_type]
    prompt = grader_config.prompt_template.format(
        question=question,
        target=gold_answer,
        predicted_answer=predicted_answer,
    )

    agent = Agent(
        name="GraderAgent",
        model=LitellmModel(
            model=grader_config.model.value,
            base_url=os.getenv("LITELLM_BASE_URL"),
            api_key=os.getenv("LITELLM_API_KEY"),
        ),
        output_type=grader_config.output_type,
    )

    try:
        result = await Runner.run(agent, input=prompt)
    except Exception as e:
        print(f"Error during grading: {e}")
        return 0.0, "GRADER_ERROR"

    score, feedback = grader_config.final_output_parser(result)
    return score, feedback


async def run_inference(
    task: dict,
    evaluation_parameters: EvaluationParameters,
) -> TaskOutput:
    """
    Run inference on a single task using the web search agent.

    Args:
        task: The task data (raw dict from the Dataset), must have "problem" key
        evaluation_parameters: Evaluation parameters

    Returns:
        TaskOutput with the agent's response and execution trace
    """
    question = task.get("problem") or task.get("question")
    params = evaluation_parameters.parse_task_params(WebSearchParameters)
    model_str = params.model

    # Always route through litellm proxy
    if not model_str.startswith("openai/"):
        model_str = f"openai/{model_str}"
    model = LitellmModel(
        model=model_str,
        base_url=os.getenv("LITELLM_BASE_URL"),
        api_key=os.getenv("LITELLM_API_KEY"),
    )

    try:
        result = await run_agent(question, model=model)
    except MaxTurnsExceeded as e:
        run_data = e.run_data
        input_items = run_data.input
        if not isinstance(input_items, list):
            input_items = [input_items]
        new_items = run_data.new_items
        execution_trace = input_items + new_items

        def serialize_run_item(item):
            if isinstance(item, RunItemBase):
                return item.to_input_item()
            try:
                json.dumps(item)
                return item
            except TypeError:
                return str(item)

        execution_trace = [serialize_run_item(item) for item in execution_trace]
        return TaskOutput(error=e, execution_trace=execution_trace)

    return TaskOutput(output=result.final_output, execution_trace=result.to_input_list())
