import asyncio
from logging import getLogger

import numpy as np
from openai import AsyncOpenAI

from agents import Agent, Runner

client = AsyncOpenAI()
logger = getLogger(__name__)


def word_length_deviation_grader(summary: str) -> float:
    word_count = len(summary.split())

    expected_summary_length = 100
    tolerance = 0.2  # 20% band around target

    # relative deviation
    deviation = abs(word_count - expected_summary_length) / expected_summary_length

    # If within tolerance band → full score
    if deviation <= tolerance:
        return 1.0

    # Outside band → score decays linearly, capped at 0
    # e.g., deviation 0.3 → score 0.8, deviation 1.0+ → 0.0
    score = 1.0 - (deviation - tolerance)
    return max(0.0, score)


def chemical_name_grader(summary: str, section: str) -> float:
    CHEMICALS_MASTER = [
        "[1-¹³C]Pyruvic acid",
        "[1-¹³C]Pyruvate",
        "¹²C Pyruvic acid",
        "Sodium [1-¹³C]pyruvate",
        "Sodium pyruvate (¹²C)",
        "AH111501 (Trityl radical)",
        "Tris{8-carboxyl-2,2,6,6-tetra[2-(1-methoxyethyl)]-benzo(1,2-d:4,5-d')bis(1,3)dithiole-4-yl}methyl acid",
        "AH111501 sodium salt",
        "Methyl, tris[8-carboxy-2,2,6,6-tetrakis(2-methoxyethyl)benzo[1,2-d:4,5-d']bis[1,3]dithiol-4-yl]-, trisodium salt",
        "AH111501 trisodium salt",
        "AH111576",
        "2,2′,2″,2‴-(4,8-Dibromobenzo[1,2-d:4,5-d′]bis([1,3]dithiole)-2,2,6,6-tetrayl)tetraethanol",
        "AH111586",
        "4,8-Dibromo-2,2,6,6-tetrakis(2-methoxyethyl)benzo[1,2-d:4,5-d′]bis([1,3]dithiole)",
        "AH111709",
        "AH111743",
        "AH112615",
        "4,4-Bis-hydroxymethyl-2-methyl-oxazolidine-2-carboxylic acid",
        "AH112623",
        "Parapyruvate",
        "2-Hydroxy-2-methyl-4-oxo-pentanedioic acid",
        "AH113127",
        "(4-Hydroxymethyl-oxazolidin-4-yl)-methanol",
        "AH113462/E",
        "Enol lactone",
        "AH113462/K",
        "Keto lactone",
        "Acetyl bromide",
        "Methanol",
        "Dimethyl sulfoxide",
        "DMSO",
        "Tetrahydrofuran",
        "THF",
        "Acetonitrile",
        "ACN",
        "Diethyl ether",
        "Et₂O",
        "N,N-Dimethylacetamide",
        "DMA",
        "1,3-Dimethyl-2-imidazolidinone",
        "DMI",
        "Hydrochloric acid",
        "HCl",
        "Sodium hydroxide",
        "NaOH",
        "Disodium ethylenediaminetetraacetate",
        "Na₂EDTA",
        "Ethylenediaminetetraacetic acid",
        "EDTA",
        "Tris(hydroxymethyl)aminomethane",
        "TRIS",
        "Trometamol",
        "Trifluoroacetic acid",
        "TFA",
        "Toluene",
        "Heptane",
        "Ethyl acetate",
        "Ethanol",
        "Water",
        "H₂O",
        "Sodium chloride",
        "NaCl",
        "Cuprous [1-¹³C]cyanide",
        "Cu¹³CN",
        "Gadolinium",
        "Gd",
        "Tin",
        "Sn",
        "Phosphorus",
        "P",
        "Carbon dioxide",
        "CO₂",
        "Sodium [1-13C]pyruvate",
        "[1-13C]Pyruvic acid",
        "1-13C pyruvate",
    ]

    # Identify the chemicals present in the section
    present = [chem for chem in CHEMICALS_MASTER if chem in section]

    # If no chemicals present, consider it satisfied
    if not present:
        return 1.0

    correct = 0
    for chem in present:
        # Only count as correct if the exact chemical string appears in the summary
        if chem in summary:
            correct += 1

    return correct / len(present)


async def compute_cosine_similarity(summary: str, section: str) -> float:
    embeddings = await client.embeddings.create(model="text-embedding-3-large", input=[summary, section])
    summary_embedding = np.array(embeddings.data[0].embedding)
    section_embedding = np.array(embeddings.data[1].embedding)
    similarity = np.dot(summary_embedding, section_embedding) / (
        np.linalg.norm(summary_embedding) * np.linalg.norm(section_embedding)
    )
    return similarity


async def llm_as_judge(summary: str, section: str) -> float:
    prompt = """\
    You are an expert technical summarization evaluator.
    Evaluate whether the summary captures and preserves the important technical facts and specific details from the section, allowing for occasional minor rewording or omissions of less important points, but not major technical inaccuracies or information loss.\n\n
    Scoring Guidelines:
    - Return a numerical score between 0 and 1 (with up to two decimal places).
    - A score of 1 means the summary is almost flawless: it is comprehensive, highly faithful, and technically accurate, with virtually no important or meaningful details missing, and no significant misstatements or distortions.
    - 0.75-0.99 indicates excellent work: all main facts are represented, but there may be trivial omissions or very minor rewording that do not materially affect understanding.
    - 0.5-0.75 indicates good but imperfect: most technical information is retained and correctly presented, some less critical details might be missing or slightly rephrased, but overall fidelity is preserved.
    - 0.3-0.5 means significant information is missing, or some technical inaccuracies are present, but the summary retains a reasonable portion of key facts.
    - 0.0-0.3 means there are major omissions, misunderstandings, or a failure to capture the most important technical content.\n\n
    Respond only with a single number between 0 and 1 indicating summary quality by these criteria."""

    judge_agent = Agent(name="JudgeAgent", instructions=prompt, tools=[], model="gpt-5")
    response = await Runner.run(judge_agent, input=f"Summary: {summary}\n\nSection: {section}")

    try:
        score = float(response.final_output)
    except ValueError:
        logger.error(f"Failed to convert response to float: {response.final_output}")
        return 0.0
    return score


async def evaluate_summary(summary: str, section: str) -> tuple[float, str]:
    """
    Evaluate the summary using the following metrics:
    - Word length deviation
    - Chemical name presence
    - Cosine similarity
    - LLM as judge
    Returns: float: 1 if the summary is good, 0 if the summary is bad
    """

    word_length_deviation_grade = word_length_deviation_grader(summary)
    chemical_name_grade = chemical_name_grader(summary, section)
    cosine_similarity_grade, llm_as_judge_grade = await asyncio.gather(
        compute_cosine_similarity(summary, section), llm_as_judge(summary, section)
    )

    passed = (
        (word_length_deviation_grade >= 0.85)
        and (chemical_name_grade >= 0.8)
        and (cosine_similarity_grade >= 0.85)
        and (llm_as_judge_grade >= 0.85)
    )
    feedback = f"""

    Raw Scores:
    Word length deviation grade: {word_length_deviation_grade}
    Chemical name grade: {chemical_name_grade}
    Cosine similarity grade: {cosine_similarity_grade}
    LLM as judge grade: {llm_as_judge_grade}

    Required Thresholds:
    Word length deviation: 0.85
    Chemical name: 0.8
    Cosine similarity: 0.85
    LLM as judge: 0.85

    Passed: {passed}
    """
    return float(passed), feedback
