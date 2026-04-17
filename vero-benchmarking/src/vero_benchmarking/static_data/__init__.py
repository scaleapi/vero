"""
This module provides paths to static JSON files used for dataset building and filtering.
"""

from pathlib import Path

STATIC_DATA_DIR = Path(__file__).parent

# SimpleQA filter indices
SIMPLEQA_UNANSWERED_INDICES = STATIC_DATA_DIR / "unanswered_indices.json"
SIMPLEQA_GPT41_MINI_UNANSWERED_INDICES = STATIC_DATA_DIR / "gpt4.1_mini_unanswered_indices.json"

# AFLOW dataset mappings
AFLOW_TO_HF_MAPPINGS = STATIC_DATA_DIR / "aflow_to_hf_mappings.json"

# Tau Bench retail results
TAU_BENCH_RETAIL_RESULTS = STATIC_DATA_DIR / "tau-bench-gpt-41-results.csv"

# GAIA pure language mappings
GAIA_PURE_LANGUAGE_MAPPINGS = STATIC_DATA_DIR / "gaia_pure_language_mappings.json"

__all__ = [
    "STATIC_DATA_DIR",
    "SIMPLEQA_UNANSWERED_INDICES",
    "SIMPLEQA_GPT41_MINI_UNANSWERED_INDICES",
    "AFLOW_TO_HF_MAPPINGS",
    "TAU_BENCH_RETAIL_RESULTS",
    "GAIA_PURE_LANGUAGE_MAPPINGS",
]
