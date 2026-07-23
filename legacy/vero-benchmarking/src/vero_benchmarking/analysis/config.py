"""Configuration for interpret module."""

from vero_benchmarking.constants import DEFAULT_RESULTS_DIR

# Output directories
FIGURES_DIR = DEFAULT_RESULTS_DIR / "interpret_figures"
EMBEDDINGS_CACHE_DIR = DEFAULT_RESULTS_DIR / "embeddings_cache"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default scaffolds to include in plots and analysis (VeRO only, exclude Claude Code)
DEFAULT_SCAFFOLDS = {"vero-cookbook", "vero-orchestrator-cookbook", "vero-prompts-only"}

# Scaffold display aliases
SCAFFOLD_ALIASES = {
    "vero-cookbook": "VeRO Default",
    "vero-orchestrator-cookbook": "VeRO Orchestrator",
    "vero-prompts-only": "VeRO Prompts-Only",
    "claude-code-pure": "Claude Code",
    "claude-code-vmf-cookbook": "Claude Code + VeRO",
}

# Model display aliases
MODEL_ALIASES = {
    "claude-sonnet-4-5": "Sonnet 4.5",
    "claude-opus-4-5": "Opus 4.5",
    "gpt-5-2-codex": "GPT-5.2 Codex",
}

# Task display aliases
TASK_ALIASES = {
    "math": "MATH",
    "retail": "TauBench-Retail",
    "gaia": "GAIA",
    "simple_qa": "SimpleQA",
    "gpqa": "GPQA",
}
