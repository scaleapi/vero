import os
from pathlib import Path
from typing import Any


def get_path_to_vero_agents(env_var: str = "VERO_AGENTS_PATH") -> Path:
    """Gets the path to the vero-agents repository.

    Default: ../vero-agents relative to this package (monorepo layout).
    Override with VERO_AGENTS_PATH env var.
    """
    default = Path(__file__).resolve().parent.parent.parent.parent / "vero-agents"
    path = Path(os.getenv(env_var, default))
    assert path.exists(), f"Path to vero-agents does not exist: {path}"
    assert path.is_dir(), f"Path to vero-agents is not a directory: {path}"
    return path


def get_model(model: str | None) -> Any:
    """Gets the input model object for an OpenAI Agents SDK Agent from a model string."""
    if model is None:
        return None

    from agents.extensions.models.litellm_model import LitellmModel
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI

    if "/" not in model:
        return model

    base_url = os.getenv("LITELLM_BASE_URL")
    api_key = os.getenv("LITELLM_API_KEY", os.getenv("OPENAI_API_KEY"))

    # Use LitellmModel for Anthropic models (supports prompt caching via native API)
    if model.startswith("anthropic/"):
        # Strip /v1 suffix from base_url to avoid double /v1/ path
        if base_url:
            base_url = base_url.rstrip("/").removesuffix("/v1")
        return LitellmModel(model=model, api_key=api_key, base_url=base_url)

    # Use OpenAI-compatible client for other providers
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return OpenAIChatCompletionsModel(model=model, openai_client=client)
