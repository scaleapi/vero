"""VeroTask definitions for tau_bench."""

import os

import litellm

# Route all litellm calls through the proxy
litellm.api_base = os.getenv("LITELLM_BASE_URL")
litellm.api_key = os.getenv("LITELLM_API_KEY", os.getenv("OPENAI_API_KEY"))

# Import task modules to register them
from .retail import retail_task  # noqa: E402

__all__ = ["airline_task", "retail_task"]
