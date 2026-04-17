"""VeroTask definitions for terminus_kira."""

import os

# Route Anthropic litellm calls through the proxy.
# Harbor's Terminus2 agent uses litellm.acompletion which checks ANTHROPIC_API_KEY
# for auth and uses api_base passed via agent kwargs or ANTHROPIC_API_BASE.
_base_url = os.getenv("LITELLM_BASE_URL", "")
_api_key = os.getenv("LITELLM_API_KEY", "")
if _base_url and _api_key:
    # Strip /v1 suffix — Anthropic's native API doesn't use it
    _base_url = _base_url.rstrip("/").removesuffix("/v1")
    os.environ.setdefault("ANTHROPIC_API_KEY", _api_key)
    os.environ.setdefault("ANTHROPIC_API_BASE", _base_url)

# Import task modules to register them
from .terminal_bench import terminal_bench_2  # noqa: E402

__all__ = ["terminal_bench_2"]
