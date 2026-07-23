"""Hook registry for pre-task execution hooks.

All hooks live in vero itself. Users specify hook names via CLI.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from vero.core.evaluation import EvaluationParameters

logger = logging.getLogger(__name__)


class TaskHookRegistry:
    """Registry for pre-run hooks."""

    _hooks: dict[str, Callable[[EvaluationParameters], None]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a hook by name."""

        def decorator(func: Callable[[EvaluationParameters], None]):
            cls._hooks[name] = func
            logger.debug(f"Registered hook: {name}")
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[[EvaluationParameters], None] | None:
        """Get a hook by name."""
        return cls._hooks.get(name)

    @classmethod
    def list_hooks(cls) -> list[str]:
        """List all registered hook names."""
        return list(cls._hooks.keys())

    @classmethod
    def execute(cls, names: list[str], params: EvaluationParameters) -> None:
        """Execute hooks by name."""
        for name in names:
            hook = cls._hooks.get(name)
            if hook is None:
                logger.warning(f"Unknown hook: {name}")
                continue
            logger.info(f"Executing hook: {name}")
            hook(params)


# ============================================================================
# Built-in hooks
# ============================================================================


@TaskHookRegistry.register("setup_logging")
def setup_logging(params: EvaluationParameters) -> None:
    """Configure logging in the task subprocess.

    Silences noisy libraries (litellm, httpx) and suppresses litellm's
    direct-to-stdout prints that pollute the JSON output.
    """
    from vero.logging import setup_logging as _setup_logging

    _setup_logging()

    # litellm prints banners and info directly to stdout via print(),
    # bypassing the logging system. Suppress by setting its verbosity flag.
    try:
        import litellm
        litellm.suppress_debug_info = True
    except (ImportError, AttributeError):
        pass


@TaskHookRegistry.register("configure_litellm")
def configure_litellm(params: EvaluationParameters) -> None:
    """Configure litellm from LITELLM_BASE_URL and LITELLM_API_KEY env vars."""
    import litellm

    base_url = os.getenv("LITELLM_BASE_URL")
    api_key = os.getenv("LITELLM_API_KEY")

    if base_url:
        litellm.api_base = base_url
        logger.info(f"Set litellm.api_base = {base_url}")
    if api_key:
        litellm.api_key = api_key
        logger.info("Set litellm.api_key = ***")


@TaskHookRegistry.register("enable_litellm_serializer_patch")
def enable_litellm_serializer_patch(params: EvaluationParameters) -> None:
    """Enable litellm serializer patch for OpenAI Agents SDK."""
    os.environ["OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH"] = "true"
    logger.info("Set OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH=true")
