import asyncio
import logging
import re
from asyncio import Semaphore
from typing import Any, Callable, Coroutine, Sequence

from pydantic import BaseModel
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


class RetryConfig(BaseModel):
    """Configuration for retry behavior in limited_gather."""

    max_attempts: int = 3
    wait_min: float = 4.0
    wait_max: float = 120.0
    wait_multiplier: float = 1.0
    wait_exp_base: float = 2.0
    retry_exception_names: list[str] = [
        "openai.RateLimitError",
        "anthropic.RateLimitError",
    ]
    retry_status_codes: list[int] = [429, 503, 529]
    retry_message_patterns: list[str] = ["rate limit", "too many requests"]
    retry_on_timeout: bool = True

    def should_retry(self, e: BaseException) -> bool:
        """Determine if an exception should trigger a retry."""
        # Timeout
        if self.retry_on_timeout and isinstance(e, asyncio.TimeoutError):
            return True

        # Exception type name (string matching)
        full_name = f"{type(e).__module__}.{type(e).__name__}"
        if any(name in full_name for name in self.retry_exception_names):
            return True

        # HTTP status code
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        if status in self.retry_status_codes:
            return True

        # Message pattern matching
        msg = str(e).lower()
        if any(re.search(p, msg, re.IGNORECASE) for p in self.retry_message_patterns):
            return True

        return False


async def maybe_await(maybe_coro: Any) -> Any:
    """Maybe await a coroutine."""
    if asyncio.iscoroutine(maybe_coro):
        return await maybe_coro
    return maybe_coro


def is_valid_id(s: str) -> bool:
    """Check if string contains only alphanumeric characters, dashes, or underscores."""
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", s))


def is_valid_folder_name(name: str) -> bool:
    """
    Validates against best-practice folder naming conventions:
        - lowercase letters, digits, dashes, underscores
        - no leading dot (no hidden folders)
        - no trailing dash/underscore
        - no file-like extensions.
    """
    return re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", name) is not None


def sanitize_dirname(name: str, replacement: str = "_") -> str:
    """
    Sanitize a string so it can safely be used as a directory name.

    - Removes or replaces invalid characters (e.g., <>:"/\\|?*).
    - Collapses consecutive replacements into one.
    - Strips leading/trailing spaces and dots.
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', replacement, name)
    sanitized = re.sub(r"\s+", replacement, sanitized)
    sanitized = re.sub(rf"{re.escape(replacement)}+", replacement, sanitized)
    sanitized = sanitized.strip(" ._")
    assert is_valid_folder_name(sanitized), f"Invalid folder name: {sanitized}"
    return sanitized


def make_cli_args(
    positional_args: list[str] | None = None,
    flags: list[str] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """
    Convert a list of args and a dictionary of kwargs to a list of CLI arguments.

    Args:
        positional_args: Positional arguments to add to the CLI arguments.
        flags: Flag arguments to add to the CLI arguments.
        kwargs: A dictionary of kwargs to add to the CLI arguments.

    Returns:
        A list of CLI arguments.
    """
    args = []

    if positional_args:
        args.extend(positional_args)

    if flags is not None:
        for flag in flags:
            flag = flag.replace("_", "-")
            args.append(f"--{flag}")

    if kwargs is not None:
        for k, v in kwargs.items():
            k = k.replace("_", "-")
            args.append(f"--{k}={v}")

    return args


async def limited_gather(
    *coros: Coroutine,
    coro_factories: Sequence[Callable[[], Coroutine]] | None = None,
    limit: int = 10,
    retry_config: RetryConfig | None = None,
    desc: str = "Processing",
    return_exceptions: bool = False,
    timeout: float | None = None,
    bar_format: str = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    run_in_thread: bool = False,
):
    """Gather coroutines with concurrency limit and optional retry.

    Args:
        *coros: Coroutines to execute (mutually exclusive with coro_factories)
        coro_factories: Callables returning coroutines, required for retry support
        limit: Maximum concurrent tasks
        retry_config: Retry configuration (requires coro_factories)
        desc: Progress bar description
        return_exceptions: If True, return exceptions instead of raising
        timeout: Timeout per task in seconds
        bar_format: Progress bar format string
        run_in_thread: If True, run each coroutine in its own event loop in a separate thread.
            Useful for coroutines that may block the event loop.
    """
    # Validation
    if coros and coro_factories is not None:
        raise ValueError(
            "Provide either positional 'coros' or keyword 'coro_factories', not both."
        )
    if not coros and coro_factories is None:
        raise ValueError(
            "Must provide either positional coroutines or 'coro_factories'."
        )
    if retry_config is not None and coros:
        raise ValueError(
            "When using 'retry_config', must use 'coro_factories' instead of positional coroutines."
        )
    if run_in_thread and coros:
        raise ValueError(
            "When using 'run_in_thread', must use 'coro_factories' instead of positional coroutines."
        )

    logger.info(
        f"Running coroutines with concurrency limit {limit} {'in thread' if run_in_thread else 'in event loop'}"
    )

    semaphore = Semaphore(limit)

    # Path A: No retries, using coroutines directly
    if coros:

        async def coro_with_semaphore(coro):
            async with semaphore:
                try:
                    if timeout is not None:
                        return await asyncio.wait_for(coro, timeout=timeout)
                    else:
                        return await coro
                except Exception as e:
                    if return_exceptions:
                        return e
                    raise

        return await tqdm.gather(
            *map(coro_with_semaphore, coros), desc=desc, bar_format=bar_format
        )

    # Path B: With retry (coro_factories)
    def _run_coro_in_thread(factory: Callable[[], Coroutine]):
        """Run a coroutine in a new event loop in the current thread."""
        return asyncio.run(factory())

    if retry_config is None:
        retry_config = RetryConfig()

    async def coro_with_retry_and_semaphore(factory: Callable[[], Coroutine]):
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(retry_config.max_attempts),
                wait=wait_exponential(
                    multiplier=retry_config.wait_multiplier,
                    min=retry_config.wait_min,
                    max=retry_config.wait_max,
                    exp_base=retry_config.wait_exp_base,
                ),
                retry=lambda retry_state: (
                    retry_state.outcome is not None
                    and retry_state.outcome.exception() is not None
                    and retry_config.should_retry(retry_state.outcome.exception())
                ),
                reraise=True,
            ):
                with attempt:
                    async with semaphore:
                        if run_in_thread:
                            coro = asyncio.to_thread(_run_coro_in_thread, factory)
                        else:
                            coro = factory()
                        if timeout is not None:
                            return await asyncio.wait_for(coro, timeout=timeout)
                        else:
                            return await coro
        except Exception as e:
            if return_exceptions:
                return e
            raise

    return await tqdm.gather(
        *[coro_with_retry_and_semaphore(f) for f in coro_factories],
        desc=desc,
        bar_format=bar_format,
    )
