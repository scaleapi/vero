"""Settings and secret loading.

Reads `KEY=VALUE` files, which is the convention already in use here (`secrets.env`,
`eval.secrets.env`) as well as the usual `.env`. A twenty-line parser covers both and
avoids adding a dependency for it.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

DEFAULT_CACHE = Path.home() / ".cache" / "vero-interpret"
DEFAULT_MODEL = "gpt-5.4-mini"


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    """Parse a KEY=VALUE file into the environment. Returns what it set."""
    if not path.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def load_secrets(paths: list[Path] | None = None) -> None:
    """Load the first secrets file that exists, plus `.env` if present."""
    candidates = paths or [
        Path(".env"),
        Path("secrets.env"),
        Path("vero/secrets.env"),
    ]
    for path in candidates:
        if path.is_file():
            load_env_file(path)


class Settings(BaseModel):
    """Everything the labelling stage needs to run."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    concurrency: int = 16
    max_retries: int = 4
    request_timeout: float = 120.0
    cache_dir: Path = DEFAULT_CACHE

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        load_secrets()
        base = os.environ.get("OPENAI_BASE_URL")
        values = {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            # The gateway's OPENAI_BASE_URL ends in "/v1/" here. The OpenAI client
            # appends "/chat/completions", and the resulting double slash returns a
            # flat 403 "This route is not publicly accessible" that reads like a
            # permissions problem and is not one. Strip it once, here.
            "base_url": base.rstrip("/") if base else None,
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
