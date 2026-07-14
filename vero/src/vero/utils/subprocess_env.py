"""Build explicit environment for evaluation subprocesses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from dotenv import dotenv_values

# Bare minimum for a subprocess to function
SYSTEM_DEFAULTS = [
    "PATH",
    "HOME",
    "SHELL",
    "USER",
    "LANG",
    "TMPDIR",
    "TERM",
]

# Vars vero always forwards if present
VERO_DEFAULTS = [
    "UV_INDEX",
    "UV_CACHE_DIR",
]

# An env var spec: either a name (read from os.environ) or (name, callable) for computed values
EnvVarSpec = str | tuple[str, Callable[[], str | None]]

# subprocess_env_vars accepts a list of specs OR a path to a .env file
SubprocessEnvSource = list[EnvVarSpec] | Path | str


def load_env_file(path: Path | str) -> dict[str, str]:
    """Parse a .env file into a dict using python-dotenv.

    Does NOT modify ``os.environ`` — returns the values for the caller to use.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def apply_env_file(path: Path | str) -> None:
    """Load a .env file and set values in ``os.environ``.

    Existing env vars are NOT overwritten — the file provides defaults.
    """
    from dotenv import load_dotenv

    load_dotenv(path, override=False)


def build_subprocess_env(source: SubprocessEnvSource | None = None) -> dict[str, str]:
    """Build an explicit env dict for evaluation subprocesses.

    Args:
        source: One of:
            - ``None`` — returns system + vero defaults only
            - A list of env var specs (names or name/callable tuples)
            - A ``Path`` or string path to a ``.env`` file

    Returns:
        Clean env dict with only declared vars (no full os.environ leak).
    """
    env: dict[str, str] = {}

    # System defaults
    for key in SYSTEM_DEFAULTS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    # Vero defaults
    for key in VERO_DEFAULTS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    if source is None:
        return env

    # Path to .env file
    if isinstance(source, (str, Path)) and not isinstance(source, list):
        p = Path(source)
        if p.exists() and p.is_file():
            env.update(load_env_file(p))
            return env

    # List of env var specs
    if isinstance(source, list):
        for spec in source:
            if isinstance(spec, str):
                key = spec
                val = os.environ.get(key)
            else:
                key, factory = spec
                val = factory()
            if val is not None:
                env[key] = val

    return env
