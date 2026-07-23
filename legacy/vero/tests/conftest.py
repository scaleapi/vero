import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_sessions():
    """No-op — tests that need session isolation use vero_home fixture or pass vero_home to Policy."""
    yield


@pytest.fixture(scope="session")
def resources_path() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def uv_index() -> str:
    result = subprocess.run(
        ["pip3", "config", "get", "global.index-url"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()
