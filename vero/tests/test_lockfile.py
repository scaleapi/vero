"""Tests to ensure project configuration files are up-to-date."""

import subprocess
import warnings
from pathlib import Path

# Get the project root (parent of tests directory)
PROJECT_ROOT = Path(__file__).parent.parent


class TestLockfile:
    """Tests for uv.lock file synchronization."""

    def test_uv_lock_is_up_to_date(self):
        """Ensure uv.lock is synchronized with pyproject.toml.

        If this test warns, run `uv lock` to update the lock file.
        """
        result = subprocess.run(
            ["uv", "lock", "--check"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            warnings.warn(
                f"uv.lock is out of sync with pyproject.toml. "
                f"Run `uv lock` to update it.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}",
                UserWarning,
            )
