from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, JsonValue

from vero.core.constants import samples_dir_name

if TYPE_CHECKING:
    from vero.core.db.result import SampleResult

logger = logging.getLogger(__name__)


def get_vero_home_dir() -> Path:
    return Path(os.getenv("VERO_HOME_DIR", Path.home() / ".vero")).resolve()


@contextmanager
def ephemeral_vero_home():
    """Context manager that creates a temporary vero home directory.

    Creates sessions/ and datasets/ subdirectories. The temp directory
    is cleaned up on exit. Pass the yielded path to Policy(vero_home=...).

    Usage::

        with ephemeral_vero_home() as vero_home:
            policy = Policy(vero_home=vero_home, ...)
            await policy.init()
        # vero_home is deleted
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "sessions").mkdir()
        (td_path / "datasets").mkdir()
        yield td_path


class FileNotFoundInCacheError(FileNotFoundError): ...


# -----------------------------------------------------------------------------
# Session directory management (optimization sessions)
# -----------------------------------------------------------------------------


def get_session_dir(sessions_dir: Path, session_id: str) -> Path:
    """Returns the path to a session directory for a given session ID."""
    return sessions_dir / session_id


def create_session_dir(sessions_dir: Path, session_id: str) -> Path:
    """Creates a session directory and returns its path."""
    session_dir = get_session_dir(sessions_dir, session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_session_db_path(sessions_dir: Path, session_id: str) -> Path:
    """Returns the path to the database dump file for a session."""
    return get_session_dir(sessions_dir, session_id) / "database.json"


def get_session_config_path(sessions_dir: Path, session_id: str) -> Path:
    """Returns the path to the config dump file for a session."""
    return get_session_dir(sessions_dir, session_id) / "config.json"


def get_session_result_path(sessions_dir: Path, session_id: str) -> Path:
    """Returns the path to the run result dump file for a session."""
    return get_session_dir(sessions_dir, session_id) / "result.json"


def get_session_state_path(sessions_dir: Path, session_id: str) -> Path:
    """Returns the path to the agent state file for a session."""
    return get_session_dir(sessions_dir, session_id) / "agent_state.json"


def get_session_experiments_dir(sessions_dir: Path, session_id: str) -> Path:
    """Returns the path to the experiments directory within a session."""
    return get_session_dir(sessions_dir, session_id) / "experiments"


def find_project_dir_in_session(sessions_dir: Path, session_id: str) -> Path | None:
    """Find the isolated project directory (contains .git/) in a session."""
    session_dir = get_session_dir(sessions_dir, session_id)
    if not session_dir.exists():
        return None
    for child in session_dir.iterdir():
        if child.is_dir() and (child / ".git").exists():
            return child
    return None


# -----------------------------------------------------------------------------
# Experiment directory management
# -----------------------------------------------------------------------------


def get_experiment_dir(sessions_dir: Path, session_id: str, result_id: str) -> Path:
    """Returns the path to an experiment directory within a session."""
    return get_session_experiments_dir(sessions_dir, session_id) / result_id


def initialize_result_store(sessions_dir: Path, session_id: str, result_id: str) -> Path:
    """Initialize the result store directory.

    Args:
        sessions_dir: Path to the sessions root directory.
        session_id: Session ID.
        result_id: Experiment result ID.

    Returns:
        Path to the result directory.
    """
    result_dir = get_experiment_dir(sessions_dir, session_id, result_id)
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized result store: {result_dir}")
    else:
        file_count = len(list(result_dir.iterdir()))
        logger.info(
            f"Result store {result_dir} already exists with {file_count} files."
        )
    return result_dir


def load_json_from_cache(
    sessions_dir: Path, session_id: str, result_id: str, basename: str, model: type[BaseModel] | None = None
) -> Any:
    """Loads a JSON file from the experiment directory.

    Args:
        sessions_dir: Path to the sessions root directory.
        session_id: Session ID.
        result_id: Experiment result ID.
        basename: The filename to load.
        model: Optional Pydantic model to validate the data against.

    Returns:
        The loaded JSON data, or the validated model if provided.

    Raises:
        FileNotFoundInCacheError: If the file does not exist.
    """
    path_to_json_file = get_experiment_dir(sessions_dir, session_id, result_id) / basename

    if not path_to_json_file.exists():
        raise FileNotFoundInCacheError(
            f"JSON file {path_to_json_file} not found in cache."
        )

    if model is not None:
        return model.model_validate_json(path_to_json_file.read_text())

    return json.loads(path_to_json_file.read_text())


def save_json_to_cache(
    sessions_dir: Path,
    session_id: str,
    result_id: str,
    basename: str,
    data: JsonValue | BaseModel,
    indent: int = 2,
) -> Path:
    """Saves a JSON file to the experiment directory."""
    path_to_json_file = get_experiment_dir(sessions_dir, session_id, result_id) / basename
    path_to_json_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")

    path_to_json_file.write_text(json.dumps(data, indent=indent))
    return path_to_json_file


def clear_result_cache(
    sessions_dir: Path, session_id: str, result_id: str, result_basenames: list[str] | None = None
) -> None:
    """Clear cached results for an experiment directory.

    Clears the samples directory and any specified result files. Use this before
    running a new task to ensure stale results are not read if the run fails.

    Args:
        sessions_dir: Path to the sessions root directory.
        session_id: Session ID.
        result_id: Experiment result ID.
        result_basenames: List of result file basenames to clear (e.g. pytest report).
            If None, only clears the samples directory.
    """
    result_dir = get_experiment_dir(sessions_dir, session_id, result_id)
    # Clear sample results directory
    samples_dir = get_samples_dir(result_dir)
    if samples_dir.exists():
        num_samples = len(list(samples_dir.glob("*.json")))
        shutil.rmtree(samples_dir)
        logger.info(f"Cleared {num_samples} cached sample results from {samples_dir}")

    # Clear specified result files
    if result_basenames:
        for basename in result_basenames:
            path = result_dir / basename
            if path.exists():
                path.unlink()
                logger.info(f"Cleared cached result file: {path}")


# -----------------------------------------------------------------------------
# Per-sample result I/O
# -----------------------------------------------------------------------------


def get_samples_dir(result_dir: Path) -> Path:
    """Returns the path to the samples directory within a result directory."""
    return result_dir / samples_dir_name


def save_sample_result(
    sessions_dir: Path, session_id: str, result_id: str, sample_id: int, result: SampleResult
) -> Path:
    """Save a single sample result to its own JSON file.

    Args:
        sessions_dir: Path to the sessions root directory.
        session_id: Session ID.
        result_id: Experiment result ID.
        sample_id: The sample ID.
        result: The SampleResult to save.

    Returns:
        Path to the saved JSON file.
    """
    result_dir = get_experiment_dir(sessions_dir, session_id, result_id)
    samples_dir = get_samples_dir(result_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"{sample_id}.json"
    path.write_text(result.model_dump_json(indent=2))
    return path


def load_sample_result(
    sessions_dir: Path, session_id: str, result_id: str, sample_id: int
) -> SampleResult | None:
    """Load a single sample result by ID.

    Args:
        sessions_dir: Path to the sessions root directory.
        session_id: Session ID.
        result_id: Experiment result ID.
        sample_id: The sample ID to load.

    Returns:
        The SampleResult if found, None otherwise.
    """
    from vero.core.db.result import SampleResult

    result_dir = get_experiment_dir(sessions_dir, session_id, result_id)
    path = get_samples_dir(result_dir) / f"{sample_id}.json"
    if not path.exists():
        return None
    return SampleResult.model_validate_json(path.read_text())


def load_all_sample_results(sessions_dir: Path, session_id: str, result_id: str) -> dict[int, SampleResult]:
    """Load all sample results from an experiment directory.

    Args:
        sessions_dir: Path to the sessions root directory.
        session_id: Session ID.
        result_id: Experiment result ID.

    Returns:
        Dictionary mapping sample_id to SampleResult.
    """
    from vero.core.db.result import SampleResult

    result_dir = get_experiment_dir(sessions_dir, session_id, result_id)
    samples_dir = get_samples_dir(result_dir)
    if not samples_dir.exists():
        return {}
    return {
        int(p.stem): SampleResult.model_validate_json(p.read_text())
        for p in samples_dir.glob("*.json")
    }
