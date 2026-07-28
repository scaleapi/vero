"""Content-addressed dataset store with per-session ID mappings."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """Fallback serializer for types not natively supported by json.dumps."""
    if isinstance(obj, bytes):
        return obj.hex()
    if hasattr(obj, "tolist"):  # numpy arrays, torch tensors, etc.
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def hash_dataset_dict(dataset_dict: Any) -> str:
    """Compute a canonical SHA-256 content hash for a DatasetDict or Dataset.

    Canonical guarantees:
    - Split order doesn't matter (splits are sorted by name).
    - Column order doesn't matter (keys are sorted per row).
    - Row order DOES matter (preserves dataset semantics).

    Args:
        dataset_dict: A HuggingFace DatasetDict or Dataset.

    Returns:
        A hex-encoded SHA-256 digest string.
    """
    from datasets import Dataset, DatasetDict

    # Wrap single Dataset into DatasetDict
    if isinstance(dataset_dict, Dataset):
        dataset_dict = DatasetDict({"train": dataset_dict})

    hasher = hashlib.sha256()

    for split_name in sorted(dataset_dict.keys()):
        hasher.update(f"split:{split_name}\n".encode())

        dataset = dataset_dict[split_name]

        # Hash schema
        schema = {col: str(dataset.features[col]) for col in sorted(dataset.features)}
        hasher.update(f"schema:{json.dumps(schema, sort_keys=True)}\n".encode())

        # Hash every row in order
        for row in dataset:
            row_bytes = json.dumps(row, sort_keys=True, default=_json_default).encode()
            hasher.update(row_bytes)
            hasher.update(b"\n")

    return hasher.hexdigest()


def _get_mapping_path(sessions_dir: Path, session_id: str) -> Path:
    return sessions_dir / session_id / "datasets.json"


def _read_mapping(sessions_dir: Path, session_id: str) -> dict[str, str]:
    path = _get_mapping_path(sessions_dir, session_id)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _write_mapping(sessions_dir: Path, session_id: str, mapping: dict[str, str]) -> None:
    path = _get_mapping_path(sessions_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2))


def save_dataset(
    sessions_dir: Path, dataset_cache: Path, session_id: str, dataset_id: str, dataset: Any
) -> str:
    """Save a DatasetDict (or Dataset) to the content-addressed cache
    and register the ID in the session mapping.

    Args:
        sessions_dir: Path to the sessions root directory.
        dataset_cache: Path to the dataset cache directory.
        session_id: Session to register the dataset in.
        dataset_id: Human-readable name for this dataset.
        dataset: A HuggingFace DatasetDict or Dataset.

    Returns:
        The content fingerprint (cache key).
    """
    from datasets import Dataset, DatasetDict

    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    fp = hash_dataset_dict(dataset)
    cache_path = dataset_cache / fp

    if not cache_path.exists():
        cache_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cache_path))
        logger.info(f"Saved dataset '{dataset_id}' to cache: {cache_path}")
    else:
        logger.debug(f"Dataset '{dataset_id}' already in cache: {fp}")

    # Update session mapping
    mapping = _read_mapping(sessions_dir, session_id)
    mapping[dataset_id] = fp
    _write_mapping(sessions_dir, session_id, mapping)

    return fp


def resolve_and_save_dataset(
    dataset: Any,
    sessions_dir: Path,
    dataset_cache: Path,
    session_id: str,
    dataset_id: str = "dataset",
) -> str:
    """Resolve a dataset from various sources and save to the session store.

    Accepts:
        - ``DatasetDict`` or ``Dataset`` objects directly
        - ``Path`` or ``str`` to a saved DatasetDict on disk
        - ``str`` that doesn't exist on disk — treated as an already-registered dataset ID

    Returns:
        The resolved dataset_id.
    """
    from datasets import Dataset, DatasetDict

    if isinstance(dataset, (Dataset, DatasetDict)):
        save_dataset(sessions_dir, dataset_cache, session_id, dataset_id, dataset)
    elif isinstance(dataset, (str, Path)):
        path = Path(dataset)
        if path.exists():
            ds = DatasetDict.load_from_disk(str(path))
            dataset_id = path.stem
            save_dataset(sessions_dir, dataset_cache, session_id, dataset_id, ds)
        else:
            dataset_id = str(dataset)
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    return dataset_id


def load_dataset(sessions_dir: Path, dataset_cache: Path, session_id: str, dataset_id: str) -> Any:
    """Load a DatasetDict by ID from the session mapping → cache.

    Args:
        sessions_dir: Path to the sessions root directory.
        dataset_cache: Path to the dataset cache directory.
        session_id: Session to look up the dataset in.
        dataset_id: Human-readable dataset name.

    Returns:
        A HuggingFace DatasetDict.

    Raises:
        FileNotFoundError: If the dataset ID is not in the session mapping
            or the cache entry is missing.
    """
    from datasets import DatasetDict

    mapping = _read_mapping(sessions_dir, session_id)
    fp = mapping.get(dataset_id)
    if fp is None:
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' not found in session {session_id}"
        )

    cache_path = dataset_cache / fp
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache entry '{fp}' missing for dataset '{dataset_id}'"
        )

    return DatasetDict.load_from_disk(str(cache_path))


def dataset_exists(sessions_dir: Path, session_id: str, dataset_id: str) -> bool:
    """Check if a dataset ID is registered in a session."""
    mapping = _read_mapping(sessions_dir, session_id)
    return dataset_id in mapping


def list_datasets(sessions_dir: Path, session_id: str) -> list[str]:
    """List all dataset IDs registered in a session."""
    return list(_read_mapping(sessions_dir, session_id).keys())
