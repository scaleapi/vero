"""Materialize benchmark datasets into the canonical JSONL case boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _validate_json_case(case: Any, index: int) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError(f"benchmark case {index} is not an object")
    value = dict(case)
    value.setdefault("id", str(index))
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"benchmark case {index} is not JSON-serializable") from error
    return value


def _validate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_ids = [str(case["id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case IDs must be unique")
    if not cases:
        raise ValueError("benchmark evaluation sets must contain at least one case")
    return cases


def _load_stored_dataset(path: Path, partition: str) -> list[dict[str, Any]]:
    from datasets import Dataset, DatasetDict, load_from_disk

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if partition not in dataset:
            raise ValueError(
                f"dataset {path} has no partition {partition!r}; "
                f"available: {sorted(dataset)}"
            )
        dataset = dataset[partition]
    elif not isinstance(dataset, Dataset):
        raise TypeError(f"unsupported stored dataset type: {type(dataset).__name__}")
    return _validate_cases(
        [_validate_json_case(case, index) for index, case in enumerate(dataset)]
    )


def materialize_cases(
    *,
    dataset_path: Path | str,
    partition: str,
    output_path: Path | str,
) -> Path:
    """Write one immutable JSONL evaluation set and return its path."""

    source = Path(dataset_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"benchmark dataset does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    if source.is_file() and source.suffix == ".jsonl":
        cases = _validate_cases([
            _validate_json_case(json.loads(line), index)
            for index, line in enumerate(
                line
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        ])
    elif source.is_file() and source.suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "cases" in value:
            value = value["cases"]
        if not isinstance(value, list):
            raise ValueError("JSON benchmark datasets must contain a case list")
        cases = _validate_cases(
            [_validate_json_case(case, index) for index, case in enumerate(value)]
        )
    elif source.is_dir():
        cases = _load_stored_dataset(source, partition)
    else:
        raise ValueError(f"unsupported benchmark dataset path: {source}")

    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = "".join(
        json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n"
        for case in cases
    )
    temporary.write_text(payload, encoding="utf-8")
    if output.exists():
        if output.read_text(encoding="utf-8") != payload:
            temporary.unlink()
            raise ValueError(
                f"materialized cases for an existing session changed: {output}"
            )
        temporary.unlink()
        return output
    temporary.replace(output)
    return output
