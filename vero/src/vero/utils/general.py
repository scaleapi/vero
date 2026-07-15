from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import (
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

from pydantic import BaseModel, JsonValue

JsonT = TypeVar("JsonT", bound=JsonValue)


@runtime_checkable
class IsDataclass(Protocol):
    __dataclass_fields__: dict  # all dataclasses have this attribute


@overload
def recursively_serialize(data: BaseModel) -> dict[str, JsonValue]: ...


@overload
def recursively_serialize(data: JsonT) -> JsonT: ...


@overload
def recursively_serialize(data: Path) -> str: ...


@overload
def recursively_serialize(data: IsDataclass) -> dict[str, JsonValue]: ...


def recursively_serialize(
    data: BaseModel | JsonValue | Path | IsDataclass,
) -> JsonValue:
    """Recursively serialize a BaseModel or JsonValue or dataclass to a JSON value."""
    if is_dataclass(data):
        return recursively_serialize(asdict(data))

    if isinstance(data, dict):
        return {k: recursively_serialize(v) for k, v in data.items()}

    if isinstance(data, (list, tuple, set)):
        return [recursively_serialize(item) for item in data]

    if isinstance(data, BaseModel):
        return recursively_serialize(data.model_dump())

    if isinstance(data, Path):
        return str(data)

    return data  # type: ignore


def camel_to_snake(s: str) -> str:
    """Convert a camel case string to a snake case string."""
    snake = []
    for char in s:
        if char.isupper() and snake:
            snake.append("_")
        snake.append(char.lower())
    return "".join(snake)


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from a string."""
    _ANSI_ESCAPE_PATTERN = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return _ANSI_ESCAPE_PATTERN.sub("", text)


def paginate(
    items: list[str], max_chars: int, offset: int = 0, limit: int | None = None
) -> list[str]:
    """Paginate a list of items into a list of strings. max_chars is not respected for single item lists."""

    if offset < 0:
        raise ValueError("offset must be greater than or equal to 0.")

    if offset > len(items):
        raise ValueError(
            f"offset must be less than the length of the items {len(items)}."
        )

    items = items[offset:]
    if limit is None:
        limit = len(items)

    paginated_items = []
    paginated_items_length = 0

    for item in items[:limit]:
        paginated_items.append(item)
        paginated_items_length += len(item)

        if paginated_items_length > max_chars:
            break

    return paginated_items
