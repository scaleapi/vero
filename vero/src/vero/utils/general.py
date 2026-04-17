from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

from pydantic import BaseModel, JsonValue

if TYPE_CHECKING:
    import pandas as pd

JsonT = TypeVar("JsonT", bound=JsonValue)


@runtime_checkable
class IsDataclass(Protocol):
    __dataclass_fields__: dict  # all dataclasses have this attribute


def normalize_dash_underscore(s: str) -> str:
    """Just to make things look nice."""
    underscore_count = s.count("_")
    dash_count = s.count("-")
    if underscore_count > dash_count:
        return s.replace("_", "-")
    else:
        return s.replace("-", "_")


def random_readable_id(token_length: int = 0) -> str:
    """Generates a random readable ID, e.g. fragrant-bread"""
    from haikunator import Haikunator

    haikunator = Haikunator()
    return haikunator.haikunate(token_length=token_length)


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


def df_to_format(
    df: pd.DataFrame,
    fmt: str
    | Literal[
        "csv", "json", "jsonl", "html", "markdown", "yaml", "ini", "pipe", "kv_markdown"
    ],
    **kwargs: Any,
) -> str | None:
    """
    Convert a Pandas DataFrame to a string in the given format.

    Args:
        df: The DataFrame to convert
        fmt: The format to convert to
        kwargs: Format-specific options (e.g. indent, table_name, etc.)

    Returns:
        A string in the given format
    """

    fmt = str(fmt).lower()

    if fmt == "csv":
        import io

        buf = io.StringIO()
        df.to_csv(buf, index=False, **kwargs)
        return buf.getvalue()

    elif fmt == "json":
        return df.to_json(orient="records", force_ascii=False, date_format="iso", **kwargs)

    elif fmt == "jsonl":
        import json

        records = df.to_dict(orient="records")
        lines = [json.dumps(rec, ensure_ascii=False, **kwargs) for rec in records]
        return "\n".join(lines)

    elif fmt == "html":
        return df.to_html(index=False, **kwargs)

    elif fmt == "markdown":
        return df.to_markdown(index=False, **kwargs)

    elif fmt == "yaml":
        import yaml

        records = df.to_dict(orient="records")
        wrapper = kwargs.get("top_key", "records")
        return yaml.dump({wrapper: records}, sort_keys=False, allow_unicode=True)

    elif fmt == "ini":
        import io
        from configparser import ConfigParser

        table_name = kwargs.get("record_prefix", "record")
        cfg = ConfigParser()
        records = df.to_dict(orient="records")
        for i, rec in enumerate(records, start=0):
            section = f"{table_name}_{i}"
            cfg[section] = {str(k): str(v) for k, v in rec.items()}
        s = io.StringIO()
        cfg.write(s)
        return s.getvalue()

    elif fmt == "pipe":
        recs = df.to_dict(orient="records")
        lines = []
        for rec in recs:
            parts = [f"{k}: {v}" for k, v in rec.items()]
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    elif fmt == "kv_markdown":
        recs = df.to_dict(orient="records")
        lines = []

        title = kwargs.get("title")
        record_prefix = kwargs.get("record_prefix", "Record")

        if title is not None:
            lines.append(f"# {title}")

        for i, rec in enumerate(recs, start=0):
            lines.append(f"\n## {record_prefix} {i}")
            lines.append("```markdown")
            for k, v in rec.items():
                lines.append(f"{k}: {v}")
            lines.append("```\n")
        return "\n".join(lines)

    else:
        raise ValueError(f"Unsupported format: {fmt}")


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
