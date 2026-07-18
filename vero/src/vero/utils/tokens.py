from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents import RunResult, RunResultStreaming, TResponseInputItem


def run_result_to_messages(
    result: RunResult | RunResultStreaming | list[TResponseInputItem],
) -> list[dict | TResponseInputItem]:
    """Convert a RunResult or list of TResponseInputItem to a list of messages."""

    result = deepcopy(result)

    if not isinstance(result, list):
        result = result.to_input_list()

    def process_item(item: dict | TResponseInputItem) -> dict | TResponseInputItem:
        for key in ["output", "content"]:
            if key in item:
                if isinstance(item[key], list):
                    item[key] = item[key][0] if item[key] else ""
        return item

    return [process_item(item) for item in result]
