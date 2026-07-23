from typing import Callable

_IS_TOOL = "_is_tool"


def is_tool(method: Callable) -> Callable:
    """A decorator to mark a method of a class as a tool."""
    setattr(method, _IS_TOOL, True)
    return method


def get_tools_from_class(
    cls_or_instance: type | object,
) -> list[Callable]:
    """Get all methods marked with @is_tool from a class or instance.

    If the instance has an `exclude_tools` attribute (from the ToolSet protocol),
    those method names are excluded from the result.

    Args:
        cls_or_instance: A class type or an instance to get tools from.

    Returns:
        List of methods marked with @is_tool decorator.
    """
    exclude = getattr(cls_or_instance, "exclude_tools", []) or []
    return [
        getattr(cls_or_instance, attr)
        for attr in dir(cls_or_instance)
        if getattr(getattr(cls_or_instance, attr), _IS_TOOL, False)
        and attr not in exclude
    ]
