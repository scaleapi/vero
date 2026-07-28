from inspect import isclass, isfunction, ismethod
from typing import Any, Callable

from agents import FunctionTool, function_tool
from vero.tools.utils import get_tools_from_class
from vero.utils.general import camel_to_snake


def callable_to_oai_tool(
    tool: Callable | object, method_name: str | None = None, strict_mode: bool = True, **kwargs: Any
) -> FunctionTool:
    """Converts a function or class method to an OpenAI Agents FunctionTool.

    Args:
        tool: The function or callable object to convert.
        method_name: If the tool is a class, the name of the method to convert. If None, we assume the tool is a callable object and use __call__.
        strict_mode: Strict mode toggle for function_tool adapter.
        **kwargs: Additional keyword arguments to the OpenAI function_tool adapter.

    Returns:
        An OpenAI Agents FunctionTool.
    """

    if isfunction(tool) or ismethod(tool):
        return function_tool(tool, strict_mode=strict_mode, **kwargs)

    if isclass(tool):
        raise ValueError(
            "Cannot convert a class to an OpenAI Agents FunctionTool. Pass an instance instead."
        )

    tool_cls = tool.__class__

    if method_name is None:
        method_name = "__call__"

        if "name_override" not in kwargs:
            kwargs["name_override"] = camel_to_snake(tool_cls.__name__)
    else:
        if "name_override" not in kwargs:
            kwargs["name_override"] = camel_to_snake(tool_cls.__name__) + "_" + method_name

    # use the method of the instance not the class
    method = getattr(tool, method_name)
    return function_tool(method, strict_mode=strict_mode, **kwargs)


def tool_set_instance_to_oai_tools(
    instance: object,
    strict_mode: bool = True,
    prefix_class_name: bool = True,
    exclude_methods: list[str] | None = None,
) -> list[FunctionTool]:
    """Creates a list of OpenAI Agents FunctionTools from an instance."""

    exclude_methods = exclude_methods or []

    if exclude_methods:
        tools = [
            getattr(instance, tool.__name__)
            for tool in get_tools_from_class(instance.__class__)
            if tool.__name__ not in exclude_methods
        ]
    else:
        tools = [
            getattr(instance, tool.__name__) for tool in get_tools_from_class(instance.__class__)
        ]

    # Defaault to using the __call__ method of the instance if no tools are found.
    if not tools:
        return [callable_to_oai_tool(instance, strict_mode=strict_mode)]

    oai_tools = []
    for tool in tools:
        kwargs = {"strict_mode": strict_mode}
        if prefix_class_name:
            kwargs["name_override"] = f"{instance.__class__.__name__}_{tool.__name__}"
        oai_tools.append(callable_to_oai_tool(tool, **kwargs))
    return oai_tools
