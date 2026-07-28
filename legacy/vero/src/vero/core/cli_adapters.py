from __future__ import annotations

import argparse
import os
from enum import Enum, auto
from typing import Annotated, Any, Callable, Literal, Type, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .utils import make_cli_args

T = TypeVar("T", bound="CLIAdapter")


class CLIFieldMetadata(str, Enum):
    flag = auto()
    positional = auto()


class CLIAdapter(BaseModel):
    """Parameters for a CLI command."""

    @classmethod
    def field_name_to_cli_arg(cls, field_name: str) -> str:
        return field_name.replace("_", "-")

    @classmethod
    def is_flag(cls, field_name: str) -> bool:
        """Checks if a field is a flag based on it's metadata."""
        return CLIFieldMetadata.flag in cls.model_fields[field_name].metadata

    @classmethod
    def is_positional(cls, field_name: str) -> bool:
        """Checks if a field is a positional argument based on it's metadata."""
        return CLIFieldMetadata.positional in cls.model_fields[field_name].metadata

    @classmethod
    def is_kwarg(cls, field_name: str) -> bool:
        """Checks if a field is a keyword argument based on it's metadata."""
        return not cls.is_flag(field_name) and not cls.is_positional(field_name)

    @classmethod
    def get_type_converter(cls, field_name: str) -> Callable[[Any], Any]:
        """Get a TypeAdapter-based converter function for the field's type."""
        field_type: Type | None = cls.model_fields[field_name].annotation

        def converter(value: Any):
            try:
                return TypeAdapter(field_type).validate_python(value)
            except ValidationError as e:
                raise argparse.ArgumentTypeError(str(e))

        return converter

    def get_flags(self) -> list[str]:
        """Returns a list of flags for the CLI command."""
        flags = []
        for field_name, field_value in self.model_dump().items():
            if self.is_flag(field_name) and field_value is True:
                flags.append(self.field_name_to_cli_arg(field_name))
        return flags

    def get_kwargs(self) -> dict[str, Any]:
        """Returns a dictionary of kwargs for the CLI command."""
        kwargs = {}
        for field_name, field_value in self.model_dump().items():
            if self.is_kwarg(field_name) and field_value is not None:
                kwargs[self.field_name_to_cli_arg(field_name)] = field_value
        return kwargs

    def get_positional_args(self) -> list[str]:
        """Returns a list of positional arguments for the CLI command."""
        positional_args = []
        for field_name, field_value in self.model_dump().items():
            if self.is_positional(field_name) and field_value:
                assert isinstance(field_value, list), (
                    "Positional arguments must be a list"
                )
                positional_args.extend(field_value)
        return positional_args

    def get_cli_args(self) -> list[str]:
        """Returns a list of CLI arguments for the CLI command."""

        positional_args = self.get_positional_args()
        flags = self.get_flags()
        kwargs = self.get_kwargs()
        return make_cli_args(
            positional_args=positional_args, flags=flags, kwargs=kwargs
        )

    def get_cmd(self) -> list[str]:
        raise NotImplementedError(
            f"get_cmd is not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def from_args(cls: Type[T], args: argparse.Namespace) -> T:
        """Create an instance from parsed command line arguments."""
        args_dict = {}
        for field_name in cls.model_fields:
            if hasattr(args, field_name):
                value = getattr(args, field_name)
                if value is not None:
                    args_dict[field_name] = value

        return cls(**args_dict)


class UvRunParameters(CLIAdapter):
    """Parameters for a uv run command.

    Attributes:
        no_sync: Use existing dependencies instead of trying to download them. Set this to True when in offline environments.
        env_file: Path to the environment file relative to the project home.
        index: The uv index to use
        project: Path to the uv project
    """

    no_sync: Annotated[bool, CLIFieldMetadata.flag] = False
    env_file: str | None = None
    index: str | None = None
    project: str | None = None
    with_editable: str | None = None

    def get_cmd(self) -> list[str]:
        return ["uv", "run", *self.get_cli_args()]

    @classmethod
    def from_env(
        cls,
        project: str | None = None,
        env_file: str | None = None,
        with_editable: str | None = None,
    ) -> UvRunParameters:
        return cls(
            index=os.environ.get("UV_INDEX"),
            no_sync=bool(os.environ.get("IN_DOCKER_CONTAINER")),
            project=project,
            env_file=env_file,
            with_editable=with_editable,
        )


class PytestParameters(CLIAdapter):
    """Parameters for a pytest run.

    Attributes:
        json_report_file: Path to the JSON file to write the report to
        file_or_dir: Files or directories to test
        json_report: Whether to write a JSON report
        json_report_indent: Indentation level for the JSON report
        json_report_omit: Items to omit from the JSON report
        evaluation_parameters: JSON string containing all evaluation parameters
    """

    json_report_file: str | None = None
    file_or_dir: Annotated[list[str], CLIFieldMetadata.positional] = Field(
        default_factory=list
    )
    json_report: Annotated[Literal[True], CLIFieldMetadata.flag] = True
    json_report_indent: int = 2
    json_report_omit: list[str] = Field(default_factory=lambda: ["collectors"])
    evaluation_parameters: str | None = None

    def get_cmd(self) -> list[str]:
        return ["pytest", *self.get_cli_args()]
