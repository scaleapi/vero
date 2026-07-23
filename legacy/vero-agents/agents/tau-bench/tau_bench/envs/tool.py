import abc
from typing import Any


class Tool(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def invoke(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_info() -> dict[str, Any]:
        raise NotImplementedError
