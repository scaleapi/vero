"""Tests for VeroResource abstraction."""

from pathlib import Path

from vero.core.resource import ResourceDiscovery, StaticResourceInfo, resource


class TestResourceDecorator:
    """Test the @resource decorator."""

    def test_decorator_is_passthrough(self):
        """Decorator returns the original function unchanged."""

        @resource("test_ns")
        def my_func(x: int) -> str:
            return str(x)

        # Function still works normally
        assert my_func(42) == "42"
        assert my_func.__name__ == "my_func"

    def test_decorator_with_custom_name(self):
        """Decorator accepts custom name parameter."""

        @resource("test_ns", name="custom")
        def my_func(x: int) -> str:
            return str(x)

        # Function still works
        assert my_func(10) == "10"


class TestStaticResourceInfo:
    """Test StaticResourceInfo data class."""

    def test_qualified_name(self):
        """Qualified name combines namespace and name."""
        info = StaticResourceInfo(
            namespace="my_ns",
            name="my_func",
            target_name="my_func",
            file_path=Path("/test.py"),
            line_number=10,
            module="test",
            signature_str="(x: int) -> str",
            docstring="A test function.",
            source="def my_func(x: int) -> str:\n    pass",
        )
        assert info.qualified_name == "my_ns.my_func"

    def test_description_from_docstring(self):
        """Description extracts first line of docstring."""
        info = StaticResourceInfo(
            namespace="my_ns",
            name="my_func",
            target_name="my_func",
            file_path=Path("/test.py"),
            line_number=10,
            module="test",
            signature_str="(x: int) -> str",
            docstring="First line.\nSecond line.",
            source="def my_func(x: int) -> str:\n    pass",
        )
        assert info.description == "First line."

    def test_description_empty_when_no_docstring(self):
        """Description is empty string when no docstring."""
        info = StaticResourceInfo(
            namespace="my_ns",
            name="my_func",
            target_name="my_func",
            file_path=Path("/test.py"),
            line_number=10,
            module="test",
            signature_str="(x: int) -> str",
            docstring=None,
            source="def my_func(x: int) -> str:\n    pass",
        )
        assert info.description == ""

    def test_function_name_backwards_compat(self):
        """function_name property returns target_name for backwards compat."""
        info = StaticResourceInfo(
            namespace="my_ns",
            name="my_func",
            target_name="actual_name",
            file_path=Path("/test.py"),
            line_number=10,
            module="test",
            signature_str="()",
            docstring=None,
            source="def actual_name(): pass",
        )
        assert info.function_name == "actual_name"
        assert info.target_name == "actual_name"

    def test_kind_defaults_to_function(self):
        """kind defaults to 'function' when not specified."""
        info = StaticResourceInfo(
            namespace="my_ns",
            name="my_func",
            target_name="my_func",
            file_path=Path("/test.py"),
            line_number=10,
            module="test",
            signature_str="()",
            docstring=None,
            source="def my_func(): pass",
        )
        assert info.kind == "function"
        assert info.class_name is None


class TestResourceDiscovery:
    """Test AST-based resource discovery."""

    def test_parse_simple_resource(self):
        """Can parse a simple @resource decorated function."""
        source = '''
from vero.core.resource import resource

@resource("my_ns")
def simple_func(x: int) -> str:
    """A simple function."""
    return str(x)
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.namespace == "my_ns"
        assert r.name == "simple_func"
        assert r.function_name == "simple_func"
        assert "int" in r.signature_str
        assert "str" in r.signature_str
        assert r.docstring == "A simple function."

    def test_parse_resource_with_custom_name(self):
        """Can parse @resource with custom name."""
        source = '''
from vero.core.resource import resource

@resource("my_ns", name="custom")
def renamed_func(data: dict) -> list:
    """Process data."""
    return []
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        assert resources[0].namespace == "my_ns"
        assert resources[0].name == "custom"
        assert resources[0].function_name == "renamed_func"

    def test_parse_async_function(self):
        """Can parse async functions."""
        source = '''
from vero.core.resource import resource

@resource("async_ns")
async def async_handler(request: dict) -> dict:
    """Handle request asynchronously."""
    return {}
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        assert resources[0].name == "async_handler"
        assert resources[0].namespace == "async_ns"

    def test_parse_multiple_resources(self):
        """Can parse multiple resources in one file."""
        source = """
from vero.core.resource import resource

@resource("ns1")
def func_a(x: int) -> int:
    return x

@resource("ns2")
def func_b(y: str) -> str:
    return y

def not_a_resource():
    pass

@resource("ns1", name="custom")
def func_c() -> None:
    pass
"""
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 3

        names = {(r.namespace, r.name) for r in resources}
        assert ("ns1", "func_a") in names
        assert ("ns2", "func_b") in names
        assert ("ns1", "custom") in names

    def test_parse_invalid_source_returns_empty(self):
        """Invalid Python source returns empty list."""
        resources = ResourceDiscovery._parse_resources_from_source(
            "this is not valid python {{{",
            file_path=Path("/fake.py"),
            module="fake",
        )
        assert resources == []

    def test_parse_no_resources_returns_empty(self):
        """File with no @resource decorators returns empty list."""
        source = """
def regular_function():
    pass

class SomeClass:
    pass
"""
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake.py"),
            module="fake",
        )
        assert resources == []

    def test_source_includes_decorator(self):
        """Extracted source includes the decorator."""
        source = """
from vero.core.resource import resource

@resource("my_ns")
def my_func(x: int) -> str:
    return str(x)
"""
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        assert "@resource" in resources[0].source
        assert "def my_func" in resources[0].source

    def test_path_to_module_conversion(self):
        """File paths are converted to module names correctly."""
        assert (
            ResourceDiscovery._path_to_module("src/mypackage/foo/bar.py", "src/mypackage")
            == "foo.bar"
        )

        assert (
            ResourceDiscovery._path_to_module("src/mypackage/foo/__init__.py", "src/mypackage")
            == "foo"
        )

        assert ResourceDiscovery._path_to_module("other/path.py", "src/mypackage") == "other.path"


class TestClassResourceDiscovery:
    """Test discovery of @resource decorated classes."""

    def test_parse_simple_class(self):
        """Can parse a simple @resource decorated class."""
        source = '''
from vero.core.resource import resource

@resource("models")
class MyModel:
    """A simple model."""
    def __init__(self, name: str):
        self.name = name
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.namespace == "models"
        assert r.name == "MyModel"
        assert r.target_name == "MyModel"
        assert r.kind == "class"
        assert r.class_name is None
        assert r.docstring == "A simple model."
        assert "(name: str)" in r.signature_str

    def test_parse_class_with_custom_name(self):
        """Can parse @resource class with custom name."""
        source = '''
from vero.core.resource import resource

@resource("models", name="custom_model")
class InternalModel:
    """Internal model."""
    pass
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        assert resources[0].namespace == "models"
        assert resources[0].name == "custom_model"
        assert resources[0].target_name == "InternalModel"
        assert resources[0].kind == "class"

    def test_parse_class_without_init(self):
        """Class without __init__ has empty signature."""
        source = '''
from vero.core.resource import resource

@resource("utils")
class EmptyClass:
    """No init."""
    pass
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        assert resources[0].signature_str == "()"

    def test_parse_dataclass(self):
        """Can parse @resource decorated dataclass."""
        source = '''
from dataclasses import dataclass
from vero.core.resource import resource

@resource("configs")
@dataclass
class Config:
    """Configuration dataclass."""
    name: str
    value: int = 0
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.namespace == "configs"
        assert r.name == "Config"
        assert r.kind == "class"
        assert r.docstring == "Configuration dataclass."
        assert "@resource" in r.source
        assert "@dataclass" in r.source

    def test_parse_dataclass_decorator_order(self):
        """Can parse dataclass with @resource after @dataclass."""
        source = '''
from dataclasses import dataclass
from vero.core.resource import resource

@dataclass
@resource("configs")
class Config:
    """Config with reversed decorators."""
    name: str
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.namespace == "configs"
        assert r.name == "Config"
        assert r.kind == "class"

    def test_parse_pydantic_model(self):
        """Can parse @resource decorated Pydantic BaseModel."""
        source = '''
from pydantic import BaseModel
from vero.core.resource import resource

@resource("schemas")
class UserSchema(BaseModel):
    """User schema for validation."""
    name: str
    age: int
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.namespace == "schemas"
        assert r.name == "UserSchema"
        assert r.kind == "class"
        assert r.docstring == "User schema for validation."
        # Pydantic models typically don't have explicit __init__
        assert r.signature_str == "()"

    def test_parse_pydantic_model_with_init(self):
        """Can parse Pydantic model with custom __init__."""
        source = '''
from pydantic import BaseModel
from vero.core.resource import resource

@resource("schemas")
class CustomSchema(BaseModel):
    """Schema with custom init."""
    name: str

    def __init__(self, name: str, extra: int = 0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.extra = extra
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.namespace == "schemas"
        assert "name: str" in r.signature_str
        assert "extra: int" in r.signature_str


class TestMethodResourceDiscovery:
    """Test discovery of @resource decorated methods."""

    def test_parse_method(self):
        """Can parse a @resource decorated method."""
        source = '''
from vero.core.resource import resource

class Evaluators:
    """Evaluator collection."""

    @resource("evaluators")
    def score(self, output: str, expected: str) -> float:
        """Score output against expected."""
        return 1.0 if output == expected else 0.0
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.namespace == "evaluators"
        assert r.name == "score"
        assert r.target_name == "score"
        assert r.kind == "method"
        assert r.class_name == "Evaluators"
        assert r.docstring == "Score output against expected."
        # self should be excluded from signature
        assert "self" not in r.signature_str
        assert "output: str" in r.signature_str
        assert "expected: str" in r.signature_str
        assert "float" in r.signature_str

    def test_parse_method_with_custom_name(self):
        """Can parse method with custom resource name."""
        source = """
from vero.core.resource import resource

class Tools:
    @resource("tools", name="custom_tool")
    def internal_method(self, data: dict) -> list:
        return []
"""
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        assert resources[0].name == "custom_tool"
        assert resources[0].target_name == "internal_method"
        assert resources[0].kind == "method"
        assert resources[0].class_name == "Tools"

    def test_parse_classmethod(self):
        """Can parse @resource decorated classmethod."""
        source = '''
from vero.core.resource import resource

class Factory:
    @classmethod
    @resource("factories")
    def create(cls, config: dict) -> "Factory":
        """Create instance from config."""
        return cls()
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.namespace == "factories"
        assert r.name == "create"
        assert r.kind == "method"
        assert r.class_name == "Factory"
        # cls should be excluded
        assert "cls" not in r.signature_str
        assert "config: dict" in r.signature_str

    def test_parse_staticmethod(self):
        """Can parse @resource decorated staticmethod."""
        source = '''
from vero.core.resource import resource

class Utils:
    @staticmethod
    @resource("utils")
    def helper(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.namespace == "utils"
        assert r.name == "helper"
        assert r.kind == "method"
        # staticmethod has no self/cls, but we still mark as method
        # The signature should include both params
        assert "x: int" in r.signature_str
        assert "y: int" in r.signature_str

    def test_parse_async_method(self):
        """Can parse async method."""
        source = '''
from vero.core.resource import resource

class AsyncHandlers:
    @resource("handlers")
    async def handle_request(self, request: dict) -> dict:
        """Handle async request."""
        return {}
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 1
        r = resources[0]
        assert r.name == "handle_request"
        assert r.kind == "method"
        assert "self" not in r.signature_str
        assert "request: dict" in r.signature_str

    def test_parse_multiple_methods(self):
        """Can parse multiple methods in one class."""
        source = """
from vero.core.resource import resource

class Evaluators:
    @resource("eval")
    def exact_match(self, a: str, b: str) -> bool:
        return a == b

    def not_a_resource(self):
        pass

    @resource("eval")
    def fuzzy_match(self, a: str, b: str) -> float:
        return 0.5
"""
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 2
        names = {r.name for r in resources}
        assert "exact_match" in names
        assert "fuzzy_match" in names
        for r in resources:
            assert r.kind == "method"
            assert r.class_name == "Evaluators"

    def test_parse_class_and_methods(self):
        """Can parse both class and methods with @resource."""
        source = '''
from vero.core.resource import resource

@resource("models")
class MyModel:
    """A model with resources."""

    @resource("methods")
    def process(self, data: str) -> str:
        """Process data."""
        return data
'''
        resources = ResourceDiscovery._parse_resources_from_source(
            source,
            file_path=Path("/fake/path.py"),
            module="fake.module",
        )

        assert len(resources) == 2

        class_resources = [r for r in resources if r.kind == "class"]
        method_resources = [r for r in resources if r.kind == "method"]

        assert len(class_resources) == 1
        assert len(method_resources) == 1

        assert class_resources[0].name == "MyModel"
        assert class_resources[0].namespace == "models"

        assert method_resources[0].name == "process"
        assert method_resources[0].namespace == "methods"
        assert method_resources[0].class_name == "MyModel"


class TestDecoratorRuntime:
    """Test that decorator works correctly at runtime with classes/methods."""

    def test_class_decorator_passthrough(self):
        """Class decorator returns original class unchanged."""

        @resource("test")
        class MyClass:
            def __init__(self, x: int):
                self.x = x

        instance = MyClass(42)
        assert instance.x == 42
        assert MyClass.__name__ == "MyClass"

    def test_dataclass_decorator_works(self):
        """@resource works with @dataclass."""
        from dataclasses import dataclass

        @resource("test")
        @dataclass
        class Config:
            name: str
            value: int = 0

        config = Config(name="test", value=10)
        assert config.name == "test"
        assert config.value == 10

    def test_method_decorator_passthrough(self):
        """Method decorator returns original method unchanged."""

        class Evaluator:
            @resource("eval")
            def score(self, x: int) -> int:
                return x * 2

        e = Evaluator()
        assert e.score(5) == 10
        assert e.score.__name__ == "score"
