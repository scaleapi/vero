"""VeroResource: A callable abstraction with namespace, signature, and git-aware introspection.

Resources are discovered via AST parsing (no imports required), making them
safe to inspect at any git commit without executing code.

Usage:
    from vero.core.resource import resource, ResourceStore

    # Mark functions as resources with @resource decorator
    @resource("my_namespace")
    def my_processor(data: dict) -> str:
        '''Process data and return result.'''
        return str(data)

    @resource("my_namespace", name="custom_name")
    def another_func(x: int, y: int) -> int:
        return x + y

    # Mark classes as resources
    @resource("models")
    class MyModel:
        '''A model resource.'''
        def __init__(self, config: dict):
            self.config = config

    # Mark methods as resources (class itself not decorated)
    class Evaluators:
        @resource("evaluators")
        def score(self, output: str) -> float:
            return 1.0

    # Create a store to discover and cache resources
    store = ResourceStore(
        repo_path=Path("/path/to/repo"),
        package_rel_path="src/mypackage",
    )

    # Get resources at HEAD (discovers lazily)
    resources = store.get_resources("HEAD")

    # Get resources at any commit
    resources = store.get_resources("abc123")

    # Get a specific resource
    resource = store.get_resource("my_namespace.my_processor", "HEAD")
"""

from __future__ import annotations

import ast
import logging
import subprocess
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Callable, Literal, ParamSpec, TypeVar, overload

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound=type)

ResourceKind = Literal["function", "class", "method"]


@overload
def resource(
    namespace: str,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def resource(
    namespace: str,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[T], T]: ...


def resource(
    namespace: str,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[[T], T]:
    """Decorator to mark a function, method, or class as a VeroResource.

    This is a marker decorator - it doesn't modify the target or
    register it at runtime. Resources are discovered via AST parsing
    by ResourceStore.

    Args:
        namespace: Namespace to group the resource under
        name: Optional name (defaults to function/class name)
        description: Optional description (extracted from docstring if not provided)

    Returns:
        The original function/class, unchanged

    Usage:
        @resource("evaluators")
        def score_output(output: str, expected: str) -> float:
            '''Score model output against expected.'''
            return 1.0 if output == expected else 0.0

        @resource("models")
        class MyModel:
            '''A model resource.'''
            def __init__(self, config: dict):
                self.config = config

        class Evaluators:
            @resource("evaluators")
            def score(self, output: str) -> float:
                return 1.0
    """

    # The decorator is a no-op marker - discovery happens via AST parsing
    def decorator(target: Callable[P, R] | T) -> Callable[P, R] | T:
        return target

    return decorator


@dataclass(frozen=True, slots=True)
class StaticResourceInfo:
    """Resource metadata extracted via AST parsing.

    This is the primary resource data type. All fields are extracted
    statically from source code without importing.
    """

    namespace: str
    name: str
    target_name: str  # The actual function/class/method name in code
    file_path: Path
    line_number: int
    module: str
    signature_str: str  # e.g., "(x: int, y: str) -> float"
    docstring: str | None
    source: str
    kind: ResourceKind = "function"  # "function", "class", or "method"
    class_name: str | None = None  # Parent class name for methods

    @property
    def qualified_name(self) -> str:
        """Full qualified name: namespace.name."""
        return f"{self.namespace}.{self.name}"

    @property
    def description(self) -> str:
        """First line of docstring, or empty string."""
        if not self.docstring:
            return ""
        return self.docstring.split("\n")[0].strip()

    @property
    def function_name(self) -> str:
        """Alias for target_name (backwards compatibility)."""
        return self.target_name


class ResourceDiscovery:
    """AST-based discovery of @resource decorated functions."""

    @classmethod
    def discover_at_commit(
        cls,
        repo_path: PathLike[str] | str,
        commit: str,
        package_rel_path: str,
    ) -> list[StaticResourceInfo]:
        """Discover resources at a git commit using AST parsing.

        This does NOT import code - it parses Python files to find
        @resource decorators and extract metadata statically.

        Args:
            repo_path: Path to the git repository root
            commit: Git commit hash or reference
            package_rel_path: Relative path from repo root to the package

        Returns:
            List of StaticResourceInfo

        Example:
            resources = ResourceDiscovery.discover_at_commit(
                repo_path=Path("/code/myproject"),
                commit="abc123",
                package_rel_path="src/mypackage",
            )
        """
        repo_path = Path(repo_path).resolve()
        resources: list[StaticResourceInfo] = []

        # List Python files at the commit
        try:
            result = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", commit, package_rel_path],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to list files at commit {commit}: {e.stderr}")

        py_files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f]

        for rel_file_path in py_files:
            # Get file content at commit
            try:
                content_result = subprocess.run(
                    ["git", "show", f"{commit}:{rel_file_path}"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                file_content = content_result.stdout
            except subprocess.CalledProcessError:
                continue

            # Parse and extract resources
            file_resources = cls._parse_resources_from_source(
                file_content,
                file_path=repo_path / rel_file_path,
                module=cls._path_to_module(rel_file_path, package_rel_path),
            )
            resources.extend(file_resources)

        return resources

    @classmethod
    def _path_to_module(cls, file_path: str, package_rel_path: str) -> str:
        """Convert file path to module name."""
        if file_path.startswith(package_rel_path):
            rel = file_path[len(package_rel_path) :].lstrip("/")
        else:
            rel = file_path

        parts = rel.replace(".py", "").split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else "<unknown>"

    @classmethod
    def _parse_resources_from_source(
        cls,
        source: str,
        file_path: Path,
        module: str,
    ) -> list[StaticResourceInfo]:
        """Parse Python source to find @resource decorated functions, classes, and methods."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        resources: list[StaticResourceInfo] = []
        lines = source.splitlines()

        for node in ast.iter_child_nodes(tree):
            # Top-level functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                resource_info = cls._extract_resource_decorator(node)
                if resource_info:
                    resources.append(
                        cls._create_resource_info(
                            node, resource_info, lines, file_path, module, kind="function"
                        )
                    )

            # Classes
            elif isinstance(node, ast.ClassDef):
                # Check if class itself is decorated
                resource_info = cls._extract_resource_decorator(node)
                if resource_info:
                    resources.append(
                        cls._create_resource_info(
                            node, resource_info, lines, file_path, module, kind="class"
                        )
                    )

                # Check methods within the class
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_resource_info = cls._extract_resource_decorator(child)
                        if method_resource_info:
                            resources.append(
                                cls._create_resource_info(
                                    child,
                                    method_resource_info,
                                    lines,
                                    file_path,
                                    module,
                                    kind="method",
                                    class_name=node.name,
                                )
                            )

        return resources

    @classmethod
    def _create_resource_info(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        resource_info: tuple[str, str | None],
        lines: list[str],
        file_path: Path,
        module: str,
        kind: ResourceKind,
        class_name: str | None = None,
    ) -> StaticResourceInfo:
        """Create a StaticResourceInfo from an AST node."""
        namespace, name = resource_info
        name = name or node.name

        # Extract signature
        sig_str = cls._extract_signature_str(node, is_method=(kind == "method"))

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract source lines (includes decorators)
        first_decorator_line = node.lineno
        for dec in node.decorator_list:
            first_decorator_line = min(first_decorator_line, dec.lineno)

        start = first_decorator_line - 1
        end = node.end_lineno or start + 1
        source = "\n".join(lines[start:end])

        return StaticResourceInfo(
            namespace=namespace,
            name=name,
            target_name=node.name,
            file_path=file_path,
            line_number=first_decorator_line,
            module=module,
            signature_str=sig_str,
            docstring=docstring,
            source=source,
            kind=kind,
            class_name=class_name,
        )

    @classmethod
    def _extract_resource_decorator(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> tuple[str, str | None] | None:
        """Extract namespace and name from @resource decorator if present.

        Returns (namespace, name) or None if no @resource decorator.
        """
        for decorator in node.decorator_list:
            # Handle @resource("ns") or @resource("ns", name="x")
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name) and func.id == "resource":
                    return cls._parse_resource_call(decorator)
                if isinstance(func, ast.Attribute) and func.attr == "resource":
                    return cls._parse_resource_call(decorator)
        return None

    @classmethod
    def _parse_resource_call(cls, call: ast.Call) -> tuple[str, str | None] | None:
        """Parse @resource(...) call to extract namespace and name."""
        namespace = None
        name = None

        # First positional arg is namespace
        if call.args and isinstance(call.args[0], ast.Constant):
            namespace = call.args[0].value

        # Check keyword arguments
        for kw in call.keywords:
            if kw.arg == "namespace" and isinstance(kw.value, ast.Constant):
                namespace = kw.value.value
            elif kw.arg == "name" and isinstance(kw.value, ast.Constant):
                name = kw.value.value

        # Second positional arg could be name
        if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
            name = call.args[1].value

        if namespace is None:
            return None

        return (namespace, name)

    @classmethod
    def _extract_signature_str(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        is_method: bool = False,
    ) -> str:
        """Extract signature as a string from AST.

        For functions/methods: extracts parameter list and return type.
        For classes: extracts __init__ parameters (excluding self).
        For methods: skips self/cls first parameter if present.
        """
        if isinstance(node, ast.ClassDef):
            # Find __init__ method and extract its signature
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                    return cls._extract_signature_str(child, is_method=True)
            return "()"  # No __init__ found

        # Function or method
        args = node.args.args

        # Skip self/cls only if it's actually the first param name
        if is_method and args and args[0].arg in ("self", "cls"):
            args = args[1:]

        params = []
        for arg in args:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            params.append(param)

        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"

        return f"({', '.join(params)}){returns}"


@dataclass
class ResourceStore:
    """Manages discovered resources across git commits.

    Provides lazy discovery and caching of resources per commit.
    All discovery is AST-based (no imports).

    Usage:
        store = ResourceStore(
            repo_path=Path("/code/myproject"),
            package_rel_path="src/mypackage",
        )

        # Get resources at HEAD (discovers lazily)
        resources = store.get_resources("HEAD")

        # Get resources at specific commit
        resources = store.get_resources("abc123")

        # After creating a new commit, trigger rediscovery
        store.on_commit_created("def456")
    """

    repo_path: Path
    package_rel_path: str
    _cache: dict[str, list[StaticResourceInfo]] = field(default_factory=dict)
    _commit_index: dict[str, dict[str, StaticResourceInfo]] = field(default_factory=dict)

    def __post_init__(self):
        self.repo_path = Path(self.repo_path).resolve()

    def _resolve_commit(self, commit: str) -> str:
        """Resolve a commit reference to its full hash."""
        result = subprocess.run(
            ["git", "rev-parse", commit],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"Invalid commit reference '{commit}': {result.stderr.strip()}")
        return result.stdout.strip()

    def _discover(self, commit: str) -> list[StaticResourceInfo]:
        """Run AST-based discovery for a commit."""
        resolved = self._resolve_commit(commit)

        if resolved in self._cache:
            return self._cache[resolved]

        logger.info(f"Discovering resources at commit {resolved[:8]}...")

        resources = ResourceDiscovery.discover_at_commit(
            self.repo_path,
            resolved,
            self.package_rel_path,
        )

        # Cache by resolved hash
        self._cache[resolved] = resources

        # Build index for fast lookup
        self._commit_index[resolved] = {r.qualified_name: r for r in resources}

        logger.info(f"Discovered {len(resources)} resources at commit {resolved[:8]}")

        return resources

    def get_resources(self, commit: str = "HEAD") -> list[StaticResourceInfo]:
        """Get all resources at a commit, discovering lazily if needed.

        Args:
            commit: Git commit hash or reference (default: HEAD)

        Returns:
            List of StaticResourceInfo for resources at that commit
        """
        return self._discover(commit)

    def get_resource(
        self,
        qualified_name: str,
        commit: str = "HEAD",
    ) -> StaticResourceInfo | None:
        """Get a specific resource by qualified name at a commit.

        Args:
            qualified_name: The full qualified name (namespace.name)
            commit: Git commit hash or reference

        Returns:
            StaticResourceInfo or None if not found
        """
        resolved = self._resolve_commit(commit)

        # Ensure discovered
        if resolved not in self._commit_index:
            self._discover(commit)

        return self._commit_index.get(resolved, {}).get(qualified_name)

    def get_resource_by_parts(
        self,
        namespace: str,
        name: str,
        commit: str = "HEAD",
    ) -> StaticResourceInfo | None:
        """Get a specific resource by namespace and name at a commit."""
        return self.get_resource(f"{namespace}.{name}", commit)

    def list_namespaces(self, commit: str = "HEAD") -> list[str]:
        """List all namespaces at a commit."""
        resources = self.get_resources(commit)
        return sorted(set(r.namespace for r in resources))

    def list_namespace(
        self,
        namespace: str,
        commit: str = "HEAD",
    ) -> list[StaticResourceInfo]:
        """List all resources in a namespace at a commit."""
        resources = self.get_resources(commit)
        return [r for r in resources if r.namespace == namespace]

    def on_commit_created(self, commit: str) -> list[StaticResourceInfo]:
        """Called when a new commit is created. Triggers discovery.

        Args:
            commit: The new commit hash

        Returns:
            List of discovered resources at the new commit
        """
        # Force fresh discovery (don't use cache even if somehow present)
        resolved = self._resolve_commit(commit)
        self._cache.pop(resolved, None)
        self._commit_index.pop(resolved, None)

        return self._discover(commit)

    def invalidate(self, commit: str | None = None) -> None:
        """Invalidate cached discovery results.

        Args:
            commit: Specific commit to invalidate, or None to clear all
        """
        if commit is None:
            self._cache.clear()
            self._commit_index.clear()
        else:
            try:
                resolved = self._resolve_commit(commit)
                self._cache.pop(resolved, None)
                self._commit_index.pop(resolved, None)
            except ValueError:
                pass

    def is_cached(self, commit: str) -> bool:
        """Check if a commit's resources are already cached."""
        try:
            resolved = self._resolve_commit(commit)
            return resolved in self._cache
        except ValueError:
            return False

    def cached_commits(self) -> list[str]:
        """List all commits with cached discovery results."""
        return list(self._cache.keys())
