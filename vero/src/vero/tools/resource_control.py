"""Tool for LLMs to list, get, and modify VeroResources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from vero.core.resource import ResourceDiscovery, ResourceStore, StaticResourceInfo
from vero.exceptions import StringNotFoundError
from vero.tools.base import FileSystemWriteBase
from vero.tools.utils import is_tool


class ResourceEditResult(NamedTuple):
    """Result of editing a resource."""

    message: str
    qualified_name: str
    file_path: str


@dataclass
class ResourceControl(FileSystemWriteBase):
    """List, get, and modify resources in the codebase with automatic commits."""

    allowed_namespaces: set[str] | None = None
    content_char_limit: int = 500_000

    # Runtime fields — set during bind()
    package_rel_path: str | None = None
    store: ResourceStore | None = None

    def bind(self, session) -> None:
        super().bind(session)
        root = self.workspace.root.rstrip("/") + "/"
        project = self.workspace.project_path
        self.package_rel_path = project[len(root):] if project.startswith(root) else "."
        self.store = ResourceStore(
            repo_path=self.workspace.root, package_rel_path=self.package_rel_path
        )

    def _is_namespace_allowed(self, namespace: str) -> bool:
        """Check if a namespace is in the allowed list."""
        if self.allowed_namespaces is None:
            return True
        return namespace in self.allowed_namespaces

    def _filter_allowed_namespaces(self, namespaces: list[str]) -> list[str]:
        """Filter namespaces to only include allowed ones."""
        if self.allowed_namespaces is None:
            return namespaces
        return [ns for ns in namespaces if ns in self.allowed_namespaces]

    def _filter_allowed_resources(
        self, resources: list[StaticResourceInfo]
    ) -> list[StaticResourceInfo]:
        """Filter resources to only include those in allowed namespaces."""
        if self.allowed_namespaces is None:
            return resources
        return [r for r in resources if r.namespace in self.allowed_namespaces]

    def _namespace_denied_error(self, namespace: str) -> str:
        """Return error message for denied namespace access."""
        allowed = sorted(self.allowed_namespaces) if self.allowed_namespaces else []
        return (
            f"Access denied: namespace '{namespace}' is not in allowed namespaces. "
            f"Allowed: {', '.join(allowed)}"
        )

    @is_tool
    def readme(self) -> str:
        """Return a README string for the resource control tool."""

        return """# What Are Resources?

Resources are Python functions, classes, or methods marked with the `@resource("namespace")` decorator. They represent the mutable parts of the agent codebase that you can modify. Each resource has:
- **Namespace**: A grouping category (e.g., "prompts", "tools", "evaluators")
- **Name**: The function/class/method name
- **Qualified name**: The full identifier (`namespace.name`)

Resources are discovered via AST parsing without executing code, allowing safe inspection at any git commit."""

    @is_tool
    async def list_resources(
        self,
        namespace: str | None = None,
        commit: str = "HEAD",
    ) -> str:
        """List all resources, optionally filtered by namespace.

        Args:
            namespace: Optional namespace to filter by. If not provided, lists all allowed.
            commit: Git commit to list resources at (default: HEAD)

        Returns:
            Formatted string listing resources with signatures and locations.
        """
        try:
            if namespace:
                # Check if namespace is allowed
                if not self._is_namespace_allowed(namespace):
                    return self._namespace_denied_error(namespace)

                resources = self.store.list_namespace(namespace, commit)
                if not resources:
                    namespaces = self._filter_allowed_namespaces(
                        self.store.list_namespaces(commit)
                    )
                    if namespaces:
                        return (
                            f"No resources found in namespace '{namespace}' at {commit}. "
                            f"Available namespaces: {', '.join(namespaces)}"
                        )
                    return f"No resources found in namespace '{namespace}' at {commit}."
            else:
                resources = self._filter_allowed_resources(
                    self.store.get_resources(commit)
                )
                if not resources:
                    return f"No resources found at {commit}."
        except ValueError as e:
            return f"Error: {e}"

        commit_info = f" at {commit}" if commit != "HEAD" else ""
        lines = [
            f"Found {len(resources)} resource(s)"
            + (f" in namespace '{namespace}'" if namespace else "")
            + f"{commit_info}:\n"
        ]

        for r in resources:
            lines.append(f"  • {r.qualified_name}")
            lines.append(f"    Signature: {r.signature_str}")
            lines.append(f"    Location: {r.file_path.name}:{r.line_number}")
            if r.docstring:
                lines.append(f"    Description: {r.docstring.split(chr(10))[0]}")
            lines.append("")

        return "\n".join(lines)

    @is_tool
    async def list_namespaces(self, commit: str = "HEAD") -> str:
        """List all registered resource namespaces.

        Args:
            commit: Git commit to list namespaces at (default: HEAD)

        Returns:
            Formatted string listing all allowed namespaces and their resource counts.
        """
        try:
            namespaces = self._filter_allowed_namespaces(
                self.store.list_namespaces(commit)
            )
        except ValueError as e:
            return f"Error: {e}"

        if not namespaces:
            return f"No namespaces found at {commit}."

        commit_info = f" at {commit}" if commit != "HEAD" else ""
        lines = [f"Found {len(namespaces)} namespace(s){commit_info}:\n"]

        for ns in namespaces:
            count = len(self.store.list_namespace(ns, commit))
            lines.append(f"  • {ns} ({count} resource{'s' if count != 1 else ''})")

        return "\n".join(lines)

    @is_tool
    async def get_resource(
        self,
        namespace: str,
        name: str,
        commit: str = "HEAD",
    ) -> str:
        """Get detailed information about a resource, including its source code.

        Args:
            namespace: The resource namespace
            name: The resource name within the namespace
            commit: Git commit to get the resource at (default: HEAD)

        Returns:
            Formatted string with resource details and source code.
        """
        # Check if namespace is allowed
        if not self._is_namespace_allowed(namespace):
            return self._namespace_denied_error(namespace)

        try:
            resource = self.store.get_resource_by_parts(namespace, name, commit)
        except ValueError as e:
            return f"Error: {e}"

        if resource is None:
            available = self.store.list_namespace(namespace, commit)
            if available:
                names = [r.name for r in available]
                return (
                    f"Resource '{name}' not found in namespace '{namespace}' at {commit}. "
                    f"Available: {', '.join(names)}"
                )
            namespaces = self._filter_allowed_namespaces(
                self.store.list_namespaces(commit)
            )
            if namespaces:
                return (
                    f"Namespace '{namespace}' not found at {commit}. "
                    f"Available namespaces: {', '.join(namespaces)}"
                )
            return f"No resources found at {commit}."

        return self._format_resource(resource, commit)

    @is_tool
    async def get_resource_by_qualified_name(
        self,
        qualified_name: str,
        commit: str = "HEAD",
    ) -> str:
        """Get detailed information about a resource by its qualified name.

        Args:
            qualified_name: The full qualified name (namespace.name)
            commit: Git commit to get the resource at (default: HEAD)

        Returns:
            Formatted string with resource details and source code.
        """
        parts = qualified_name.split(".", 1)
        if len(parts) != 2:
            return (
                f"Invalid qualified name '{qualified_name}'. "
                f"Expected format: 'namespace.name'"
            )

        # Namespace check happens in get_resource
        return await self.get_resource(parts[0], parts[1], commit)

    def _format_resource(self, resource: StaticResourceInfo, commit: str) -> str:
        """Format a resource for display."""
        commit_info = f" (at {commit})" if commit != "HEAD" else ""
        lines = [
            f"Resource: {resource.qualified_name}{commit_info}",
            f"Signature: {resource.signature_str}",
            f"Description: {resource.docstring.split(chr(10))[0] if resource.docstring else '(none)'}",
            f"Location: {resource.file_path}:{resource.line_number}",
            f"Module: {resource.module}",
            "\nSource:",
            "```python",
            resource.source,
            "```",
        ]
        return "\n".join(lines)

    def _validate_resource_integrity(
        self,
        old_content: str,
        new_content: str,
        file_path: str | Path,
        expected_qualified_name: str,
    ) -> None:
        """Validate resource decorators are preserved and no new ones added.

        Raises:
            ValueError: If resource decorator was removed, changed, or new ones added
        """
        # Parse both old and new content
        old_resources = ResourceDiscovery._parse_resources_from_source(
            old_content,
            file_path=file_path,
            module="<validation>",
        )
        new_resources = ResourceDiscovery._parse_resources_from_source(
            new_content,
            file_path=file_path,
            module="<validation>",
        )

        old_names = {r.qualified_name for r in old_resources}
        new_names = {r.qualified_name for r in new_resources}

        # Check the target resource still exists
        if expected_qualified_name not in new_names:
            raise ValueError(
                f"Edit rejected: the @resource decorator for '{expected_qualified_name}' "
                f"was removed or its namespace/name was changed. "
                f"Resource identity must be preserved during edits."
            )

        # Check no new resources were added
        added_resources = new_names - old_names
        if added_resources:
            raise ValueError(
                f"Edit rejected: new @resource decorator(s) cannot be added. "
                f"Attempted to add: {', '.join(sorted(added_resources))}"
            )

        # Check no existing resources were removed (other than potentially renamed ones)
        removed_resources = old_names - new_names
        if removed_resources:
            raise ValueError(
                f"Edit rejected: existing @resource decorator(s) cannot be removed. "
                f"Attempted to remove: {', '.join(sorted(removed_resources))}"
            )

    async def _edit_resource(
        self,
        resource: StaticResourceInfo,
        old_string: str,
        new_string: str,
    ) -> ResourceEditResult:
        """Helper to edit a resource's source code."""
        file_path = resource.file_path

        # Validate the file is within our filesystem
        absolute_path = self.workspace.validate_write(str(file_path))

        if not await self.sandbox.exists(absolute_path):
            raise FileNotFoundError(f"Resource file '{absolute_path}' does not exist.")

        if len(new_string) > self.content_char_limit:
            raise ValueError(
                f"new_string is too long. Must be less than {self.content_char_limit} characters."
            )

        if old_string == new_string:
            raise ValueError(
                "old_string and new_string are identical. No changes made."
            )

        # Read current file content
        content = await self.sandbox.read_file(absolute_path)

        # Validate old_string exists
        if old_string not in content:
            raise StringNotFoundError(
                f"The string to replace was not found in '{absolute_path}'. "
                f"The resource may have changed. Try getting the latest source first."
            )

        # Perform replacement (single occurrence only for safety)
        new_content = content.replace(old_string, new_string, 1)

        # Validate resource decorators are preserved and no new ones added
        self._validate_resource_integrity(
            old_content=content,
            new_content=new_content,
            file_path=Path(absolute_path),
            expected_qualified_name=resource.qualified_name,
        )

        # Write back (only after validation passes)
        await self.sandbox.write_file(absolute_path, new_content)

        return ResourceEditResult(
            message=f"Successfully edited resource '{resource.qualified_name}'",
            qualified_name=resource.qualified_name,
            file_path=str(absolute_path),
        )

    @is_tool
    async def edit_resource(
        self,
        commit_message: str,
        namespace: str,
        name: str,
        old_string: str,
        new_string: str,
    ) -> str:
        """Edit a resource's source code by replacing text.

        Performs a search-and-replace within the resource's file,
        commits the change, and triggers rediscovery.

        Args:
            commit_message: The message for the commit
            namespace: The resource namespace
            name: The resource name within the namespace
            old_string: The text to find and replace
            new_string: The replacement text

        Returns:
            String message indicating success, new commit hash, and updated resource info.
        """
        # Check if namespace is allowed
        if not self._is_namespace_allowed(namespace):
            return self._namespace_denied_error(namespace)

        # Get current resource at HEAD
        resource = self.store.get_resource_by_parts(namespace, name, "HEAD")

        if resource is None:
            available = self.store.list_namespace(namespace, "HEAD")
            if available:
                names = [r.name for r in available]
                return (
                    f"Resource '{name}' not found in namespace '{namespace}'. "
                    f"Available: {', '.join(names)}"
                )
            return f"Namespace '{namespace}' not found."

        output = await self.run_and_commit(
            self._edit_resource(resource, old_string, new_string),
            commit_message,
        )

        # Trigger rediscovery at the new commit
        new_resources = self.store.on_commit_created(output.commit)

        # Find the updated resource
        updated = next(
            (r for r in new_resources if r.qualified_name == resource.qualified_name),
            None,
        )

        result_msg = f"Created commit {output.commit}. {output.result.message}"
        if updated:
            result_msg += f"\n\nUpdated resource now at line {updated.line_number}."

        return result_msg

    @is_tool
    async def edit_resource_by_qualified_name(
        self,
        commit_message: str,
        qualified_name: str,
        old_string: str,
        new_string: str,
    ) -> str:
        """Edit a resource's source code by its qualified name.

        Args:
            commit_message: The message for the commit
            qualified_name: The full qualified name (namespace.name)
            old_string: The text to find and replace
            new_string: The replacement text

        Returns:
            String message indicating success and the new commit hash.
        """
        parts = qualified_name.split(".", 1)
        if len(parts) != 2:
            return (
                f"Invalid qualified name '{qualified_name}'. "
                f"Expected format: 'namespace.name'"
            )

        return await self.edit_resource(
            commit_message, parts[0], parts[1], old_string, new_string
        )

    @is_tool
    async def compare_resource(
        self,
        qualified_name: str,
        commit_a: str,
        commit_b: str = "HEAD",
    ) -> str:
        """Compare a resource's source code between two commits.

        Args:
            qualified_name: The full qualified name (namespace.name)
            commit_a: First commit to compare
            commit_b: Second commit to compare (default: HEAD)

        Returns:
            Formatted string showing the resource at both commits.
        """
        # Extract and check namespace
        parts = qualified_name.split(".", 1)
        if len(parts) != 2:
            return (
                f"Invalid qualified name '{qualified_name}'. "
                f"Expected format: 'namespace.name'"
            )

        namespace = parts[0]
        if not self._is_namespace_allowed(namespace):
            return self._namespace_denied_error(namespace)

        try:
            resource_a = self.store.get_resource(qualified_name, commit_a)
            resource_b = self.store.get_resource(qualified_name, commit_b)
        except ValueError as e:
            return f"Error: {e}"

        lines = [f"Comparing '{qualified_name}':\n"]

        if resource_a is None:
            lines.append(f"--- Not found at {commit_a}")
        else:
            lines.append(f"--- At {commit_a} (line {resource_a.line_number}):")
            lines.append("```python")
            lines.append(resource_a.source)
            lines.append("```\n")

        if resource_b is None:
            lines.append(f"+++ Not found at {commit_b}")
        else:
            lines.append(f"+++ At {commit_b} (line {resource_b.line_number}):")
            lines.append("```python")
            lines.append(resource_b.source)
            lines.append("```")

        return "\n".join(lines)

    @is_tool
    async def list_cached_commits(self) -> str:
        """List all commits with cached discovery results.

        Returns:
            Formatted string listing cached commits.
        """
        commits = self.store.cached_commits()

        if not commits:
            return "No commits cached. Resources will be discovered lazily on first access."

        lines = [f"Cached discovery results for {len(commits)} commit(s):\n"]
        for c in commits:
            count = len(self.store.get_resources(c))
            lines.append(f"  • {c[:8]} ({count} resources)")

        return "\n".join(lines)

    @is_tool
    async def invalidate_cache(self, commit: str | None = None) -> str:
        """Invalidate cached discovery results.

        Args:
            commit: Specific commit to invalidate, or None to clear all caches.

        Returns:
            Confirmation message.
        """
        if commit:
            self.store.invalidate(commit)
            return f"Invalidated cache for commit {commit}."
        else:
            self.store.invalidate()
            return "Invalidated all cached discovery results."
