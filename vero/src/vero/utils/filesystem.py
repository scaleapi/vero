"""Filesystem visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vero.filesystem import AccessType
from vero.sandbox import Sandbox

if TYPE_CHECKING:
    from rich.console import Console
    from rich.tree import Tree


def visualize_access(
    fs: Sandbox,
    path: str | Path = ".",
    max_depth: int = 3,
    offset: int = 0,
    limit: int | None = None,
    show: set[AccessType] | None = None,
    console: Console | None = None,
) -> tuple[Tree, int]:
    """Visualize file access permissions as a rich Tree.

    Enumerates files in the filesystem and displays their access levels
    using color-coded indicators:
    - 🔴 (red): excluded
    - 🟡 (yellow): read-only
    - 🟢 (green): writeable

    Args:
        fs: The Filesystem to visualize.
        path: Relative path within root to start from. Defaults to ".".
        max_depth: Maximum depth to traverse. Defaults to 3.
        offset: Number of files to skip from the beginning. Defaults to 0.
        limit: Maximum number of files to display. None means no limit.
        show: Set of access types to display. None means show all.
        console: Optional Console instance to print to. If None, creates one.

    Returns:
        A tuple of (Tree, total_count) where total_count is the total number
        of files found (before pagination, after filtering by access type).
    """
    from rich.console import Console as RichConsole
    from rich.tree import Tree

    if show is None:
        show = {AccessType.EXCLUDE, AccessType.READ, AccessType.WRITE}

    start_path = Path(fs.root, path).resolve()
    if not start_path.exists():
        raise ValueError(f"Path does not exist: {start_path}")

    access_styles = {
        AccessType.EXCLUDE: ("🔴", "red"),
        AccessType.READ: ("🟡", "yellow"),
        AccessType.WRITE: ("🟢", "green"),
    }

    file_count = 0
    files_shown = 0
    end_idx = offset + limit if limit is not None else None

    def build_tree(dir_path: Path, parent: Tree, current_depth: int) -> int:
        nonlocal file_count, files_shown

        if current_depth > max_depth:
            return 0

        subtree_files = 0
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name))
            for entry in entries:
                if entry.is_file():
                    access = fs.get_access(entry)
                    if access not in show:
                        continue
                    subtree_files += 1
                    file_count += 1
                    if file_count <= offset:
                        continue
                    if end_idx is not None and file_count > end_idx:
                        continue
                    _, color = access_styles[access]
                    parent.add(f"[{color}]{entry.name}[/{color}]")
                    files_shown += 1
                elif entry.is_dir():
                    dir_access = fs.get_access(entry)
                    dir_icon, dir_color = access_styles[dir_access]
                    subtree = Tree(f"{dir_icon} 📁 [{dir_color}]{entry.name}/[/{dir_color}]")
                    child_files = build_tree(entry, subtree, current_depth + 1)
                    if child_files > 0:
                        parent.add(subtree)
                    subtree_files += child_files
        except PermissionError:
            pass

        return subtree_files

    rel_start = start_path.relative_to(Path(fs.root)) if str(start_path) != fs.root else Path(".")
    root_access = fs.get_access(start_path)
    root_icon, root_color = access_styles[root_access]
    tree = Tree(f"{root_icon} 📁 [{root_color}]{rel_start}/[/{root_color}]")
    total_count = build_tree(start_path, tree, 1)

    tree.label = (
        f"{root_icon} 📁 [{root_color}]{rel_start}/[/{root_color}] "
        f"(showing {files_shown}/{total_count} files, depth≤{max_depth}, offset={offset})"
    )

    if console is None:
        console = RichConsole()

    console.print("[red]● exclude[/red]  [yellow]● read-only[/yellow]  [green]● writeable[/green]")
    console.print(tree)

    return tree, total_count
