"""Resolve which symbol an edit landed in, from the syntax tree.

Git's hunk header is not good enough. Its Python `xfuncname` matches the last
preceding definition line, which for a method inside a class reports the *class* — so
a one-line fix to a shell-exec helper is attributed to the whole agent. Parsing the
post-image and mapping changed line numbers to the innermost enclosing definition
gives function-level locus exactly, with no model involved.

Module-level bindings get split further by target name and value shape, because that
bucket is otherwise the largest and least informative: on one real candidate it held
170 changed lines spanning a system prompt, the tool table, nine tuning constants and
six regexes, which are four different kinds of modification.
"""

from __future__ import annotations

import ast
import re

from vero.interpret.models import SymbolKind

_HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.M)
_PROMPT_MIN_CHARS = 200


def changed_lines(diff: str) -> set[int]:
    """Post-image line numbers touched by a `-U0` diff of a single file.

    A `+start,0` hunk is a deletion with nothing added, and `start` is the surviving
    line *before* the removal, not a changed one. Counting it invents a post-image
    line and hands the deletion to whatever symbol now sits there, so these hunks
    contribute nothing here; `decompose` carries their removals on the module row.
    """
    lines: set[int] = set()
    for match in _HUNK.finditer(diff):
        start = int(match.group(1))
        count = int(match.group(2) or 1)
        if count == 0:
            continue
        lines.update(range(start, start + count))
    return lines


def _binding_kind(value: ast.expr) -> SymbolKind:
    if isinstance(value, ast.Constant):
        if isinstance(value.value, str) and len(value.value) >= _PROMPT_MIN_CHARS:
            return SymbolKind.PROMPT_TEXT
        return SymbolKind.SCALAR_CONST
    if isinstance(value, ast.Call):
        func = ast.unparse(value.func)
        if func in {"re.compile", "compile"}:
            return SymbolKind.REGEX
        return SymbolKind.COLLECTION
    if isinstance(value, (ast.Tuple, ast.List, ast.Dict, ast.Set)):
        return SymbolKind.COLLECTION
    if isinstance(value, ast.JoinedStr):
        return SymbolKind.PROMPT_TEXT
    return SymbolKind.COLLECTION


def symbol_map(source: str) -> dict[int, tuple[str, SymbolKind]]:
    """line number -> (qualified symbol, kind).

    Definitions are laid down widest-first so that narrower spans overwrite them and
    the innermost enclosing scope wins.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    spans: list[tuple[int, int, str, SymbolKind]] = []

    def walk(node: ast.AST, prefix: str, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{prefix}{child.name}"
                if isinstance(child, ast.ClassDef):
                    kind = SymbolKind.CLASS
                else:
                    kind = SymbolKind.METHOD if in_class else SymbolKind.FUNCTION
                spans.append(
                    (
                        child.lineno,
                        getattr(child, "end_lineno", child.lineno),
                        qualified,
                        kind,
                    )
                )
                walk(child, f"{qualified}.", isinstance(child, ast.ClassDef))
            else:
                walk(child, prefix, in_class)

    walk(tree, "", False)

    # Module-level bindings: named, so tuning a constant is distinguishable from
    # rewriting a system prompt.
    for node in tree.body:
        targets: list[str] = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        if not targets or node.value is None:
            continue
        spans.append(
            (
                node.lineno,
                getattr(node, "end_lineno", node.lineno),
                targets[0],
                _binding_kind(node.value),
            )
        )

    mapping: dict[int, tuple[str, SymbolKind]] = {}
    for start, end, qualified, kind in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        for line in range(start, end + 1):
            mapping[line] = (qualified, kind)
    return mapping


def symbol_source(source: str, symbol: str) -> str | None:
    """Full source text of one qualified symbol, for context and history comparison."""
    if not source or symbol in ("<module>", "<file>"):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    want = symbol.split(".")

    def find(node: ast.AST, path: list[str]) -> ast.AST | None:
        if not path:
            return node
        for child in ast.iter_child_nodes(node):
            if getattr(child, "name", None) == path[0]:
                return find(child, path[1:])
            if isinstance(child, (ast.Assign, ast.AnnAssign)) and len(path) == 1:
                targets = (
                    [child.target] if isinstance(child, ast.AnnAssign) else child.targets
                )
                if any(getattr(t, "id", None) == path[0] for t in targets):
                    return child
        return None

    found = find(tree, want)
    if found is None:
        return None
    try:
        return ast.unparse(found)
    except Exception:
        return None


def scalar_value(source: str, name: str) -> str | None:
    """Literal text of a module-level scalar binding, for before/after capture."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        targets: list[str] = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        if name in targets and node.value is not None:
            try:
                return ast.unparse(node.value)[:200]
            except Exception:
                return None
    return None
