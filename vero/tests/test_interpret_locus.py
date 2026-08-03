"""Locus resolution is the load-bearing deterministic step; test it directly.

These cases are the ones that actually went wrong on real data: git's hunk header
reports the enclosing class for a method, and module-level bindings collapse a system
prompt, a tool table and a dozen tuning constants into one bucket.
"""

from __future__ import annotations

from vero.interpret.edits.locus import changed_lines, scalar_value, symbol_map
from vero.interpret.models import SymbolKind

SOURCE = '''
MAX_TURNS = 24
SHELL_TIMEOUT_SEC = 150
INSTRUCTIONS = """
{}
"""
TOOLS = [{{"name": "run_shell"}}, {{"name": "submit"}}]
_ANSWER_RE = re.compile(r"x")


def helper(value):
    return value


class Agent:
    def run(self):
        return 1

    def _run_shell(self, cmd):
        return cmd
'''.format("guidance " * 40)


def test_method_resolves_to_method_not_class():
    mapping = symbol_map(SOURCE)
    line = next(
        i for i, text in enumerate(SOURCE.splitlines(), 1) if "return cmd" in text
    )
    symbol, kind = mapping[line]
    assert symbol == "Agent._run_shell"
    assert kind is SymbolKind.METHOD


def test_module_bindings_split_by_name_and_shape():
    mapping = symbol_map(SOURCE)
    got = {symbol: kind for symbol, kind in mapping.values()}
    assert got["MAX_TURNS"] is SymbolKind.SCALAR_CONST
    assert got["INSTRUCTIONS"] is SymbolKind.PROMPT_TEXT
    assert got["TOOLS"] is SymbolKind.COLLECTION
    assert got["_ANSWER_RE"] is SymbolKind.REGEX


def test_plain_function_is_not_a_method():
    mapping = symbol_map(SOURCE)
    line = next(
        i for i, text in enumerate(SOURCE.splitlines(), 1) if "return value" in text
    )
    assert mapping[line] == ("helper", SymbolKind.FUNCTION)


def test_changed_lines_parses_zero_context_hunks():
    diff = "@@ -1 +1 @@\n-a\n+b\n@@ -10,0 +11,3 @@\n+x\n+y\n+z\n"
    assert changed_lines(diff) == {1, 11, 12, 13}


def test_scalar_value_reads_the_literal():
    assert scalar_value(SOURCE, "MAX_TURNS") == "24"
    assert scalar_value(SOURCE, "SHELL_TIMEOUT_SEC") == "150"
    assert scalar_value(SOURCE, "nonexistent") is None


def test_unparsable_source_yields_no_map_rather_than_raising():
    assert symbol_map("def broken(:\n") == {}
