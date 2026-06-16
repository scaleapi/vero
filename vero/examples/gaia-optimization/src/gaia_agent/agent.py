"""A GAIA optimization target: Harbor's terminus-2 with an editable prompt.

``GaiaAgent`` subclasses Harbor's ``Terminus2`` and points its prompt template at
this package's ``prompts/`` directory instead of the copy baked into the harbor
package. That makes the prompt the *optimization surface*: an optimizer (e.g.
Claude Code, driving ``vero harbor eval``) edits ``prompts/terminus-json-plain.txt``
to improve the agent's GAIA score, while the terminal loop, tmux session, and
response parsing are reused unchanged from ``Terminus2``.

The agent runs in the Harbor orchestrator process (where the LLM creds live) and
drives the task sandbox via ``environment.exec``; see the example README.
"""

from __future__ import annotations

from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Terminus2

_PROMPTS = Path(__file__).parent / "prompts"


class GaiaAgent(Terminus2):
    """Terminus-2 with its prompt sourced from this package's editable ``prompts/``."""

    @staticmethod
    def name() -> str:
        return "gaia-agent"

    def version(self) -> str:
        return "0.1.0"

    def _get_prompt_template_path(self) -> Path:
        if self._parser_name == "json":
            return _PROMPTS / "terminus-json-plain.txt"
        if self._parser_name == "xml":
            return _PROMPTS / "terminus-xml-plain.txt"
        return super()._get_prompt_template_path()
