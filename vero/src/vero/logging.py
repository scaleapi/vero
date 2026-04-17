from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax
from rich.theme import Theme

if TYPE_CHECKING:
    import wandb

    from vero.core.db.database import Experiment

logger = logging.getLogger(__name__)

DEFAULT_LEVELS = {
    "httpx": logging.WARNING,
    "agents": logging.WARNING,
    "litellm": logging.WARNING,
    "LiteLLM": logging.WARNING,
    "harbor": logging.WARNING,
}


def log_experiments_to_wandb(
    wandb_run: wandb.Run, experiments: list[Experiment]
) -> None:
    """Logs the results of the experiments to wandb."""

    for experiment in experiments:
        wandb_run.log(experiment.summary())
    logger.info(f"Logged {len(experiments)} experiments to wandb.")


def setup_logging(verbose: bool = False, levels: dict[str, int] | None = None):
    """Setup logging configuration with Rich formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    rich_handler = RichHandler(rich_tracebacks=True, markup=True, show_path=verbose)
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    rich_handler.setLevel(level)
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(level)

    if levels is None:
        levels = DEFAULT_LEVELS

    for name, level in levels.items():
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(rich_handler)
        logger.setLevel(level)
        # Don't propagate to root to avoid double logging
        logger.propagate = False


def setup_console() -> Console:
    """Setup a console with a monokai theme."""
    monokai_theme = Theme(
        {
            "info": "#66D9EF",  # blue/cyan
            "warning": "#FD971F",  # orange
            "error": "#F92672",  # pink/red
            "success": "#A6E22E",  # green
            "debug": "#AE81FF",  # purple
            "highlight": "#F8F8F2",  # off-white
        }
    )
    return Console(theme=monokai_theme)


def setup_sgp_tracing(
    account_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Setup SGP tracing."""
    import scale_gp_beta.lib.tracing as tracing
    from scale_gp_beta import SGPClient

    tracing.init(
        SGPClient(api_key=api_key, account_id=account_id, base_url=base_url),
        disabled=False,
    )


def setup_sgp_agents_sdk_tracing(
    account_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Setup SGP tracing for the OpenAI Agents SDK."""
    import agents
    from agents import set_trace_processors
    from scale_gp_beta.lib.tracing.integrations import OpenAITracingSGPProcessor

    setup_sgp_tracing(account_id=account_id, api_key=api_key, base_url=base_url)

    agents.run.RunConfig.tracing_disabled = True
    sgp_processor = OpenAITracingSGPProcessor()
    set_trace_processors([sgp_processor])


class SessionLogger:
    """Single configurable object for all session-scoped logging.

    Handles three concerns:
    1. Event logging — JSONL trace of agent events (replaces TraceWriter)
    2. General logging — Python logging output captured to session dir
    3. Console rendering — Rich panels for agent turns (replaces AgentTurnRenderer)

    Callable — use directly as a ``policy.on_event`` callback.
    """

    def __init__(
        self,
        session_dir: Path,
        enable_event_log: bool = True,
        enable_general_log: bool = True,
        enable_console: bool = True,
        console_verbose: bool = True,
        console_title: str = "Agent",
        event_log_filename: str = "agent_trace.jsonl",
        general_log_filename: str = "session.log",
    ) -> None:
        self._session_dir = Path(session_dir)
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._turn = 0

        # Event log (per-turn JSON files)
        self._event_log_dir = None
        if enable_event_log:
            self._event_log_dir = self._session_dir / "agent_trace"
            self._event_log_dir.mkdir(exist_ok=True)

        # General log (Python logging handler)
        self._log_handler = None
        self._original_root_level: int | None = None
        if enable_general_log:
            path = self._session_dir / general_log_filename
            self._log_handler = logging.FileHandler(path)
            self._log_handler.setLevel(logging.DEBUG)
            self._log_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            root = logging.getLogger()
            root.addHandler(self._log_handler)
            self._original_root_level = root.level
            if root.level > logging.DEBUG:
                root.setLevel(logging.DEBUG)

        # Console rendering
        self._console = None
        self._console_title = console_title
        self._console_verbose = console_verbose
        if enable_console:
            self._console = Console(width=120)

    def __call__(self, event: dict[str, Any]) -> None:
        """Handle a serialized agent event: write to JSONL and render to console."""
        self._write_event(event)
        self._render_event(event)
        self._turn += 1

    def _write_event(self, event: dict[str, Any]) -> None:
        """Write event as an individual JSON file."""
        if self._event_log_dir is None:
            return
        data = {
            "turn": self._turn,
            "ts": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        path = self._event_log_dir / f"turn_{self._turn:04d}.json"
        path.write_text(json.dumps(data, indent=2, default=str))

    def _render_event(self, event: dict[str, Any]) -> None:
        """Render event to the console."""
        if self._console is None:
            return
        if self._console_verbose:
            self._render_verbose(event)
        else:
            self._render_compact(event)

    def _render_verbose(self, event: dict[str, Any]) -> None:
        """Full JSON panel rendering."""
        type_name = event.get("kind", "event")
        try:
            content_str = json.dumps(event, indent=2, default=str)
            syntax = Syntax(
                content_str, "json", theme="monokai", line_numbers=False, word_wrap=True
            )
        except Exception:
            syntax = str(event)

        panel = Panel(
            syntax,
            title=f"[bold green]{self._console_title} :: Turn {self._turn + 1}: {type_name}[/bold green]",
            border_style="#FD971F",
            expand=True,
        )
        if self._console:
            self._console.print(panel)

    def _render_compact(self, event: dict[str, Any]) -> None:
        """One-line-per-turn rendering based on normalized AgentEvent kinds."""
        kind = event.get("kind", "")

        if kind == "message":
            text = event.get("text", "")
            if text and self._console:
                self._console.print(f"[bold]💬 {text[:200]}[/bold]")

        elif kind == "thinking":
            text = event.get("text", "")
            if self._console:
                self._console.print(f"[dim]💭 {text[:200]}[/dim]")

        elif kind == "tool_call":
            name = event.get("name", "?")
            args = event.get("args", "")
            if len(args) > 100:
                args = args[:100] + "..."
            if self._console:
                self._console.print(f"[cyan]🔧 {name}({args})[/cyan]")

        elif kind == "tool_result":
            output = event.get("output", "")
            is_error = event.get("is_error", False)
            preview = output.split("\n")[0][:150] if output else "(empty)"
            lines = output.count("\n") + 1 if output else 0

            if self._console:
                if is_error:
                    self._console.print(f"[red]  ⎿ ❌ {preview}[/red]")
                elif lines > 1:
                    self._console.print(f"[dim]  ⎿ {preview}... ({lines} lines)[/dim]")
                else:
                    self._console.print(f"[dim]  ⎿ {preview}[/dim]")

        elif kind == "system":
            text = event.get("text", "")
            if self._console:
                self._console.print(f"[dim]⚙ {text[:150]}[/dim]")

        elif kind == "result":
            text = event.get("text", "")
            if self._console:
                self._console.print(f"[green]✓ {text[:200]}[/green]")

    def close(self) -> None:
        """Close file handles and remove logging handler."""
        if self._log_handler is not None:
            root = logging.getLogger()
            root.removeHandler(self._log_handler)
            if self._original_root_level is not None:
                root.setLevel(self._original_root_level)
            self._log_handler.close()
            self._log_handler = None

    def __enter__(self) -> SessionLogger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __getstate__(self) -> dict:
        """Support pickling by excluding open handles."""
        state = self.__dict__.copy()
        state.pop("_log_handler", None)
        state.pop("_console", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Reopen on unpickle."""
        self.__dict__.update(state)
        self._log_handler = None
        self._console = None
