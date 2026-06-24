"""HarborConfig — the Mode-B configuration.

User-facing config that turns "evaluate my agent on a set of Harbor tasks" into a
`harbor run` invocation. A typed projection of the user-controllable `harbor run`
flags; the per-eval-derived flags (task selection, jobs dir, source/agent resolution)
are filled in by the runner, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HarborConfig:
    task_source: str  # registry ref "org/name[@ver]" OR a local path to a task dir/dataset
    agent_import_path: str  # module path to the candidate agent, e.g. "pkg.mod:Class"
    model: str | None = None
    environment: str = "modal"  # cloud provider (docker allowed for local testing)
    n_attempts: int = 1
    max_retries: int = 2
    reward_key: str | None = None  # primary reward; default pass -> reward -> mean
    extra_args: list[str] = field(default_factory=list)  # passthrough harbor run flags

    @property
    def is_registry(self) -> bool:
        """Local if the source resolves to an existing path; otherwise a registry ref."""
        return not Path(self.task_source).expanduser().exists()

    def source_args(self) -> list[str]:
        """`harbor run` source selector: `-d <ref>` (registry) or `-p <path>` (local)."""
        if self.is_registry:
            return ["-d", self.task_source]
        return ["-p", str(Path(self.task_source).expanduser())]
