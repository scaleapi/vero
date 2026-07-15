"""Optional Weights & Biases reporting for canonical runtime events."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from vero.runtime.artifacts import ArtifactStore
from vero.runtime.events import RuntimeEvent


class WandbEventSink:
    """Log one optimization session as one W&B run.

    W&B is imported only when this sink is constructed, so the core runtime has
    no mandatory tracking dependency.
    """

    def __init__(
        self,
        *,
        project: str,
        session_id: str,
        session_dir: Path,
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
        notes: str | None = None,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
        client: Any | None = None,
    ):
        if client is None:
            try:
                import wandb as client
            except ImportError as error:
                raise RuntimeError(
                    "W&B reporting requires `pip install scale-vero[wandb]`"
                ) from error

        wandb_dir = session_dir / "artifacts" / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(session_dir / "artifacts")
        self.state_path = "wandb/state.json"
        if self.artifacts.path(self.state_path).exists():
            state = self.artifacts.read_json(self.state_path)
            self.logged_evaluations = set(state.get("evaluation_ids", []))
            self.next_step = int(state.get("next_step", len(self.logged_evaluations)))
        else:
            self.logged_evaluations: set[str] = set()
            self.next_step = 0
        stable_id = run_id or (
            "vero-" + hashlib.sha256(session_id.encode()).hexdigest()[:16]
        )
        init_kwargs: dict[str, Any] = {
            "project": project,
            "id": stable_id,
            "resume": "allow",
            "dir": str(wandb_dir),
            "config": {**(config or {}), "vero/session_id": session_id},
        }
        for key, value in {
            "entity": entity,
            "name": name,
            "group": group,
            "tags": tags or None,
            "mode": mode,
            "notes": notes,
        }.items():
            if value is not None:
                init_kwargs[key] = value
        self.run = client.init(**init_kwargs)

    def _save_state(self) -> None:
        self.artifacts.write_json(
            self.state_path,
            {
                "evaluation_ids": sorted(self.logged_evaluations),
                "next_step": self.next_step,
            },
        )

    def __call__(self, event: RuntimeEvent) -> None:
        if event.kind == "evaluation_completed":
            payload = dict(event.payload)
            payload.pop("step")
            evaluation_id = str(payload["evaluation_id"])
            if evaluation_id in self.logged_evaluations:
                return
            self.run.log(payload, step=self.next_step)
            self.logged_evaluations.add(evaluation_id)
            self.next_step += 1
            self._save_state()
            return
        if event.kind == "session_completed":
            self.run.summary.update(event.payload)
            self.run.finish()
            return
        if event.kind == "session_failed":
            self.run.summary.update(
                {
                    "status": "failed",
                    "error_type": event.payload.get("error_type"),
                    "error_message": event.payload.get("message"),
                }
            )
            self.run.finish(exit_code=1)
