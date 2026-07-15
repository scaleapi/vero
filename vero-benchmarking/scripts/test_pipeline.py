"""Compose a canonical benchmark session without spending an evaluation."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vero_benchmarking.runner import build_benchmark_session  # noqa: E402


async def main():
    session = await build_benchmark_session(
        model="anthropic/claude-sonnet-4-5-20250929",
        task_name="gpqa-nosplit",
        agent_name="claude",
        max_turns=1,
    )
    print(f"Session ID: {session.id}")
    print(f"Project path: {session.optimizer.workspace.project_path}")
    print(f"Evaluation set: {session.optimizer.evaluation_set}")
    print("Pipeline composed successfully; no evaluation was run.")


if __name__ == "__main__":
    asyncio.run(main())
