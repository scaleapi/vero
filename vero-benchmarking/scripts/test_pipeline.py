"""Minimal pipeline test — isolate project, init session, 1 agent turn, shut down."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vero_benchmarking.runner import make_claude_code_policy  # noqa: E402


async def main():
    # Use current branch so git archive finds vero-agents files
    import subprocess
    git_ref = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent.parent,
        text=True,
    ).strip()
    print(f"Using git_ref={git_ref}")

    policy = make_claude_code_policy(
        model="anthropic/claude-sonnet-4-5-20250929",
        task_name="gpqa-nosplit",
        max_turns=1,
        ref=git_ref,
    )

    async with policy:
        print(f"Session ID: {policy.session_id}")
        print(f"Project path: {policy.session.project_path}")
        print(f"Base version: {policy.session.base_version}")
        print("Pipeline initialized successfully!")


if __name__ == "__main__":
    asyncio.run(main())
