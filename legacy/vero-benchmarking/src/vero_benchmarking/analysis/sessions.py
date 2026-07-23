"""Batch session analysis using TraceAnalyzer.

Orchestrates LLM-based trace analysis across multiple sessions with
concurrency control and caching.

Typical usage:
    from vero.traces.analysis import TraceAnalyzer
    from vero_benchmarking.analysis.sessions import batch_analyze_sessions

    analyzer = TraceAnalyzer(model="gpt-4.1")
    status = await batch_analyze_sessions(session_ids, project_path, analyzer)
"""

from __future__ import annotations

import asyncio

from tqdm.asyncio import tqdm

from vero.traces.analysis import TraceAnalyzer


async def batch_analyze_sessions(
    session_ids: list[str],
    project_path: str,
    analyzer: TraceAnalyzer | None = None,
    max_concurrency: int = 5,
    use_cache: bool = True,
    save_to_cache: bool = True,
) -> dict[str, list[str]]:
    """Analyze multiple sessions in parallel with concurrency control.

    Args:
        session_ids: List of session UUIDs to analyze.
        project_path: Path to the project repo (for git diffs).
        analyzer: TraceAnalyzer instance. Creates default if None.
        max_concurrency: Maximum concurrent LLM calls.
        use_cache: Load from cached analysis_df.csv if available.
        save_to_cache: Save results to analysis_df.csv in session dir.

    Returns:
        Dict with "success" and "error" lists of session IDs.
    """
    if analyzer is None:
        analyzer = TraceAnalyzer()

    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_one(session_id: str):
        async with semaphore:
            try:
                await analyzer.analyze_session(
                    session_id=session_id,
                    project_path=project_path,
                    show_progress=False,
                    return_payload=False,
                    save_to_cache=save_to_cache,
                    use_cache=use_cache,
                )
                return session_id
            except Exception as e:
                return e

    status = {"success": [], "error": []}
    coros = [analyze_one(sid) for sid in session_ids]
    results = await tqdm.gather(*coros, desc="Analyzing sessions")

    for session_id, result in zip(session_ids, results):
        if isinstance(result, Exception):
            status["error"].append(session_id)
            print(f"Error analyzing session {session_id}: {result}")
        else:
            status["success"].append(session_id)

    return status
