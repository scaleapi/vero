"""Trusted ALE-Bench scorer for a CommandBackend.

vero drives the OUTER loop (a coding agent edits /work/agent/solution.cpp); each
candidate is scored here by ALE-Bench's deterministic judge. Disclosure maps onto
ALE-Bench's public/private split:
  development / validation -> session.public_eval  (public seeds, disclosed feedback)
  test                     -> session.private_eval (private seeds, held-out)

The judge runs the submission in Docker, so this process needs Docker daemon
access (DOCKER_HOST) and the language image (ale-bench:<lang>-<version>) present.
Config (problem id, language, workers) is read from harness/config.json.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


def _load_config(harness_dir: Path) -> dict:
    cfg = json.loads((harness_dir / "config.json").read_text(encoding="utf-8"))
    cfg.setdefault("code_language", "cpp20")
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("lite_version", True)
    return cfg


def _partition(command_input: dict) -> str:
    request = command_input.get("request") or {}
    evaluation_set = request.get("evaluation_set") or {}
    return evaluation_set.get("partition") or "development"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()

    harness_dir = Path(__file__).resolve().parent
    config = _load_config(harness_dir)
    command_input = json.loads(args.request.read_text(encoding="utf-8"))
    partition = _partition(command_input)
    held_out = partition == "test"

    artifact_dir = args.artifacts / "ale-bench"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        code = (args.workspace / "solution.cpp").read_text(encoding="utf-8")

        import ale_bench

        session = ale_bench.start(
            problem_id=config["problem_id"],
            lite_version=config["lite_version"],
            num_workers=int(config["num_workers"]),
            run_visualization_server=False,
        )
        try:
            if held_out:
                result, rank, performance = session.private_eval(
                    code, code_language=config["code_language"]
                )
                extra = {"rank": rank, "performance": performance}
            else:
                result = session.public_eval(
                    code, code_language=config["code_language"]
                )
                extra = {}
        finally:
            close = getattr(session, "close", None)
            if callable(close):
                close()

        judge = str(getattr(result, "overall_judge_result", ""))
        accepted = 1.0 if judge.upper().endswith("ACCEPTED") else 0.0
        score = float(getattr(result, "overall_absolute_score", 0.0) or 0.0)
        runtime_ms = (time.perf_counter() - started) * 1000.0

        (artifact_dir / "result.json").write_text(
            json.dumps(
                {"partition": partition, "judge": judge, "score": score, **extra},
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        report = {
            "schema_version": 1,
            "status": "success",
            "metrics": {
                "score": score,
                "accepted": accepted,
                "runtime_ms": runtime_ms,
            },
            "artifacts": [
                {
                    "path": "ale-bench/result.json",
                    "media_type": "application/json",
                    "description": f"ALE-Bench {partition} judge result",
                }
            ],
        }
        if accepted == 0.0:
            report["diagnostics"] = [
                {
                    "code": "not_accepted",
                    "message": f"judge result: {judge}",
                    "severity": "warning",
                    "phase": "judge",
                }
            ]
    except BaseException as error:  # noqa: BLE001 - report failure as a scored 0
        runtime_ms = (time.perf_counter() - started) * 1000.0
        (artifact_dir / "failure.log").write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8"
        )
        report = {
            "schema_version": 1,
            "status": "failed",
            "metrics": {"score": 0.0, "accepted": 0.0, "runtime_ms": runtime_ms},
            "diagnostics": [
                {
                    "code": "scorer_error",
                    "message": f"{type(error).__name__}: {error}",
                    "severity": "error",
                    "phase": "scoring",
                }
            ],
        }

    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
