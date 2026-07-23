"""Trusted C build, correctness, and performance harness."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("schema_version") != 1:
        raise ValueError("unsupported command input schema")

    artifact_dir = args.artifacts / "c-matmul"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    binary = artifact_dir / "benchmark"
    benchmark_source = Path(__file__).with_name("benchmark.c")
    compile_result = subprocess.run(
        [
            "cc",
            "-O2",
            "-std=c11",
            "-I",
            str(args.workspace),
            str(args.workspace / "matmul.c"),
            str(benchmark_source),
            "-lm",
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    (artifact_dir / "compiler.log").write_text(
        compile_result.stdout + compile_result.stderr,
        encoding="utf-8",
    )
    if compile_result.returncode != 0:
        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "diagnostics": [
                        {
                            "code": "compile_failed",
                            "message": "C candidate did not compile",
                            "severity": "error",
                            "phase": "compile",
                        }
                    ],
                    "artifacts": [
                        {
                            "path": "c-matmul/compiler.log",
                            "media_type": "text/plain",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return

    measurements: list[float] = []
    correct = True
    for _ in range(3):
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        if run.returncode != 0:
            correct = False
            break
        correct_value, latency_value = run.stdout.strip().split()
        correct = correct and correct_value == "1"
        measurements.append(float(latency_value))

    latency_ms = min(measurements) if measurements else 1.0e12
    args.report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "success",
                "metrics": {
                    "latency_ms": latency_ms,
                    "correct": 1.0 if correct else 0.0,
                },
                "artifacts": [
                    {
                        "path": "c-matmul/compiler.log",
                        "media_type": "text/plain",
                    },
                    {
                        "path": "c-matmul/benchmark",
                        "media_type": "application/octet-stream",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
