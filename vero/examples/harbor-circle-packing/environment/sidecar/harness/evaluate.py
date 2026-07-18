"""Trusted validator and scorer for the 26-circle packing benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


@dataclass(frozen=True)
class Validation:
    valid: bool
    message: str
    computed_sum: float
    reported_sum: float
    sum_error: float
    minimum_boundary_clearance: float
    minimum_pair_clearance: float


def _load_candidate(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("candidate_packing", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate program: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_runtime(seed: int | None) -> None:
    """Seed common in-process RNGs before calling the candidate entry point."""

    if seed is None:
        return
    os.environ["VERO_EVALUATION_SEED"] = str(seed)
    random.seed(seed)
    numpy = sys.modules.get("numpy")
    numpy_random = getattr(numpy, "random", None)
    numpy_seed = getattr(numpy_random, "seed", None)
    if callable(numpy_seed):
        numpy_seed(seed)


def _sequence(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not text")
    try:
        return list(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error


def _normalize_output(
    output: Any,
) -> tuple[list[list[float]], list[float], float]:
    values = _sequence(output, name="run_packing output")
    if len(values) != 3:
        raise ValueError("run_packing must return (centers, radii, reported_sum)")

    raw_centers = _sequence(values[0], name="centers")
    raw_radii = _sequence(values[1], name="radii")
    centers: list[list[float]] = []
    for index, center in enumerate(raw_centers):
        coordinates = _sequence(center, name=f"centers[{index}]")
        if len(coordinates) != 2:
            raise ValueError(f"centers[{index}] must have exactly two coordinates")
        centers.append([float(coordinates[0]), float(coordinates[1])])
    radii = [float(radius) for radius in raw_radii]
    return centers, radii, float(values[2])


def validate_packing(
    centers: Sequence[Sequence[float]],
    radii: Sequence[float],
    reported_sum: float,
) -> Validation:
    if len(centers) != 26:
        raise ValueError(f"expected 26 centers, received {len(centers)}")
    if len(radii) != 26:
        raise ValueError(f"expected 26 radii, received {len(radii)}")

    flattened = [coordinate for center in centers for coordinate in center]
    if not all(math.isfinite(value) for value in [*flattened, *radii, reported_sum]):
        raise ValueError("centers, radii, and reported_sum must all be finite")
    if any(radius < 0.0 for radius in radii):
        raise ValueError("radii must be non-negative")

    computed_sum = math.fsum(radii)
    sum_error = abs(computed_sum - reported_sum)
    sum_matches = math.isclose(computed_sum, reported_sum, rel_tol=1e-9, abs_tol=1e-12)
    boundary_clearances = [
        min(x - radius, y - radius, 1.0 - x - radius, 1.0 - y - radius)
        for (x, y), radius in zip(centers, radii, strict=True)
    ]
    pair_clearances = [
        math.dist(centers[left], centers[right]) - radii[left] - radii[right]
        for left in range(26)
        for right in range(left + 1, 26)
    ]
    minimum_boundary_clearance = min(boundary_clearances)
    minimum_pair_clearance = min(pair_clearances)

    failures: list[str] = []
    if not sum_matches:
        failures.append(
            f"reported sum differs from the computed sum by {sum_error:.6g}"
        )
    if minimum_boundary_clearance < 0.0:
        failures.append(
            "at least one circle crosses the square boundary "
            f"by {-minimum_boundary_clearance:.6g}"
        )
    if minimum_pair_clearance < 0.0:
        failures.append(
            f"at least two circles overlap by {-minimum_pair_clearance:.6g}"
        )

    return Validation(
        valid=not failures,
        message="; ".join(failures) if failures else "packing is geometrically valid",
        computed_sum=computed_sum,
        reported_sum=reported_sum,
        sum_error=sum_error,
        minimum_boundary_clearance=minimum_boundary_clearance,
        minimum_pair_clearance=minimum_pair_clearance,
    )


def _write_layout_json(
    path: Path,
    centers: Sequence[Sequence[float]],
    radii: Sequence[float],
    validation: Validation,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "circles": [
                    {"index": index, "x": x, "y": y, "radius": radius}
                    for index, ((x, y), radius) in enumerate(
                        zip(centers, radii, strict=True)
                    )
                ],
                "validation": asdict(validation),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_layout_svg(
    path: Path,
    centers: Sequence[Sequence[float]],
    radii: Sequence[float],
    validation: Validation,
) -> None:
    size = 800
    circles = "\n".join(
        (
            f'  <circle cx="{x * size:.6f}" cy="{(1.0 - y) * size:.6f}" '
            f'r="{radius * size:.6f}" fill="hsl({(index * 137.5) % 360:.1f} 65% 70%)" '
            'fill-opacity="0.7" stroke="#17202a" stroke-width="1.5"/>'
        )
        for index, ((x, y), radius) in enumerate(zip(centers, radii, strict=True))
    )
    label = (
        f"sum={validation.computed_sum:.12f}; "
        f"valid={'yes' if validation.valid else 'no'}"
    )
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="0 0 {size} {size + 48}" role="img">',
                '  <rect width="800" height="800" fill="#f8f9f9" '
                'stroke="#17202a" stroke-width="3"/>',
                circles,
                f'  <text x="12" y="832" font-family="monospace" font-size="22">{label}</text>',
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _artifact(path: str, media_type: str, description: str) -> dict[str, str]:
    return {"path": path, "media_type": media_type, "description": description}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()

    command_input = json.loads(args.request.read_text(encoding="utf-8"))
    if command_input.get("schema_version") != 1:
        raise ValueError("unsupported command input schema")
    request = command_input.get("request")
    if not isinstance(request, dict):
        raise ValueError("command input must contain an evaluation request")
    seed = request.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("evaluation seed must be an integer or null")

    artifact_dir = args.artifacts / "circle-packing"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        _seed_runtime(seed)
        candidate = _load_candidate(args.workspace / "packing.py")
        run_packing = getattr(candidate, "run_packing", None)
        if not callable(run_packing):
            raise AttributeError("packing.py must define a callable run_packing()")
        # Reset after import-time work and seed NumPy if the candidate imported it.
        _seed_runtime(seed)
        centers, radii, reported_sum = _normalize_output(run_packing())
        runtime_ms = (time.perf_counter() - started) * 1000.0
        validation = validate_packing(centers, radii, reported_sum)

        _write_layout_json(
            artifact_dir / "layout.json", centers, radii, validation
        )
        _write_layout_svg(artifact_dir / "layout.svg", centers, radii, validation)
        (artifact_dir / "validation.json").write_text(
            json.dumps(asdict(validation), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts = [
            _artifact(
                "circle-packing/layout.json",
                "application/json",
                "Circle centers, radii, and validation measurements",
            ),
            _artifact(
                "circle-packing/layout.svg",
                "image/svg+xml",
                "Rendered circle packing",
            ),
            _artifact(
                "circle-packing/validation.json",
                "application/json",
                "Geometric feasibility diagnostics",
            ),
        ]
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "success",
            "metrics": {
                "sum_radii": validation.computed_sum,
                "valid": 1.0 if validation.valid else 0.0,
                "minimum_boundary_clearance": validation.minimum_boundary_clearance,
                "minimum_pair_clearance": validation.minimum_pair_clearance,
                "runtime_ms": runtime_ms,
            },
            "artifacts": artifacts,
        }
        if not validation.valid:
            report["diagnostics"] = [
                {
                    "code": "invalid_packing",
                    "message": validation.message,
                    "severity": "warning",
                    "phase": "validation",
                }
            ]
    except BaseException as error:
        runtime_ms = (time.perf_counter() - started) * 1000.0
        (artifact_dir / "failure.log").write_text(
            "".join(traceback.format_exception(error)),
            encoding="utf-8",
        )
        report = {
            "schema_version": 1,
            "status": "failed",
            "metrics": {"runtime_ms": runtime_ms},
            "diagnostics": [
                {
                    "code": "candidate_execution_failed",
                    "message": f"{type(error).__name__}: {error}",
                    "severity": "error",
                    "phase": "candidate",
                }
            ],
            "artifacts": [
                _artifact(
                    "circle-packing/failure.log",
                    "text/plain",
                    "Candidate exception and traceback",
                )
            ],
        }

    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
