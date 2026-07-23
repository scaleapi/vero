"""Constructor-based circle packing for 26 circles in a unit square.

Adapted from ShinkaEvolve's circle-packing ``initial.py`` baseline:
https://github.com/SakanaAI/ShinkaEvolve/tree/main/examples/circle_packing

Copyright 2025 Sakana AI. Licensed under the Apache License, Version 2.0.
This VeRO adaptation replaces NumPy with the Python standard library and lets
the coding agent edit the complete program instead of a marked evolve block.
"""

from __future__ import annotations

import math


def construct_packing() -> tuple[list[list[float]], list[float]]:
    """Return centers and non-overlapping radii for 26 circles."""

    centers = [[0.0, 0.0] for _ in range(26)]
    centers[0] = [0.5, 0.5]

    for index in range(8):
        angle = 2.0 * math.pi * index / 8
        centers[index + 1] = [
            0.5 + 0.3 * math.cos(angle),
            0.5 + 0.3 * math.sin(angle),
        ]

    for index in range(16):
        angle = 2.0 * math.pi * index / 16
        centers[index + 9] = [
            0.5 + 0.7 * math.cos(angle),
            0.5 + 0.7 * math.sin(angle),
        ]

    centers = [
        [min(0.99, max(0.01, x)), min(0.99, max(0.01, y))]
        for x, y in centers
    ]
    return centers, compute_max_radii(centers)


def compute_max_radii(centers: list[list[float]]) -> list[float]:
    """Greedily shrink initially maximal radii until every pair is feasible."""

    radii = [min(x, y, 1.0 - x, 1.0 - y) for x, y in centers]
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            distance = math.dist(centers[left], centers[right])
            radius_sum = radii[left] + radii[right]
            if radius_sum > distance:
                scale = distance / radius_sum
                radii[left] *= scale
                radii[right] *= scale
    return [max(radius, 0.0) for radius in radii]


def run_packing() -> tuple[list[list[float]], list[float], float]:
    """Entry point called by the trusted evaluator."""

    centers, radii = construct_packing()
    return centers, radii, sum(radii)
