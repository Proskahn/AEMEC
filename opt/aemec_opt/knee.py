"""Knee-point selection for two-objective Pareto fronts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .engine import OptimizationError, pareto_mask


@dataclass(frozen=True)
class ChebyshevKnee:
    """A Pareto point selected by normalized distance to the ideal point."""

    index: int
    distance: float
    normalized_objectives: tuple[float, ...]


@dataclass(frozen=True)
class BendAngleKnee:
    """A Pareto point selected by its normalized bend angle."""

    index: int
    angle_radians: float
    normalized_objectives: tuple[float, float]
    left_index: int
    right_index: int

    @property
    def angle_degrees(self) -> float:
        return math.degrees(self.angle_radians)


def normalize_objectives(
    objective_vectors: Sequence[Sequence[float]],
    directions: Sequence[str],
) -> list[tuple[float, ...]]:
    """Normalize objectives to [0, 1], with zero denoting the ideal value."""
    if not directions or any(
        direction not in {"minimize", "maximize"} for direction in directions
    ):
        raise OptimizationError("Objectives must be minimized or maximized")
    if any(len(values) != len(directions) for values in objective_vectors):
        raise OptimizationError("Objective vectors and directions have different sizes")
    if any(
        not math.isfinite(value)
        for values in objective_vectors
        for value in values
    ):
        raise OptimizationError("Knee-point objectives must be finite")
    if not objective_vectors:
        return []

    minimization_values = [
        tuple(
            value if directions[index] == "minimize" else -value
            for index, value in enumerate(values)
        )
        for values in objective_vectors
    ]
    ideal = [
        min(values[index] for values in minimization_values)
        for index in range(len(directions))
    ]
    nadir = [
        max(values[index] for values in minimization_values)
        for index in range(len(directions))
    ]

    normalized: list[tuple[float, ...]] = []
    for values in minimization_values:
        point: list[float] = []
        for index, value in enumerate(values):
            span = nadir[index] - ideal[index]
            point.append(0.0 if span == 0.0 else (value - ideal[index]) / span)
        normalized.append(tuple(point))
    return normalized


def find_chebyshev_knee(
    pareto_objectives: Sequence[Sequence[float]],
    directions: Sequence[str],
) -> ChebyshevKnee | None:
    """Return the Pareto point nearest the ideal point in normalized L-infinity distance."""
    normalized = normalize_objectives(pareto_objectives, directions)
    if not normalized:
        return None
    index = min(range(len(normalized)), key=lambda item: (max(normalized[item]), item))
    return ChebyshevKnee(index, max(normalized[index]), normalized[index])


def find_bend_angle_knee(
    pareto_objectives: Sequence[Sequence[float]],
    directions: Sequence[str],
    threshold_degrees: float = 0.0,
) -> BendAngleKnee | None:
    """Return the maximum positive bend-angle point between the front's extremes.

    The left and right points are the two extreme trade-off points after converting
    every direction to minimization and normalizing each objective to [0, 1].
    A result is returned only when its angle strictly exceeds ``threshold_degrees``.
    """
    if len(directions) != 2:
        raise OptimizationError("The bend-angle method requires exactly two objectives")
    if not math.isfinite(threshold_degrees) or threshold_degrees < 0.0:
        raise OptimizationError(
            "The bend-angle threshold must be finite and non-negative"
        )
    if pareto_objectives and not all(pareto_mask(pareto_objectives, directions)):
        raise OptimizationError("The bend-angle method requires a Pareto-optimal front")

    normalized = normalize_objectives(pareto_objectives, directions)
    if not normalized:
        return None

    # Exact duplicate objective vectors describe the same geometric front point.
    # Keep the first occurrence so the selected trial remains deterministic.
    ordered_indices = sorted(
        range(len(normalized)),
        key=lambda index: (normalized[index][0], -normalized[index][1], index),
    )
    unique_indices: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index in ordered_indices:
        if normalized[index] not in seen:
            seen.add(normalized[index])
            unique_indices.append(index)
    if len(unique_indices) < 3:
        return None

    for previous_index, next_index in zip(unique_indices, unique_indices[1:]):
        previous = normalized[previous_index]
        next_point = normalized[next_index]
        if not (previous[0] < next_point[0] and previous[1] > next_point[1]):
            raise OptimizationError(
                "Bend-angle front points must form a strict two-objective trade-off"
            )

    left_index = unique_indices[0]
    right_index = unique_indices[-1]
    left = normalized[left_index]
    right = normalized[right_index]
    angles: list[tuple[float, int]] = []
    for index in unique_indices[1:-1]:
        point = normalized[index]
        theta_left = math.atan2(left[1] - point[1], point[0] - left[0])
        theta_right = math.atan2(point[1] - right[1], right[0] - point[0])
        angles.append((theta_left - theta_right, index))

    angle, index = max(angles, key=lambda item: (item[0], -item[1]))
    # Avoid classifying round-off around a mathematically straight front as a knee.
    required_angle = max(math.radians(threshold_degrees), 1.0e-12)
    if angle <= required_angle:
        return None
    point = normalized[index]
    return BendAngleKnee(
        index,
        angle,
        (point[0], point[1]),
        left_index,
        right_index,
    )
