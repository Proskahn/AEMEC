"""Geometry and acquisition functions for two minimized objectives.

The runner passes objectives in the same fixed normalized coordinate system
for every algorithm and seed. A smaller objective value is preferable, whereas
a larger hypervolume is preferable. The reference point is an upper-right
corner that defines the objective-space region included in the comparison.

``hypervolume_2d`` measures the quality of already evaluated designs.
``expected_hypervolume_improvement`` scores unevaluated candidate designs using
the two Gaussian-process predictive distributions. Neither function evaluates
the AEMEC simulator or chooses a normalization/reference from the current data.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def pareto_indices(values):
    """Return original row indices of the distinct nondominated objective pairs.

    ``values`` has shape (n, 2). A point is dominated if another point is no
    worse in either objective and strictly better in at least one. Identical
    pairs represent the same objective-space point, so only one is retained.
    An empty input produces an empty integer index array.

    Sorting by objective 1 allows an O(n log n) algorithm: each retained point
    must improve the best objective-2 value encountered so far. The resulting
    front has increasing first coordinates and decreasing second coordinates.
    """
    values = np.asarray(values, dtype=float).reshape(-1, 2)
    if not np.isfinite(values).all():
        raise ValueError('Pareto values must be finite')
    indices, best_y = [], np.inf
    # lexsort uses its LAST key as the primary key: first sort on objective 1,
    # then on objective 2. At equal objective 1, the best objective 2 comes first.
    for i in np.lexsort((values[:, 1], values[:, 0])):
        # Strict improvement removes both dominated points and duplicate pairs.
        if values[i, 1] < best_y:
            indices.append(int(i))
            best_y = values[i, 1]
    return np.asarray(indices, dtype=int)


def bounded_front(values, reference):
    """Keep the Pareto points that can contribute positive reference-bounded area.

    A point with either coordinate at or above the reference has a zero-width
    or zero-height dominated rectangle inside the measured region. It therefore
    contributes nothing to this hypervolume, even if it is globally nondominated.
    Values below zero remain valid; fixed normalization does not clip objectives.
    """
    values = np.asarray(values, dtype=float).reshape(-1, 2)
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (2,) or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError('Expected finite two-objective values and reference point')
    inside = values[np.all(values < reference, axis=1)]
    return inside[pareto_indices(inside)]


def hypervolume_2d(values, reference):
    """Measure the union of rectangles from nondominated points to ``reference``.

    For minimization, each point (a, b) covers [a, reference[0]] x
    [b, reference[1]]. Count only newly covered horizontal strips as the sorted
    front descends in objective 2; this avoids double-counting overlapping areas.
    For example, [(1, 4), (3, 2)] with reference (5, 5) has area 4*1 + 2*2 = 8.
    """
    front = bounded_front(values, reference)
    height, area = float(reference[1]), 0.
    for a, b in front:
        # Earlier points already cover everything above ``height``. This point
        # adds only the band between b and height, extending right to reference[0].
        area += (reference[0]-a)*(height-b)
        height = b
    return float(area)


def positive_part_expectation(bound, mean, std):
    """Compute E[max(bound - Y, 0)] for Y ~ Normal(mean, std**2).

    With delta = bound - mean and z = delta/std, the Gaussian formula is
    delta * Phi(z) + std * phi(z), where Phi is the normal CDF and phi its PDF.
    This is also the integral of P(Y <= t) from -infinity to ``bound``.
    Inputs broadcast over the acquisition candidate pool.

    At std=0 the prediction is deterministic, so the answer is max(delta, 0).
    ``safe_std`` prevents division by zero before np.where chooses this branch;
    it does not add uncertainty to a deterministic prediction.
    """
    delta = bound-mean
    safe_std = np.where(std > 0, std, 1.)
    z = delta/safe_std
    stochastic = delta*ndtr(z)+safe_std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    # Small cancellation errors can make a theoretically nonnegative expectation
    # slightly negative in the far tail. Clamp numerical negatives to zero.
    return np.maximum(np.where(std > 0, stochastic, np.maximum(delta, 0)), 0)


def expected_hypervolume_improvement(mean, std, observations, reference):
    """Exact 2D q=1 EHVI for independent normal predictive marginals.

    Parameters
    ----------
    mean, std : arrays with shape (n_candidates, 2)
        Predictive means and latent-function standard deviations, expressed
        in the SAME normalized objective coordinates as observations/reference.
    observations : array with shape (n_successful, 2)
        Objective pairs already evaluated successfully by this optimizer.
    reference : array with shape (2,)
        Fixed upper-right corner of the measured objective-space region.

    Returns
    -------
    Array with shape (n_candidates,)
        Expected additional dominated area from evaluating one candidate.

    The improvement is the area dominated by a candidate but not by the existing
    front. Exchange expectation and area integration: at location (u1, u2), the
    probability that a candidate dominates it is P(Y1 <= u1)*P(Y2 <= u2).
    This factorization requires independent objective predictions. Integrating
    over vertical strips makes both integrals one-dimensional and analytic.

    q=1 means one new design, not a simultaneous batch. Existing observations
    form a fixed plug-in front; uncertainty in that front is not integrated as
    it would be in noisy expected hypervolume improvement (NEHVI).
    """
    mean, std, reference = map(lambda x: np.asarray(x, dtype=float), (mean, std, reference))
    if (mean.ndim != 2 or mean.shape[1] != 2 or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0)):
        raise ValueError('EHVI needs finite (n, 2) means and nonnegative standard deviations')
    front = bounded_front(observations, reference)
    # Before the leftmost Pareto point, the entire height up to reference[1]
    # is undominated. The first strip starts at -infinity, whose integral is 0.
    previous_integral = np.zeros(len(mean))
    cap = reference[1]
    result = np.zeros(len(mean))
    # At each Pareto x-coordinate, the vertical ceiling drops to that point's
    # second coordinate. Append the reference x-coordinate to close the final
    # strip. With no front, this loop integrates the whole reference rectangle.
    for right, next_cap in [*front.tolist(), [reference[0], None]]:
        integral = positive_part_expectation(right, mean[:, 0], std[:, 0])
        # Integral over x in [previous_right, right] times integral over y in
        # (-infinity, cap]. These disjoint strips cover all possible improvement.
        result += np.maximum(integral-previous_integral, 0)*positive_part_expectation(
            cap, mean[:, 1], std[:, 1])
        previous_integral = integral
        # The final None sentinel is assigned only after the last strip is used.
        cap = next_cap
    return np.maximum(result, 0)
