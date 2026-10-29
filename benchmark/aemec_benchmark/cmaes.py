"""Bounded, sequential multiobjective CMA-ES for the benchmark's two objectives.

Ordinary CMA-ES ranks a scalar objective. Here DEAP's StrategyMultiObjective
retains objective pairs and selects parents by Pareto rank, breaking an
overfull front by hypervolume contribution. Each parent owns a Gaussian search
distribution with its own step size and covariance. No weighted sum is needed.

The adapter uses one offspring per update (lambda=1), so every backend call can
inform the next proposal and any attempt budget can be honored exactly. Inputs
and objectives use the same normalized coordinates as the other optimizers.
"""
from __future__ import annotations

import numpy as np

from .algorithms import sobol_points
from .metrics import hypervolume_2d


def require_deap():
    """Load the optional CMA dependency before any expensive evaluations start.

    Keeping this import lazy lets existing GP/TPE/NSGA-II-only commands run in
    environments that have not yet installed the extra benchmark dependency.
    """
    try:
        from deap import base, cma
    except ImportError as exc:
        raise ValueError('CMA-ES requires DEAP. Install with: '
                         'python -m pip install -r benchmark/requirements.txt') from exc
    return base, cma


class MultiObjectiveCMAES:
    """Adapt DEAP's MO-CMA-ES to one pending ask/evaluate/tell at a time.

    Successful shared initial designs become the parent population. If some
    fail, the population is smaller; no extra evaluations replace them. If all
    fail, a logged Sobol fallback searches until the first success, which then
    initializes a one-parent strategy. Failed offspring consume budget but do
    not enter selection or trigger an adaptation based on invented fitness.
    """

    def __init__(self, seed, reference, sigma=.2):
        """Set the RNG and initial standard deviation in the unit input box.

        With C=I initially, sigma=.2 means a Gaussian standard deviation equal
        to 20% of each input range before boundary repair. Later the covariance
        can learn correlated directions and the step sizes adapt separately.
        """
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('CMA sigma must be finite and positive')
        self.reference = np.asarray(reference, dtype=float).copy()
        if self.reference.shape != (2,) or not np.isfinite(self.reference).all():
            raise ValueError('CMA reference must contain two finite normalized objectives')
        base, cma = require_deap()
        self.strategy_type = cma.StrategyMultiObjective

        # Local classes avoid DEAP creator's global type registry, so separate
        # optimizers/seeds can coexist without overwriting each other's types.
        class Fitness(base.Fitness):
            weights = (-1., -1.)  # DEAP's negative weights mean minimization.

        class Individual(list):
            def __init__(self, point):
                super().__init__(map(float, point))
                self.fitness = Fitness()

        self.individual_type = Individual
        self.seed, self.sigma = seed, float(sigma)
        self.rng = np.random.RandomState(seed)
        self.parents, self.strategy = [], None
        self.pending, self.pending_kind = None, None
        self.attempts, self.updates = 0, 0

    def _least_contributor(self, front, ref=None):
        """Return the index whose removal preserves the largest hypervolume.

        DEAP supplies a population-dependent ``ref``. Deliberately use this
        benchmark's fixed reference instead, in the same objective coordinates
        as every recorded metric. This is a documented change to its default
        truncation rule. Pareto ranking still precedes this tie-break, including
        for observations outside the reference. Equal contributions are resolved
        by the first index, making the rule deterministic.
        """
        values = np.asarray([individual.fitness.values for individual in front])
        remaining = [hypervolume_2d(np.delete(values, i, axis=0), self.reference)
                     for i in range(len(values))]
        return int(np.argmax(remaining))

    def ask(self, fixed=None):
        """Suggest one feasible point and record how it was generated.

        Boundary handling reflects out-of-box coordinates back into [0,1].
        For example, -0.2 becomes 0.2 and 1.2 becomes 0.8; the period-two map
        also handles steps crossing multiple boundaries. The repaired design
        is BOTH evaluated and used for covariance adaptation. It needs no extra
        backend calls and does not pile all infeasible samples onto box edges.
        """
        if self.pending is not None:
            raise ValueError('CMA ask requires tell for the previous proposal')
        if fixed is not None:
            if self.strategy is not None:
                raise ValueError('Shared initialization must precede CMA adaptation')
            point, details, kind = np.asarray(fixed, dtype=float), {}, 'initial'
        elif not self.parents and self.strategy is None:
            round_seed = (self.seed+1009*(self.attempts+1)) % (2**32)
            point = sobol_points(1, round_seed)[0]
            details = {'proposal': 'sobol_fallback_no_successful_observations'}
            kind = 'fallback'
        else:
            if self.strategy is None:
                # mu is the number of successful initial points, not the total
                # attempt count. Failed points have no valid fitness to rank.
                self.strategy = self.strategy_type(
                    self.parents, self.sigma, mu=len(self.parents), lambda_=1,
                    indicator=self._least_contributor)
            # DEAP uses NumPy's legacy global RNG. Temporarily install this
            # optimizer's private state, then restore the caller's state even
            # on error. This isolates serial runs from algorithm ordering.
            # This context is intentionally for the benchmark's serial runner;
            # it must not be used concurrently from multiple threads.
            external_state = np.random.get_state()
            try:
                np.random.set_state(self.rng.get_state())
                individual = self.strategy.generate(self.individual_type)[0]
                self.rng.set_state(np.random.get_state())
            finally:
                np.random.set_state(external_state)
            raw = np.asarray(individual, dtype=float)
            if not np.isfinite(raw).all():
                raise ValueError('CMA generated a nonfinite step')
            point = raw.copy()
            outside = (point < 0) | (point > 1)
            point[outside] = 1-np.abs(1-np.mod(point[outside], 2))
            # Mutate the existing individual to preserve DEAP's offspring/parent
            # tag (_ps), required to update the correct covariance and step size.
            individual[:] = point.tolist()
            parent_index = int(individual._ps[1])
            self.pending = individual
            self.pending_kind = 'offspring'
            return point.copy(), {
                'proposal': 'mo_cmaes', 'parent_count': len(self.strategy.parents),
                'parent_index': parent_index, 'completed_updates': self.updates,
                'parent_sigma': float(self.strategy.sigmas[parent_index]),
                'boundary_repair': 'reflection', 'reflected_coordinates': int(outside.sum())}
        self.pending = self.individual_type(point)
        self.pending_kind = kind
        return point.copy(), details

    def tell(self, x, y):
        """Assign both normalized objectives, then update the MO-CMA-ES strategy.

        Selection uses parents plus the new offspring. An accepted offspring
        inherits its source parent's distribution and updates the evolution
        path/covariance. Step sizes respond to selection success. The runner's
        separate Pareto archive retains all successes even when CMA drops them.
        """
        if self.pending is None or not np.array_equal(x, self.pending):
            raise ValueError('CMA tell must match the pending proposal')
        if y is not None:
            values = np.asarray(y, dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError('CMA needs two finite objectives or None for failure')
            self.pending.fitness.values = tuple(values)
            if self.pending_kind == 'offspring':
                self.strategy.update([self.pending])
                self.updates += 1
            else:
                self.parents.append(self.pending)
        # None skips selection/adaptation; the RNG has still advanced and the
        # attempt still counts. Subsequent calls do not retry it for free.
        self.attempts += 1
        self.pending, self.pending_kind = None, None
