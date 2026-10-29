"""Direct multiobjective search on ELA proxies, followed by external CFD evaluation.

The model expressions remain frozen. Each ask optimizes a randomly weighted,
augmented Tchebycheff score over a fresh candidate pool; tell excludes attempted
designs, including failures. Only the runner may call CFD and record objectives.
"""
from __future__ import annotations

import itertools
import numpy as np

from .algorithms import sobol_points


class ELAProxySearch:
    def __init__(self, seed, assets, candidates=2048):
        if candidates < 16:
            raise ValueError('Proxy candidates must be >=16')
        self.seed, self.assets, self.candidates = seed, assets, candidates
        self.attempts, self.pending = [], None
        self.scored_designs, self.function_calls = 0, 0

    def _score(self, pool):
        """Score mathematical expressions, retaining finite rows if others fail.

        Batched evaluation is normally two cheap function calls. Expressions
        can be singular at domain boundaries; a rejected batch is retried by
        row to identify valid candidates, without spending any CFD attempts.
        """
        physical = self.assets.lower+pool*(self.assets.upper-self.assets.lower)
        scores = np.full((len(pool), 2), np.nan)
        self.scored_designs += len(pool)
        for j, model in enumerate(self.assets.models):
            self.function_calls += 1
            try:
                scores[:, j] = model(physical)
            except (ValueError, ArithmeticError):
                for i, point in enumerate(physical):
                    self.function_calls += 1
                    try:
                        scores[i, j] = model(point)
                    except (ValueError, ArithmeticError):
                        pass
        return scores

    def ask(self, fixed=None):
        """Propose a unit-box design; never substitute a proxy value for a response."""
        if self.pending is not None:
            raise ValueError('Tell the previous CFD outcome before requesting another proposal')
        if fixed is not None:
            point = np.asarray(fixed, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all() or np.any((point < 0) | (point > 1)):
                raise ValueError('Invalid initial design')
            self.pending = point.copy()
            return point.copy(), {'proposal': 'shared_initial'}

        round_seed = (self.seed+1009*(len(self.attempts)+1)) % (2**32)
        pool = np.unique(np.vstack([sobol_points(self.candidates, round_seed),
                                   list(itertools.product((0., 1.), repeat=3)), [.5, .5, .5]]), axis=0)
        if self.attempts:
            distance = np.min(np.linalg.norm(pool[:, None, :]-np.asarray(self.attempts)[None, :, :], axis=2), axis=1)
            pool, distance = pool[distance > 1e-10], distance[distance > 1e-10]
        else:
            distance = np.ones(len(pool))
        if not len(pool):
            raise ValueError('Proxy proposal pool exhausted')
        scores = self._score(pool)
        valid = np.isfinite(scores).all(axis=1)
        details = {'candidate_count': len(pool), 'finite_proxy_candidates': int(valid.sum()),
                   'proxy_scoring_designs_total': self.scored_designs,
                   'proxy_function_calls_total': self.function_calls}
        if not valid.any():
            # No valid model ranking is available. Keep the CFD workflow moving
            # with an explicitly logged space-filling proposal, not fake values.
            index = int(np.argmax(distance))
            details['proposal'] = 'all_proxy_scores_invalid_space_filling_fallback'
        else:
            indices = np.flatnonzero(valid)
            values = scores[valid]
            # Pool normalization only balances the cheap search objectives.
            # It is independent of the fixed CFD normalization used for metrics.
            # Divide by magnitudes first to avoid overflow for extreme expressions.
            magnitude = np.maximum(np.max(np.abs(values), axis=0), 1.)
            scaled = values/magnitude
            minimum, maximum = scaled.min(axis=0), scaled.max(axis=0)
            span = maximum-minimum
            scaled = (scaled-minimum)/np.where(span > 0, span, 1.)
            weights = np.maximum(np.random.default_rng(round_seed).dirichlet([1., 1.]), 1e-6)
            weights /= weights.sum()
            # The augmentation favors dominating points among Chebycheff ties.
            # Varying weights explores compromises between the two proxies.
            weighted = scaled*weights
            scalar = np.max(weighted, axis=1)+.05*np.sum(weighted, axis=1)
            tied = np.flatnonzero(np.isclose(scalar, scalar.min(), rtol=0, atol=1e-12))
            best = int(tied[np.argmax(distance[indices[tied]])])
            index = int(indices[best])
            details.update({'proposal': 'ela_proxy_augmented_tchebycheff',
                            'weights': weights.tolist(), 'proxy_search_score': float(scalar[best]),
                            'proxy_values_dimensionless': scores[index].tolist()})
        self.pending = pool[index].copy()
        return self.pending.copy(), details

    def tell(self, x, y):
        """Exclude the attempted design even if CFD failed; do not retrain proxies.

        The runner owns all CFD observations, Pareto sets, and metrics. This
        direct-search baseline uses y only through that external assessment;
        it does not claim to learn a calibrated CFD surrogate from feedback.
        """
        point = np.asarray(x, dtype=float)
        if self.pending is None or not np.array_equal(point, self.pending):
            raise ValueError('CFD outcome does not correspond to the pending proposal')
        self.attempts.append(point.copy())
        self.pending = None
