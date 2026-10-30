"""GP and Optuna optimizers with the shared benchmark algorithm registry.

All inputs here are normalized to [0, 1]^3, in the order thickness, temperature,
porosity. Objective pairs use the runner's fixed affine normalization and are
both minimized. Conversion to physical units and simulator calls live elsewhere.

The runner calls ``ask(fixed=point)`` during shared initialization, otherwise
``ask()``. It then evaluates the returned design and calls ``tell(point, y)``;
``y=None`` denotes a failed attempt. Every algorithm owns an independent history.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import optuna
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from threadpoolctl import threadpool_limits

from .metrics import expected_hypervolume_improvement
from .cmaes import require_mocma

DEFAULT_ALGORITHMS = ('gp', 'tpe', 'nsga2', 'cmaes')
ALGORITHMS = DEFAULT_ALGORITHMS + ('ela_proxy', 'llamea')
LABELS = {'gp': 'GP BO (EHVI)', 'tpe': 'TPE BO (MOTPE)', 'nsga2': 'NSGA-II',
          'cmaes': 'MO-CMA-ES (OptunaHub)', 'ela_proxy': 'ELA proxy search',
          'llamea': 'LLaMEA optimizer evolution'}


def sobol_points(count, seed):
    """Generate reproducible space-filling points in a three-dimensional unit box.

    Scrambling changes the design between seeds. Generate a power-of-two Sobol
    block and take the requested prefix; the full Sobol balance properties apply
    to power-of-two counts, which are preferable for shared initialization.
    """
    return qmc.Sobol(3, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count]


class OptunaAlgorithm:
    """Adapt TPE, NSGA-II and OptunaHub MoCmaSampler to the common interface.

    The in-memory study stores only this run's observations. It does not resume
    another study or obtain observations from competing algorithms. One trial
    is pending at a time, so each ask must be followed by its corresponding tell.
    """
    def __init__(self, algorithm, initial, seed):
        """Set the sampler/seed; ``initial`` is the NSGA-II and MO-CMA population size."""
        with warnings.catch_warnings():
            # These Optuna features are intentionally selected and documented.
            # Suppress API experimental-status warnings, not optimizer errors.
            warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
            if algorithm == 'tpe':
                # MOTPE divides past trials into relatively good/bad sets using
                # multiobjective information, models their input densities, and
                # chooses promising density-ratio candidates. It fits no GP.
                # multivariate=True models the inputs jointly. n_ei_candidates
                # controls internal proposal effort, not simulator evaluations.
                # No constant-liar treatment is needed for this serial runner.
                sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=initial,
                                                    multivariate=True, n_ei_candidates=64,
                                                    constant_liar=False)
            elif algorithm == 'nsga2':
                # NSGA-II retains parents by nondominated sorting and crowding
                # distance. SBX creates real-valued offspring near/between their
                # parents; eta=20 concentrates offspring relatively near them.
                # Optuna's default mutation independently resets a variable to
                # a sampled value with probability 1/3. This is not polynomial
                # mutation. The population matches the shared initial count.
                sampler = optuna.samplers.NSGAIISampler(
                    seed=seed, population_size=initial, mutation_prob=1/3,
                    crossover=optuna.samplers.nsgaii.SBXCrossover(eta=20), crossover_prob=.9)
            elif algorithm == 'cmaes':
                # This is the contributed multiobjective sampler, not Optuna's
                # single-objective CmaEsSampler. An explicit static space lets
                # it assign every initial trial to a distinct population slot.
                sampler = require_mocma()(
                    search_space={f'x{j}': optuna.distributions.FloatDistribution(0., 1.)
                                  for j in range(3)},
                    popsize=initial, seed=seed)
            else:
                raise ValueError(f'Unknown Optuna algorithm: {algorithm}')
        self.study = optuna.create_study(directions=['minimize', 'minimize'], sampler=sampler)
        self.algorithm = algorithm
        self.population_size = initial
        self.trial = None

    def ask(self, fixed=None):
        """Return one unit-box design and diagnostics for its pending trial.

        Queueing fixed values still creates a real Optuna trial. Reporting its
        outcome through tell makes the common initial observations available to
        the sampler's own density model or evolutionary population.
        """
        if self.trial is not None:
            raise ValueError('Complete the pending trial with tell before asking again')
        if fixed is not None:
            self.study.enqueue_trial({f'x{j}': float(v) for j, v in enumerate(fixed)})
        self.trial = self.study.ask()
        details = {}
        if self.algorithm == 'nsga2':
            # Fixed/enqueued parameters bypass sample_relative, where Optuna
            # normally tags generations. Register them explicitly so the shared
            # observations really are population 0, not unassigned warm-starts.
            details['generation'] = self.study.sampler.get_trial_generation(
                self.study, self.study.get_trials(deepcopy=False)[-1])
        elif self.algorithm == 'cmaes' and fixed is not None:
            # Enqueued parameters bypass sample_relative. MoCmaSampler needs
            # its mocma:g and mocma:k tags on ALL completed trials, including
            # the shared Sobol points. Invoke its normal initial-population
            # bookkeeping once. During generation 0 it returns {} and samples
            # no coordinates, so the exact shared design is preserved.
            # Only shared initialization uses fixed points in this runner.
            frozen = self.study._storage.get_trial(self.trial._trial_id)
            space = self.study.sampler.infer_relative_search_space(self.study, frozen)
            self.study.sampler.sample_relative(self.study, frozen, space)
        point = np.array([self.trial.suggest_float(f'x{j}', 0., 1.) for j in range(3)])
        if self.algorithm == 'cmaes':
            attrs = self.study._storage.get_trial(self.trial._trial_id).system_attrs
            generation = attrs['mocma:g']
            details = {'proposal': 'mocma_initial_population' if generation == 0 else 'optunahub_mocma',
                       'generation': generation, 'population_size': self.population_size,
                       'instance': attrs['mocma:k'], 'parent_trial_id': attrs.get('mocma:parent_id')}
        self.pending_point = point.copy()
        return point, details

    def tell(self, x, y):
        """Complete the pending trial, or mark it failed without a penalty value.

        Check that the response belongs to the point returned by ask and has
        two finite objectives. A failure consumes the runner's attempt
        budget but does not become a fabricated high objective value.
        """
        if self.trial is None:
            raise ValueError('Ask for a trial before reporting its outcome')
        if not np.array_equal(x, self.pending_point):
            raise ValueError('The reported design does not match the pending trial')
        if y is not None and (np.shape(y) != (2,) or not np.isfinite(y).all()):
            raise ValueError('Expected two finite objectives, or None for a failed attempt')
        if y is None:
            self.study.tell(self.trial, state=optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(self.trial, [float(v) for v in y])
        self.trial = None


class GaussianProcessBO:
    """Independent two-GP Bayesian optimization with single-point analytic EHVI.

    Each adaptive step fits one regression model per objective, predicts both
    mean and uncertainty on a finite candidate pool, and chooses the largest
    expected increase in dominated hypervolume. Candidate scoring is cheap model
    evaluation; the runner evaluates only the selected point on the backend.
    """
    def __init__(self, seed, reference, candidates=2048, restarts=1):
        """Store the fixed reference and controls for acquisition/kernel fitting.

        ``candidates`` is the Sobol pool size, not the expensive evaluation
        budget. ``restarts`` is the number of additional optimizer starts used
        when fitting each GP's kernel hyperparameters by marginal likelihood.
        """
        self.seed, self.reference = seed, np.asarray(reference, dtype=float)
        self.candidates, self.restarts = candidates, restarts
        # attempts includes failed points so we avoid immediately retrying them.
        # x/y contain only successful observations that can train the GPs.
        self.attempts, self.x, self.y = [], [], []

    def ask(self, fixed=None):
        """Return a design and proposal diagnostics without evaluating the backend."""
        if fixed is not None:
            return np.asarray(fixed), {}
        # A new deterministic pool per attempt avoids always searching the same
        # finite locations. The seed depends on this run only, not execution order
        # of the other algorithms; modulo keeps it in NumPy's supported range.
        round_seed = (self.seed+1009*(len(self.attempts)+1)) % (2**32)
        # Include boundaries explicitly: Sobol interior points alone can miss
        # corner optima on nearly monotonic landscapes such as the AEMEC samples.
        pool = np.vstack([sobol_points(self.candidates, round_seed),
                          list(itertools.product((0., 1.), repeat=3)), [.5, .5, .5]])
        if self.attempts:
            # Broadcasting yields (n_candidates, n_attempts, 3). The maximum
            # coordinate difference is the L-infinity distance. Exclude points
            # effectively identical to ANY prior attempt in normalized units.
            separation = np.max(np.abs(pool[:, None, :]-np.asarray(self.attempts)[None, :, :]), axis=2)
            pool = pool[np.all(separation > 1e-10, axis=1)]
        if not len(pool):
            raise ValueError('GP acquisition candidate pool is exhausted')
        if len(self.x) < 4:
            # Failures can leave fewer successful points than the initial count.
            # Four is a minimum policy for fitting in 3D, not an accuracy or rank
            # guarantee. Keep exploring and disclose the fallback in the log.
            return pool[0], {'proposal': 'sobol_fallback_insufficient_successful_observations'}
        means, deviations, details = [], [], []
        for j in range(2):
            # k(x,x') = amplitude * Matern_5/2(x,x'; length_scales) + white_noise.
            # Three separate length scales provide automatic relevance
            # determination (ARD): each input may vary on a different scale.
            # Bounds constrain hyperparameter optimization, not physical inputs.
            kernel = (ConstantKernel(1., (1e-3, 1e3))*Matern(
                length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(1e-6, (1e-10, 1e-2)))
            # normalize_y=True additionally standardizes THIS objective using
            # only THIS optimizer's successful observations. Predictions are
            # automatically returned to the runner's fixed normalized units.
            # alpha is small diagonal numerical jitter, separate from the fitted
            # WhiteKernel observation-noise term.
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-10,
                                          n_restarts_optimizer=self.restarts, random_state=(round_seed+j) % (2**32))
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                # Limit native linear-algebra threads for these small fits and
                # retain convergence/bound warnings in the proposal diagnostics.
                warnings.simplefilter('always')
                gp.fit(np.asarray(self.x), np.asarray(self.y)[:, j])
                mu, sd = gp.predict(pool, return_std=True)
            # sklearn includes WhiteKernel in test variance; EHVI uses the latent
            # function variance, with fitted observation noise removed.
            # kernel_.k2 is the fitted WhiteKernel in the sum above. Its variance
            # is in GP-standardized units, hence multiplication by output std^2
            # before subtracting from sd^2. Roundoff can leave tiny negatives.
            noise = gp.kernel_.k2.noise_level*float(gp._y_train_std)**2
            means.append(mu)
            deviations.append(np.sqrt(np.maximum(sd*sd-noise, 0)))
            details.append({'kernel': str(gp.kernel_),
                            'warnings': sorted(set(str(w.message) for w in caught))})
        mean, std = np.column_stack(means), np.column_stack(deviations)
        # Both arrays are (n_candidates, 2). The acquisition assumes independent
        # predictive marginals, consistent with fitting the objectives separately.
        acquisition = expected_hypervolume_improvement(mean, std, self.y, self.reference)
        if not np.isfinite(acquisition).all():
            raise ValueError('Nonfinite GP acquisition scores')
        if acquisition.max() > 0:
            index, rule = int(np.argmax(acquisition)), 'analytic_ehvi'
        else:
            # A numerically zero acquisition offers no ranking by improvement.
            # Use uncertainty to select an exploratory point, and record that
            # this step did not use a positive EHVI value.
            index, rule = int(np.argmax(np.sum(std*std, axis=1))), 'zero_ehvi_max_variance'
        return pool[index], {'proposal': rule, 'ehvi': float(acquisition[index]),
                             'predictive_mean_normalized': mean[index].tolist(),
                             'latent_std_normalized': std[index].tolist(), 'gp_models': details}

    def tell(self, x, y):
        """Record every attempt, but append only finite successful responses to training.

        The runner validates responses before calling this method. Copy arrays
        so a caller cannot accidentally change the stored optimization history.
        """
        self.attempts.append(np.asarray(x).copy())
        if y is not None:
            self.x.append(np.asarray(x).copy())
            self.y.append(np.asarray(y).copy())
