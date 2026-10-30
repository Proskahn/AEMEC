# Pinned OptunaHub MoCmaSampler

`mocma.py` is an **unmodified** copy of the contributed sampler from
[OptunaHub](https://hub.optuna.org/samplers/mocma/), by Yoshihiko Ozaki.
Its full MIT license is in [LICENSE](LICENSE).

- Repository: <https://github.com/optuna/optunahub-registry>
- Source: `package/samplers/mocma/mocma.py`
- Commit: `7508d98688d3eb55aab09002ea6a8dedfbe5b2c6`
- SHA-256: `a0a6fb2a9e32d9025c20ebfb0a05693d3cbb5b66fee18f396ddf1135d6793cf4`
- Integration tested with Optuna **4.9.0**, pinned because upstream uses private APIs.

The local copy avoids runtime downloads on the CFD server and prevents upstream
changes from silently changing the benchmark. No `optunahub` or `deap` package
is needed at runtime. `algorithms.py` supplies the explicit three-dimensional
unit-box search space and registers enqueued initial designs through the sampler's
own generation bookkeeping. The remaining sampling, adaptation and selection
come directly from this source. The benchmark manifest records its hash.

## Observed implementation caveats

These source-level observations are not fixed locally, so the benchmark measures
this **specific upstream sampler**, not a corrected reimplementation of the paper:

1. The covariance update uses `p_c @ p_c.T` with a one-dimensional vector.
   NumPy returns a scalar, broadcast into the covariance matrix, rather than the
   rank-one matrix `np.outer(p_c, p_c)` expected from the evolution path.
2. When a nondominated front overflows the population, selection computes
   each point's hypervolume contribution and **retains the smallest contributor**
   first (`np.argmin`). This differs from eliminating the smallest contributor
   to preserve hypervolume.

Both can affect optimizer quality. Integration tests verify initialization,
generation updates, bounds, failures, reproducibility and CFD accounting;
they do not certify mathematical equivalence to published MO-CMA-ES.
Reassess these details before drawing general conclusions about CMA-ES from a
ranking of this implementation. If upgrading upstream, review these operations,
update the commit/hash/license as needed, and rerun the benchmark tests.
