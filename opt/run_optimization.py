#!/usr/bin/env python3
"""Backward-compatible command-line entrypoint for AEMEC optimization.

The implementation is deliberately split by responsibility:

* :mod:`aemec_opt.engine` is model-neutral Optuna lifecycle and Pareto logic.
* :mod:`aemec_opt.case` owns OpenFOAM/AEMEC geometry, execution, and objectives.
* :mod:`aemec_opt.reporting` renders configured CSV and Pareto artifacts.
* :mod:`aemec_opt.cli` composes those layers for this AEMEC case.
"""

from aemec_opt.cli import build_parser, main, run_optimization
from aemec_opt.engine import OptimizationError

__all__ = ("OptimizationError", "build_parser", "main", "run_optimization")


if __name__ == "__main__":
    raise SystemExit(main())
