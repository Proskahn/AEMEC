#!/usr/bin/env python3
"""Backward-compatible command-line entrypoint for AEMEC optimization.

The implementation is deliberately split by responsibility:

* :mod:`optimization_lib` is model-neutral Optuna lifecycle and Pareto logic.
* :mod:`aemec_case` owns OpenFOAM/AEMEC geometry, execution, and objectives.
* :mod:`reporting` renders configured CSV and Pareto artifacts.
* :mod:`cli` composes those layers for this AEMEC case.
"""

from cli import build_parser, main, run_optimization
from optimization_lib import OptimizationError

__all__ = ("OptimizationError", "build_parser", "main", "run_optimization")


if __name__ == "__main__":
    raise SystemExit(main())
