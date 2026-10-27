#!/usr/bin/env python3
"""Command-line entry point for the four-algorithm benchmark.

Example from the repository root::

    ELA/.venv39/bin/python opt/run_benchmark.py \
        --backend proxies --output opt/results/benchmark_example \
        --budget 64 --initial 16 --seeds 42 43 44 45 46

Reading order for the implementation:
  1. aemec_opt/benchmark_cli.py: arguments and backend construction.
  2. aemec_opt/benchmark.py: shared initialization, ask/evaluate/tell, reports.
  3. aemec_opt/benchmark_algorithms.py: GP BO, TPE BO and NSGA-II proposals.
  4. aemec_opt/benchmark_cmaes.py: bounded multiobjective CMA-ES proposals.
  5. aemec_opt/benchmark_metrics.py: Pareto sets, hypervolume and analytic EHVI.

This file only arranges imports and forwards to the CLI. It does not launch
evaluations when imported; execution starts under the __main__ guard below.
"""
from pathlib import Path
import sys

# Python adds opt/ to the import path when this file is run as a script. Also
# expose the repository root so the proxy backend can import the sibling ELA
# expression runtime. The optimization package itself remains under opt/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aemec_opt.benchmark_cli import main


if __name__ == '__main__':
    # Propagate the CLI's success/error/interruption code to the calling shell.
    raise SystemExit(main())
