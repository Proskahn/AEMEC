#!/usr/bin/env python3
"""Command-line entry point for the CFD-only algorithm benchmark.

Example from the repository root::

    python benchmark/run_benchmark.py \
        --backend aemec --output benchmark/results/benchmark_aemec \
        --budget 64 --initial 16 --seeds 42 43 44 45 46

The only evaluator runs actual OpenFOAM CFD (fixed current by default) and requires the built
AEMEC solver in a configured OpenFOAM environment. Use --dry-run to preview.

Reading order for the implementation:
  1. aemec_benchmark/cli.py: arguments and backend construction.
  2. aemec_benchmark/runner.py: shared initialization, ask/evaluate/tell, reports.
  3. aemec_benchmark/algorithms.py: GP BO, TPE BO and NSGA-II proposals.
  4. aemec_benchmark/cmaes.py: pinned OptunaHub MoCmaSampler and protocol metadata.
  5. aemec_benchmark/llamea.py: optional LLaMEA code evolution under the CFD budget.
  6. aemec_benchmark/metrics.py: Pareto sets, hypervolume and analytic EHVI.
  7. aemec_benchmark/proxy_search.py: ELA proxy proposals, never CFD responses.
  8. aemec_benchmark/evaluation.py: common OpenFOAM evaluation contract.

This file only arranges imports and forwards to the CLI. It does not launch
evaluations when imported; execution starts under the __main__ guard below.
"""
from pathlib import Path
import sys

# Resolve imports from this file rather than the shell's working directory.
# benchmark/ owns the comparison package, opt/ supplies the shared OpenFOAM
# evaluator/result contract, and the root exposes ELA expressions to ela_proxy.
# No optimization CLI or resumable study is launched by importing that adapter.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'benchmark'), str(ROOT/'opt'), str(ROOT)]
from aemec_benchmark.cli import main


if __name__ == '__main__':
    # Propagate the CLI's success/error/interruption code to the calling shell.
    raise SystemExit(main())
