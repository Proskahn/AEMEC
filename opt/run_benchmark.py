#!/usr/bin/env python3
"""Compatibility launcher; benchmark implementation lives in benchmark/."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'benchmark'), str(ROOT/'opt'), str(ROOT)]
from aemec_benchmark.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
