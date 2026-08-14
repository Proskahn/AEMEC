"""Project paths and the small set of utilities shared with ``opt``."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "crossover_experiements"
OPT_ROOT = PROJECT_ROOT / "opt"

# ``opt/aemec_opt`` is an application package rather than a repository-level
# installation.  Reuse its well-tested safe case-copy and process helpers
# without duplicating that infrastructure here.
if str(OPT_ROOT) not in sys.path:
    sys.path.insert(0, str(OPT_ROOT))

from aemec_opt.case import (  # noqa: E402
    NORMAL_END_RE,
    case_fingerprint,
    copy_clean_case,
    parse_voltage_sweep_samples,
    rewrite_block_mesh_thickness,
    run_command,
)
from aemec_opt.engine import OptimizationError  # noqa: E402


DEFAULT_SOURCE_CASE = PROJECT_ROOT / "run" / "AEMEC"
DEFAULT_STUDY_NAME = "fixed-current-1Acm2-20s"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results" / DEFAULT_STUDY_NAME
DEFAULT_WORK_DIR = PACKAGE_ROOT / "work" / DEFAULT_STUDY_NAME


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_CASE",
    "DEFAULT_STUDY_NAME",
    "DEFAULT_WORK_DIR",
    "NORMAL_END_RE",
    "OptimizationError",
    "case_fingerprint",
    "copy_clean_case",
    "parse_voltage_sweep_samples",
    "rewrite_block_mesh_thickness",
    "run_command",
]

