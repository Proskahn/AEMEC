"""Repository paths and shared AEMEC runtime utilities."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "polarization_curve"
OPT_ROOT = PROJECT_ROOT / "opt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OPT_ROOT) not in sys.path:
    sys.path.insert(0, str(OPT_ROOT))

from aemec_opt.case import (  # noqa: E402
    AemecEvaluationConfig,
    FATAL_OUTPUT_RE,
    NORMAL_END_RE,
    case_fingerprint,
    configure_voltage_sweep,
    copy_clean_case,
    run_command,
    validate_path_layout,
)
from aemec_opt.engine import OptimizationError  # noqa: E402


DEFAULT_SOURCE_CASE = PROJECT_ROOT / "run" / "AEMEC"
DEFAULT_WORK_DIR = PACKAGE_ROOT / "work" / "former-165s-voltage-sweep"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results" / "former-165s-voltage-sweep"


__all__ = [
    "AemecEvaluationConfig",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_CASE",
    "DEFAULT_WORK_DIR",
    "FATAL_OUTPUT_RE",
    "NORMAL_END_RE",
    "OptimizationError",
    "case_fingerprint",
    "configure_voltage_sweep",
    "copy_clean_case",
    "run_command",
    "validate_path_layout",
]
