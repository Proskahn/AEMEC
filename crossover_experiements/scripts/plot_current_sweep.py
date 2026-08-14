#!/usr/bin/env python3
"""Regenerate current-sweep CSV summaries and figures from stored time series."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crossover_experiments.config import CurrentSweepConfig  # noqa: E402
from crossover_experiments.current_sweep import (  # noqa: E402
    read_current_sweep_timeseries,
    summarize_current_sweep,
    write_current_sweep_points,
)
from crossover_experiments.current_sweep_reporting import (  # noqa: E402
    write_current_sweep_reports,
)
from crossover_experiments.project import (  # noqa: E402
    DEFAULT_CURRENT_SWEEP_OUTPUT_DIR,
    OptimizationError,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate hydrogen-crossover current-sweep reports."
    )
    parser.add_argument(
        "result_dir", nargs="?", type=Path, default=DEFAULT_CURRENT_SWEEP_OUTPUT_DIR
    )
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    try:
        manifest = json.loads(
            (result_dir / "experiment.json").read_text(encoding="utf-8")
        )
        settings = dict(manifest["settings"]["config"])
        settings.pop("signed_targets_a_m2", None)
        settings["current_targets_a_cm2"] = tuple(settings["current_targets_a_cm2"])
        config = CurrentSweepConfig(**settings)
        samples = read_current_sweep_timeseries(result_dir / "data/timeseries.csv")
        points = summarize_current_sweep(samples, config)
        write_current_sweep_points(result_dir / "current_sweep.csv", points)
        write_current_sweep_reports(
            result_dir, samples, points, config.membrane_area_m2
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, OptimizationError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"Wrote {result_dir / 'current_sweep.csv'}")
    print(f"Wrote {result_dir / 'current_sweep.png'}")
    print(f"Wrote {result_dir / 'current_sweep_timeseries.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

