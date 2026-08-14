"""End-to-end OpenFOAM runner for the 0--2 A/cm2 crossover sweep."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import CurrentSweepConfig, configure_current_sweep_case
from .current_sweep import (
    CurrentSweepPoint,
    read_current_sweep_timeseries,
    summarize_current_sweep,
    write_current_sweep_points,
    write_current_sweep_timeseries,
)
from .current_sweep_reporting import write_current_sweep_reports
from .parsing import ExperimentSample, parse_experiment_log
from .project import (
    NORMAL_END_RE,
    OptimizationError,
    case_fingerprint,
    copy_clean_case,
    rewrite_block_mesh_thickness,
    run_command,
)
from .runner import REQUIRED_MESH_REGIONS


CURRENT_SWEEP_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _overlap(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


class CurrentSweepRunner:
    def __init__(
        self,
        source_case: Path,
        work_dir: Path,
        output_dir: Path,
        mesh_command: Sequence[str],
        solver_command: Sequence[str],
        config: CurrentSweepConfig,
        timeout_s: float | None,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.source_case = source_case.resolve()
        self.work_dir = work_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.work_case = self.work_dir / "case"
        self.mesh_command = tuple(mesh_command)
        self.solver_command = tuple(solver_command)
        self.config = config
        self.timeout_s = timeout_s
        self.overwrite = overwrite
        self.dry_run = dry_run
        config.validate()
        if not self.source_case.is_dir():
            raise OptimizationError(f"OpenFOAM source case not found: {self.source_case}")
        if not self.mesh_command or not self.solver_command:
            raise OptimizationError("Mesh and solver commands cannot be empty")
        for name_a, path_a, name_b, path_b in (
            ("source case", self.source_case, "work directory", self.work_dir),
            ("source case", self.source_case, "output directory", self.output_dir),
            ("work directory", self.work_dir, "output directory", self.output_dir),
        ):
            if _overlap(path_a, path_b):
                raise OptimizationError(
                    f"Unsafe overlapping paths: {name_a}={path_a} and {name_b}={path_b}"
                )

    @property
    def paths(self) -> dict[str, Path]:
        return {
            "manifest": self.output_dir / "experiment.json",
            "metadata": self.output_dir / "case.json",
            "mesh_log": self.output_dir / "logs/mesh.log",
            "solver_log": self.output_dir / "logs/solver.log",
            "timeseries": self.output_dir / "data/timeseries.csv",
            "points": self.output_dir / "current_sweep.csv",
        }

    def _settings(self) -> dict[str, object]:
        return {
            "schema_version": CURRENT_SWEEP_SCHEMA_VERSION,
            "source_case": str(self.source_case),
            "source_case_fingerprint": case_fingerprint(self.source_case),
            "mesh_command": list(self.mesh_command),
            "solver_command": list(self.solver_command),
            "config": self.config.settings(),
        }

    def _prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths["manifest"]
        settings = self._settings()
        if path.is_file():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise OptimizationError(f"Cannot read sweep manifest: {path}") from exc
            if previous.get("settings") != settings:
                raise OptimizationError(
                    "The output directory belongs to a different current sweep. "
                    "Choose new --output-dir and --work-dir paths."
                )
            return
        _write_json(
            path,
            {
                "created_at_utc": _utc_now(),
                "description": "AEMEC hydrogen-crossover current-density sweep",
                "flux_definition": "total anode H2 crossover rate / membrane area",
                "settings": settings,
            },
        )

    def _verify_mesh(self) -> None:
        expected = [self.work_case / "constant/polyMesh/points"] + [
            self.work_case / "constant" / region / "polyMesh/points"
            for region in REQUIRED_MESH_REGIONS
        ]
        missing = [path for path in expected if not path.is_file()]
        if missing:
            raise OptimizationError(
                "Mesh command completed but required mesh files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )

    def _write_products(
        self, samples: Sequence[ExperimentSample], points: Sequence[CurrentSweepPoint]
    ) -> None:
        write_current_sweep_timeseries(
            self.paths["timeseries"], samples, self.config.membrane_area_m2
        )
        write_current_sweep_points(self.paths["points"], points)
        write_current_sweep_reports(
            self.output_dir, samples, points, self.config.membrane_area_m2
        )

    def _load_completed(self) -> tuple[list[ExperimentSample], list[CurrentSweepPoint]] | None:
        if self.overwrite or not self.paths["metadata"].is_file() or not self.paths["timeseries"].is_file():
            return None
        try:
            metadata = json.loads(self.paths["metadata"].read_text(encoding="utf-8"))
            if metadata.get("status") != "completed":
                return None
            samples = read_current_sweep_timeseries(self.paths["timeseries"])
            points = summarize_current_sweep(samples, self.config)
        except (OSError, json.JSONDecodeError, OptimizationError):
            return None
        return samples, points

    def _recover_log(self) -> tuple[list[ExperimentSample], list[CurrentSweepPoint]] | None:
        if self.overwrite or self.dry_run or not self.paths["solver_log"].is_file():
            return None
        try:
            output = self.paths["solver_log"].read_text(encoding="utf-8", errors="replace")
            if not NORMAL_END_RE.search(output):
                return None
            samples = parse_experiment_log(output)
            points = summarize_current_sweep(samples, self.config)
        except (OSError, OptimizationError):
            return None
        self._write_products(samples, points)
        _write_json(
            self.paths["metadata"],
            {
                "schema_version": CURRENT_SWEEP_SCHEMA_VERSION,
                "status": "completed",
                "recovered_from_solver_log": True,
                "recovered_at_utc": _utc_now(),
                "work_case": str(self.work_case),
                "sample_count": len(samples),
                "accepted_point_count": len(points),
                "points": [asdict(point) for point in points],
            },
        )
        return samples, points

    def _run_solver(self) -> tuple[list[ExperimentSample], list[CurrentSweepPoint]] | None:
        started = time.monotonic()
        metadata: dict[str, object] = {
            "schema_version": CURRENT_SWEEP_SCHEMA_VERSION,
            "status": "configuring",
            "started_at_utc": _utc_now(),
            "work_case": str(self.work_case),
            "mesh_log": str(self.paths["mesh_log"]),
            "solver_log": str(self.paths["solver_log"]),
            "timeseries_csv": str(self.paths["timeseries"]),
            "current_sweep_csv": str(self.paths["points"]),
        }
        _write_json(self.paths["metadata"], metadata)
        try:
            copy_clean_case(self.source_case, self.work_case)
            if self.dry_run:
                configure_current_sweep_case(self.work_case, self.config)
                metadata.update(
                    status="configured",
                    completed_at_utc=_utc_now(),
                    elapsed_s=time.monotonic() - started,
                )
                _write_json(self.paths["metadata"], metadata)
                return None

            # The source-case checker invoked by ``make mesh`` validates the
            # baseline fixed-current controls. Change only geometry before
            # meshing, then install the sweep controls after the mesh passes.
            rewrite_block_mesh_thickness(
                self.work_case / "system/blockMeshDict",
                self.config.membrane_thickness_um,
            )
            metadata["status"] = "meshing"
            _write_json(self.paths["metadata"], metadata)
            run_command(
                self.mesh_command,
                self.work_case,
                self.paths["mesh_log"],
                self.timeout_s,
            )
            self._verify_mesh()
            configure_current_sweep_case(self.work_case, self.config)

            metadata["status"] = "solving"
            _write_json(self.paths["metadata"], metadata)
            output = run_command(
                self.solver_command,
                self.work_case,
                self.paths["solver_log"],
                self.timeout_s,
            )
            if not NORMAL_END_RE.search(output):
                raise OptimizationError("Solver output does not contain the normal OpenFOAM 'End'")
            samples = parse_experiment_log(output)
            write_current_sweep_timeseries(
                self.paths["timeseries"], samples, self.config.membrane_area_m2
            )
            points = summarize_current_sweep(samples, self.config)
            write_current_sweep_points(self.paths["points"], points)
            write_current_sweep_reports(
                self.output_dir, samples, points, self.config.membrane_area_m2
            )
            metadata.update(
                status="completed",
                completed_at_utc=_utc_now(),
                elapsed_s=time.monotonic() - started,
                sample_count=len(samples),
                accepted_point_count=len(points),
                points=[asdict(point) for point in points],
            )
            _write_json(self.paths["metadata"], metadata)
            return samples, points
        except Exception as exc:
            metadata.update(
                status="failed",
                completed_at_utc=_utc_now(),
                elapsed_s=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_json(self.paths["metadata"], metadata)
            raise

    def run(self) -> int:
        self._prepare()
        result = self._load_completed()
        if result is not None:
            print("[resume] current sweep already completed")
            samples, points = result
            self._write_products(samples, points)
        else:
            result = self._recover_log()
            if result is not None:
                print("[recover] current sweep from completed solver log")
                samples, points = result
            else:
                print(f"[run] current sweep -> {self.work_case}")
                result = self._run_solver()
                if result is None:
                    print("[configured] current sweep (dry run)")
                    return 0
                samples, points = result
        print(f"[done] {len(points)} accepted current targets")
        print(f"[report] {self.paths['points']}")
        print(f"[report] {self.output_dir / 'current_sweep.png'}")
        print(f"[report] {self.output_dir / 'current_sweep_timeseries.png'}")
        return 0

