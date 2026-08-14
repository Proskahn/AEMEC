"""Resumable end-to-end runner for membrane-thickness experiments."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import ExperimentConfig, case_label, configure_case
from .parsing import (
    ExperimentSample,
    ExperimentSummary,
    parse_experiment_log,
    read_timeseries_csv,
    summarize_samples,
    validate_completed_run,
    write_timeseries_csv,
)
from .project import (
    NORMAL_END_RE,
    OptimizationError,
    case_fingerprint,
    copy_clean_case,
    run_command,
)
from .reporting import write_reports


SCHEMA_VERSION = 1
REQUIRED_MESH_REGIONS = (
    "anode",
    "cathode",
    "electrolyte",
    "interconnect",
    "phiECathode",
    "phiEAnode",
    "phiAnion",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _paths_overlap(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


class ExperimentRunner:
    """Configure, run, archive, resume, and report a complete study."""

    def __init__(
        self,
        source_case: Path,
        work_dir: Path,
        output_dir: Path,
        mesh_command: Sequence[str],
        solver_command: Sequence[str],
        config: ExperimentConfig,
        timeout_s: float | None,
        overwrite: bool = False,
        fail_fast: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.source_case = source_case.resolve()
        self.work_dir = work_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.mesh_command = tuple(mesh_command)
        self.solver_command = tuple(solver_command)
        self.config = config
        self.timeout_s = timeout_s
        self.overwrite = overwrite
        self.fail_fast = fail_fast
        self.dry_run = dry_run
        config.validate()
        if not self.source_case.is_dir():
            raise OptimizationError(f"OpenFOAM source case not found: {self.source_case}")
        if not self.mesh_command or not self.solver_command:
            raise OptimizationError("Mesh and solver commands cannot be empty")
        for first_name, first, second_name, second in (
            ("source case", self.source_case, "work directory", self.work_dir),
            ("source case", self.source_case, "output directory", self.output_dir),
            ("work directory", self.work_dir, "output directory", self.output_dir),
        ):
            if _paths_overlap(first, second):
                raise OptimizationError(
                    f"Unsafe overlapping paths: {first_name}={first} and {second_name}={second}"
                )

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "experiment.json"

    def _settings(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_case": str(self.source_case),
            "source_case_fingerprint": case_fingerprint(self.source_case),
            "mesh_command": list(self.mesh_command),
            "solver_command": list(self.solver_command),
            "config": self.config.settings(),
        }

    def _prepare_manifest(self) -> None:
        settings = self._settings()
        if self.manifest_path.is_file():
            try:
                previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise OptimizationError(f"Cannot read study manifest: {self.manifest_path}") from exc
            if previous.get("settings") != settings:
                raise OptimizationError(
                    "The output directory belongs to a different experiment setup. "
                    "Choose a new --output-dir (and --work-dir)."
                )
            return
        _write_json(
            self.manifest_path,
            {
                "created_at_utc": _utc_now(),
                "description": "AEMEC fixed-current membrane-thickness crossover study",
                "settings": settings,
            },
        )

    def _case_paths(self, thickness_um: float) -> dict[str, Path]:
        label = case_label(thickness_um)
        return {
            "work": self.work_dir / label,
            "mesh_log": self.output_dir / "logs" / f"{label}_mesh.log",
            "solver_log": self.output_dir / "logs" / f"{label}_solver.log",
            "data": self.output_dir / "data" / f"{label}_timeseries.csv",
            "metadata": self.output_dir / "cases" / f"{label}.json",
        }

    def _existing_complete(
        self, thickness_um: float, paths: dict[str, Path]
    ) -> tuple[list[ExperimentSample], ExperimentSummary] | None:
        if self.overwrite or not paths["metadata"].is_file() or not paths["data"].is_file():
            return None
        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if metadata.get("status") != "completed" or not math.isclose(
            float(metadata.get("thickness_um", -1)), thickness_um, abs_tol=1.0e-12
        ):
            return None
        try:
            samples = read_timeseries_csv(paths["data"])
            validate_completed_run(
                samples, self.config.duration_s, self.config.delta_t_s
            )
            summary = summarize_samples(
                thickness_um, samples, self.config.final_window_s
            )
            self._validate_scientific_endpoint(summary)
        except OptimizationError:
            return None
        return samples, summary

    @staticmethod
    def _verify_mesh(case_path: Path) -> None:
        expected = [case_path / "constant/polyMesh/points"] + [
            case_path / "constant" / region / "polyMesh/points"
            for region in REQUIRED_MESH_REGIONS
        ]
        missing = [path for path in expected if not path.is_file()]
        if missing:
            raise OptimizationError(
                "Mesh command completed but required mesh files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )

    def _validate_scientific_endpoint(self, summary: ExperimentSummary) -> None:
        target_a_cm2 = self.config.target_current_density_a_m2 / 1.0e4
        relative_error = abs(
            summary.mean_current_density_magnitude_a_cm2 - target_a_cm2
        ) / target_a_cm2
        if relative_error > self.config.target_current_tolerance:
            raise OptimizationError(
                "Final-window current did not meet the fixed-current target: "
                f"relative error={relative_error:.6g}, "
                f"limit={self.config.target_current_tolerance:.6g}"
            )
        if summary.final_controller_accepted is False:
            raise OptimizationError(
                "The galvanostatic controller did not accept a stable endpoint by 20 s"
            )

    def _run_case(
        self, thickness_um: float, paths: dict[str, Path]
    ) -> tuple[list[ExperimentSample], ExperimentSummary] | None:
        started = time.monotonic()
        metadata: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "configuring",
            "thickness_um": thickness_um,
            "work_case": str(paths["work"]),
            "mesh_log": str(paths["mesh_log"]),
            "solver_log": str(paths["solver_log"]),
            "timeseries_csv": str(paths["data"]),
            "started_at_utc": _utc_now(),
        }
        _write_json(paths["metadata"], metadata)
        try:
            copy_clean_case(self.source_case, paths["work"])
            configure_case(paths["work"], thickness_um, self.config)
            if self.dry_run:
                metadata.update(
                    status="configured",
                    completed_at_utc=_utc_now(),
                    elapsed_s=time.monotonic() - started,
                )
                _write_json(paths["metadata"], metadata)
                return None

            metadata["status"] = "meshing"
            _write_json(paths["metadata"], metadata)
            run_command(self.mesh_command, paths["work"], paths["mesh_log"], self.timeout_s)
            self._verify_mesh(paths["work"])

            metadata["status"] = "solving"
            _write_json(paths["metadata"], metadata)
            solver_output = run_command(
                self.solver_command, paths["work"], paths["solver_log"], self.timeout_s
            )
            if not NORMAL_END_RE.search(solver_output):
                raise OptimizationError("Solver output does not contain the normal OpenFOAM 'End'")
            samples = parse_experiment_log(solver_output)
            validate_completed_run(samples, self.config.duration_s, self.config.delta_t_s)
            summary = summarize_samples(thickness_um, samples, self.config.final_window_s)
            self._validate_scientific_endpoint(summary)
            write_timeseries_csv(paths["data"], samples)
            metadata.update(
                status="completed",
                completed_at_utc=_utc_now(),
                elapsed_s=time.monotonic() - started,
                sample_count=len(samples),
                summary=asdict(summary),
            )
            _write_json(paths["metadata"], metadata)
            return samples, summary
        except Exception as exc:
            metadata.update(
                status="failed",
                completed_at_utc=_utc_now(),
                elapsed_s=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_json(paths["metadata"], metadata)
            raise

    def run(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_manifest()
        samples_by_thickness: dict[float, Sequence[ExperimentSample]] = {}
        summaries: list[ExperimentSummary] = []
        failures: list[tuple[float, str]] = []

        for thickness_um in self.config.thicknesses_um:
            paths = self._case_paths(thickness_um)
            existing = self._existing_complete(thickness_um, paths)
            if existing is not None:
                print(f"[resume] {thickness_um:g} µm already completed")
                samples, summary = existing
                samples_by_thickness[thickness_um] = samples
                summaries.append(summary)
                continue
            print(f"[run] {thickness_um:g} µm -> {paths['work']}")
            try:
                result = self._run_case(thickness_um, paths)
                if result is not None:
                    samples, summary = result
                    samples_by_thickness[thickness_um] = samples
                    summaries.append(summary)
                    print(
                        f"[done] {thickness_um:g} µm: "
                        f"crossover={summary.mean_crossover_rate_mol_s:.6g} mol/s, "
                        f"voltage={summary.mean_voltage_v:.6g} V"
                    )
                else:
                    print(f"[configured] {thickness_um:g} µm (dry run)")
            except Exception as exc:
                failures.append((thickness_um, str(exc)))
                print(f"[failed] {thickness_um:g} µm: {exc}")
                if self.fail_fast:
                    raise

        if summaries:
            write_reports(
                self.output_dir,
                samples_by_thickness,
                summaries,
                self.config.target_current_density_a_m2 / 1.0e4,
            )
            print(f"[report] {self.output_dir / 'summary.csv'}")
            print(f"[report] {self.output_dir / 'timeseries.png'}")
            print(f"[report] {self.output_dir / 'thickness_summary.png'}")
        if failures:
            print(f"{len(failures)} experiment(s) failed; rerun the same command to resume.")
            return 1
        return 0
