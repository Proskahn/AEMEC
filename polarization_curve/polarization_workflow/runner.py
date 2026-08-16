"""Run and post-process the retained 165 s AEMEC voltage sweep."""

from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# Importing project first adds the repository and application-package roots to
# sys.path for direct execution through polarization_curve/run_polarization_curve.py.
from .project import (
    AemecEvaluationConfig,
    NORMAL_END_RE,
    OptimizationError,
    case_fingerprint,
    configure_voltage_sweep,
    copy_clean_case,
    run_command,
    validate_path_layout,
)

from visualization.aemec_diagnostics import (
    build_crossover_points,
    build_voltage_points,
    parse_diagnostics,
    plot_hydrogen_crossover,
    plot_voltage_decomposition,
    write_crossover_csv,
    write_voltage_csv,
)
from visualization.polarization_curve import (
    DEFAULT_ACTIVE_AREA_CM2,
    last_sample_per_voltage,
    log_ended_normally,
    parse_log,
    plot_curve,
    write_csv,
)


SCHEMA_VERSION = 1
RUN_DESCRIPTION = "AEMEC former 165 s potentiostatic polarization curve"
WORK_MARKER = ".aemec-polarization-work-case"
EXPECTED_DURATION_S = 165.0
EXPECTED_HOLD_DURATION_S = 15.0
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def generate_visualizations(
    log_path: Path,
    data_dir: Path,
    figures_dir: Path,
    active_area_cm2: float = DEFAULT_ACTIVE_AREA_CM2,
) -> dict[str, object]:
    """Create the curve, voltage-loss, and crossover products from one log."""
    log_path = log_path.resolve()
    if not log_path.is_file():
        raise OptimizationError(f"Solver log not found: {log_path}")
    if active_area_cm2 <= 0.0 or not math.isfinite(active_area_cm2):
        raise OptimizationError("Active area must be a finite positive value")
    if not log_ended_normally(log_path):
        raise OptimizationError(
            "Solver log does not contain the normal OpenFOAM 'End' marker"
        )

    samples = parse_log(log_path, active_area_cm2)
    if not samples:
        raise OptimizationError(
            f"No collector current/voltage samples found in: {log_path}"
        )
    selected = last_sample_per_voltage(samples, voltage_precision=6)
    if not selected:
        raise OptimizationError("No voltage-hold endpoints could be selected")

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = data_dir / "polarization_curve.csv"
    curve_png = figures_dir / "polarization_curve.png"
    write_csv(selected, curve_csv)
    plot_curve(selected, curve_png, use_signed=False)

    diagnostics = parse_diagnostics(log_path)
    voltage_points, missing_voltage = build_voltage_points(selected, diagnostics)
    crossover_points, missing_crossover = build_crossover_points(
        selected, diagnostics
    )
    generated: dict[str, object] = {
        "raw_sample_count": len(samples),
        "polarization_point_count": len(selected),
        "polarization_curve_csv": str(curve_csv),
        "polarization_curve_plot": str(curve_png),
        "warnings": [],
    }
    warnings = generated["warnings"]
    assert isinstance(warnings, list)

    if voltage_points:
        voltage_csv = data_dir / "voltage_decomposition.csv"
        voltage_png = figures_dir / "voltage_decomposition.png"
        write_voltage_csv(voltage_points, voltage_csv)
        plot_voltage_decomposition(voltage_points, voltage_png)
        generated.update(
            voltage_point_count=len(voltage_points),
            voltage_decomposition_csv=str(voltage_csv),
            voltage_decomposition_plot=str(voltage_png),
        )
    else:
        warnings.append(
            "No complete voltage-decomposition diagnostics matched the selected holds"
        )

    if crossover_points:
        crossover_csv = data_dir / "hydrogen_crossover.csv"
        crossover_png = figures_dir / "hydrogen_crossover.png"
        write_crossover_csv(crossover_points, crossover_csv)
        plot_hydrogen_crossover(crossover_points, crossover_png)
        generated.update(
            crossover_point_count=len(crossover_points),
            hydrogen_crossover_csv=str(crossover_csv),
            hydrogen_crossover_plot=str(crossover_png),
        )
    else:
        warnings.append(
            "No hydrogen-crossover diagnostics matched the selected holds"
        )

    if voltage_points and missing_voltage:
        warnings.append(
            f"Voltage diagnostics were incomplete at "
            f"{len(set(missing_voltage))} selected holds"
        )
    if crossover_points and missing_crossover:
        warnings.append(
            f"Crossover diagnostics were incomplete at "
            f"{len(set(missing_crossover))} selected holds"
        )
    return generated


class PolarizationCurveRunner:
    """Prepare an isolated case, run the old voltage table, and plot it."""

    def __init__(
        self,
        source_case: Path,
        work_dir: Path,
        output_dir: Path,
        mesh_command: Sequence[str],
        solver_command: Sequence[str],
        timeout_s: float | None = None,
        active_area_cm2: float = DEFAULT_ACTIVE_AREA_CM2,
        overwrite: bool = False,
        dry_run: bool = False,
        postprocess_only: bool = False,
    ) -> None:
        self.source_case = source_case.resolve()
        self.work_dir = work_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.work_case = self.work_dir / "case"
        self.mesh_command = tuple(mesh_command)
        self.solver_command = tuple(solver_command)
        self.timeout_s = timeout_s
        self.active_area_cm2 = active_area_cm2
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.postprocess_only = postprocess_only

        if not self.postprocess_only and not self.source_case.is_dir():
            raise OptimizationError(
                f"OpenFOAM source case not found: {self.source_case}"
            )
        if not self.mesh_command or not self.solver_command:
            raise OptimizationError("Mesh and solver commands cannot be empty")
        if timeout_s is not None and timeout_s <= 0.0:
            raise OptimizationError("Timeout must be positive")
        if active_area_cm2 <= 0.0 or not math.isfinite(active_area_cm2):
            raise OptimizationError("Active area must be a finite positive value")
        if dry_run and postprocess_only:
            raise OptimizationError("--dry-run and --postprocess-only are incompatible")
        validate_path_layout(self.source_case, self.work_case, self.output_dir)

    @property
    def paths(self) -> dict[str, Path]:
        return {
            "manifest": self.output_dir / "run.json",
            "mesh_log": self.output_dir / "logs" / "mesh.log",
            "solver_log": self.output_dir / "logs" / "solver.log",
            "data": self.output_dir / "data",
            "figures": self.output_dir / "figures",
        }

    def _new_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "description": RUN_DESCRIPTION,
            "status": "preparing",
            "started_at_utc": _utc_now(),
            "source_case": str(self.source_case),
            "source_case_fingerprint": case_fingerprint(self.source_case),
            "work_case": str(self.work_case),
            "mesh_command": list(self.mesh_command),
            "solver_command": list(self.solver_command),
            "active_area_cm2": self.active_area_cm2,
        }

    def _prepare_output(self) -> None:
        if self.output_dir.exists():
            if not self.overwrite:
                raise OptimizationError(
                    f"Output directory already exists: {self.output_dir}. "
                    "Use --overwrite for a fresh run or --postprocess-only "
                    "to regenerate figures from its solver log."
                )
            entries = list(self.output_dir.iterdir())
            if entries:
                manifest_path = self.paths["manifest"]
                try:
                    previous = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise OptimizationError(
                        "Refusing to overwrite an output directory that is not "
                        "owned by this polarization workflow"
                    ) from exc
                if previous.get("description") != RUN_DESCRIPTION:
                    raise OptimizationError(
                        "Refusing to overwrite an output directory that is not "
                        "owned by this polarization workflow"
                    )
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def _prepare_work_case(self) -> None:
        marker = self.work_case / WORK_MARKER
        if self.work_case.exists() and not marker.is_file():
            raise OptimizationError(
                "Refusing to replace an unmarked work-case directory: "
                f"{self.work_case}"
            )
        copy_clean_case(self.source_case, self.work_case)
        marker.write_text(
            "Owned by polarization_curve/run_polarization_curve.py\n",
            encoding="utf-8",
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

    def _postprocess(self) -> dict[str, object]:
        products = generate_visualizations(
            self.paths["solver_log"],
            self.paths["data"],
            self.paths["figures"],
            self.active_area_cm2,
        )
        print(f"[plot] {products['polarization_curve_plot']}")
        return products

    def _run_postprocess_only(self) -> int:
        products = self._postprocess()
        manifest_path = self.paths["manifest"]
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {"schema_version": SCHEMA_VERSION}
        else:
            manifest = {"schema_version": SCHEMA_VERSION}
        manifest.update(
            status="completed",
            postprocessed_at_utc=_utc_now(),
            products=products,
        )
        _write_json(manifest_path, manifest)
        return 0

    def run(self) -> int:
        if self.postprocess_only:
            return self._run_postprocess_only()

        self._prepare_output()
        manifest = self._new_manifest()
        _write_json(self.paths["manifest"], manifest)
        started = time.monotonic()
        try:
            self._prepare_work_case()
            sweep = configure_voltage_sweep(
                self.work_case, AemecEvaluationConfig()
            )
            if not math.isclose(
                sweep.holds[-1].end_s,
                EXPECTED_DURATION_S,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise OptimizationError(
                    "The retained voltage table no longer ends at 165 s"
                )
            if any(
                not math.isclose(
                    hold.end_s - hold.start_s,
                    EXPECTED_HOLD_DURATION_S,
                    rel_tol=0.0,
                    abs_tol=0.01,
                )
                for hold in sweep.holds
            ):
                raise OptimizationError(
                    "The retained voltage table no longer consists of 15 s holds"
                )
            manifest.update(
                status="configured" if self.dry_run else "meshing",
                voltage_holds=[
                    {
                        "start_s": hold.start_s,
                        "end_s": hold.end_s,
                        "voltage_v": hold.voltage_v,
                    }
                    for hold in sweep.holds
                ],
                duration_s=sweep.holds[-1].end_s,
                delta_t_s=sweep.delta_t_s,
                outer_iterations=sweep.outer_iterations,
            )
            if self.dry_run:
                manifest.update(
                    completed_at_utc=_utc_now(),
                    elapsed_s=time.monotonic() - started,
                )
                _write_json(self.paths["manifest"], manifest)
                print(f"[configured] {self.work_case}")
                return 0

            _write_json(self.paths["manifest"], manifest)
            print(f"[mesh] {self.work_case}")
            run_command(
                self.mesh_command,
                self.work_case,
                self.paths["mesh_log"],
                self.timeout_s,
            )
            self._verify_mesh()

            manifest["status"] = "solving"
            _write_json(self.paths["manifest"], manifest)
            print(f"[solve] 165 s voltage sweep -> {self.paths['solver_log']}")
            output = run_command(
                self.solver_command,
                self.work_case,
                self.paths["solver_log"],
                self.timeout_s,
            )
            if not NORMAL_END_RE.search(output):
                raise OptimizationError(
                    "Solver output does not contain the normal OpenFOAM 'End'"
                )

            manifest["status"] = "postprocessing"
            _write_json(self.paths["manifest"], manifest)
            products = self._postprocess()
            manifest.update(
                status="completed",
                completed_at_utc=_utc_now(),
                elapsed_s=time.monotonic() - started,
                products=products,
            )
            _write_json(self.paths["manifest"], manifest)
            print(f"[done] {self.output_dir}")
            return 0
        except Exception as exc:
            manifest.update(
                status="failed",
                completed_at_utc=_utc_now(),
                elapsed_s=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_json(self.paths["manifest"], manifest)
            raise
