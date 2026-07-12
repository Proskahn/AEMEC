from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aemec_case import (
    AemecEvaluationConfig,
    AemecOpenFoamEvaluator,
    OptimizationError,
    configure_target_hold,
    copy_clean_case,
    parse_objective_samples,
    rewrite_block_mesh_thickness,
    select_target_sample,
)


ROOT = Path(__file__).resolve().parents[1]


def copy_controls(case: Path) -> None:
    (case / "constant/phiEAnode").mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(exist_ok=True)
    shutil.copy2(
        ROOT / "run/AEMEC/constant/phiEAnode/regionProperties",
        case / "constant/phiEAnode/regionProperties",
    )
    for name in ("controlDict.run", "controlDict"):
        shutil.copy2(ROOT / "run/AEMEC/system" / name, case / "system" / name)


class AemecCaseTests(unittest.TestCase):
    def test_static_case_preflight_passes(self) -> None:
        result = subprocess.run(
            [str(ROOT / "run/AEMEC/check_case.py"), "--static"],
            cwd=ROOT / "run/AEMEC",
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_case_uses_the_cathode_fed_aem_region_contract(self) -> None:
        case = ROOT / "run/AEMEC"
        regions = (case / "constant/regionProperties").read_text(encoding="utf-8")
        cathode = (case / "constant/cathode/regionProperties").read_text(encoding="utf-8")
        anode = (case / "constant/anode/regionProperties").read_text(encoding="utf-8")
        crossover = (case / "constant/phiAnion/regionProperties").read_text(encoding="utf-8")
        cathode_reaction = (case / "constant/cathode/combustionProperties").read_text(encoding="utf-8")
        anode_reaction = (case / "constant/anode/combustionProperties.oxygen").read_text(encoding="utf-8")
        cathode_diffusivity = (case / "constant/cathode/diffusivityModel.hydrogen").read_text(encoding="utf-8")
        anode_diffusivity = (case / "constant/anode/diffusivityModel.oxygen").read_text(encoding="utf-8")
        cathode_porous = (case / "constant/cathode/porousZones").read_text(encoding="utf-8")
        anode_porous = (case / "constant/anode/porousZones").read_text(encoding="utf-8")
        cathode_water_velocity = (case / "0.orig/cathode/U.water").read_text(encoding="utf-8")
        anode_water_velocity = (case / "0.orig/anode/U.water").read_text(encoding="utf-8")
        anode_thermo = (case / "constant/anode/thermophysicalProperties.oxygen").read_text(encoding="utf-8")
        anode_hydrogen = (case / "0.orig/anode/H2.oxygen").read_text(encoding="utf-8")
        anode_controller = (case / "constant/phiEAnode/regionProperties").read_text(encoding="utf-8")
        electrolyte_temperature = (case / "0.orig/electrolyte/T").read_text(encoding="utf-8")
        interconnect_temperature = (case / "0.orig/interconnect/T").read_text(encoding="utf-8")

        self.assertIn("fluid (anode cathode)", regions)
        self.assertIn("electric (phiECathode phiEAnode phiAnion)", regions)
        self.assertIn("phases (hydrogen water)", cathode)
        self.assertIn("phases (oxygen water)", anode)
        self.assertIn("sourceZone      cathodeCL", crossover)
        self.assertIn("sinkZone        anodeCL", crossover)
        self.assertIn("H2O    -1", cathode_reaction)
        self.assertIn("H2O     1", anode_reaction)
        self.assertIn("cathodeChannel\n{", cathode_diffusivity)
        self.assertIn("anodeChannel\n{", anode_diffusivity)
        self.assertIn("cellZone        cathodeMPL", cathode_porous)
        self.assertIn("cellZone        anodeMPL", anode_porous)
        self.assertIn("uniform (0.01 0 0)", cathode_water_velocity)
        self.assertIn("uniform (0.01 0 0)", anode_water_velocity)
        self.assertIn("H2", anode_thermo)
        self.assertIn("object          yH2;", anode_hydrogen)
        self.assertIn("cathodeFluidRegion  cathode;", crossover)
        self.assertIn("anodeFluidRegion    anode;", crossover)
        self.assertIn("diffusivityModel    porosityTortuosity;", crossover)
        self.assertIn("cathodeInterface", crossover)
        self.assertIn("anodeInterface", crossover)
        self.assertIn("UMembrane       (0 0 0);", crossover)
        self.assertIn("polarizationCurve", anode_controller)
        self.assertIn("minimumHoldDuration         30;", anode_controller)
        self.assertIn("stabilitySamples            5;", anode_controller)
        self.assertIn("electrolyte_to_anode", electrolyte_temperature)
        self.assertIn("electrolyte_to_cathode", electrolyte_temperature)
        self.assertIn("interconnect_to_anode", interconnect_temperature)
        self.assertIn("interconnect_to_cathode", interconnect_temperature)

    def test_thickness_rewrite_preserves_other_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "blockMeshDict"
            shutil.copy2(ROOT / "run/AEMEC/system/blockMeshDict", mesh)
            update = rewrite_block_mesh_thickness(mesh, 50.0)
            self.assertEqual(update.old_thickness_um, 30.0)
            self.assertEqual(update.new_thickness_um, 50.0)
            self.assertIn("0.025", mesh.read_text(encoding="utf-8"))

    def test_fast_mode_configures_direct_target_and_final_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            copy_controls(case)
            hold = configure_target_hold(case, AemecEvaluationConfig())
            self.assertEqual(hold.outer_iterations, 250)
            self.assertEqual(hold.delta_t_s, 1.0)
            controls = (case / "system/controlDict.run").read_text(encoding="utf-8")
            controller = (case / "constant/phiEAnode/regionProperties").read_text(encoding="utf-8")
            self.assertRegex(controls, r"endTime\s+250\s*;")
            self.assertRegex(controls, r"writeInterval\s+250\s*;")
            self.assertRegex(
                controller,
                r"polarizationCurve\s*\{[\s\S]*?active\s+false\s*;",
            )

    def test_parser_uses_post_solve_boundary_current_and_final_window(self) -> None:
        config = AemecEvaluationConfig(stability_samples=3)
        log = """
Time = 1
galvanostatic target: -10000 A/m2, raw dV: 0.0001, limited dV: 0.0001
ibar: -8000 voltage: 1.7
Controlled boundary current (A) at x: signed = -0.8, magnitude = 0.8, current density = -10000 A/m2, voltage = 1.8
Hydrogen crossover objective: anode gas source rate = 2.0e-5 mol/s
Time = 2
galvanostatic target: -10000 A/m2, raw dV: 0.0001, limited dV: 0.0001
Controlled boundary current (A) at x: signed = -0.8, magnitude = 0.8, current density = -10005 A/m2, voltage = 1.801
Hydrogen crossover objective: anode gas source rate = 2.01e-5 mol/s
Time = 3
galvanostatic target: -10000 A/m2, raw dV: 0.0001, limited dV: 0.0001
Controlled boundary current (A) at x: signed = -0.8, magnitude = 0.8, current density = -10010 A/m2, voltage = 1.8
Hydrogen crossover objective: anode gas source rate = 2.0e-5 mol/s
End
"""
        samples = parse_objective_samples(log)
        self.assertEqual(samples[0].actual_current_density_a_m2, -10000.0)
        hold = type("Hold", (), {"end_s": 3.0, "delta_t_s": 1.0})()
        selected = select_target_sample(log, config, hold)
        self.assertEqual(selected.time_s, 3.0)

    def test_copy_refuses_to_delete_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "case"
            source.mkdir()
            marker = source / "keep"
            marker.write_text("safe", encoding="utf-8")
            with self.assertRaises(OptimizationError):
                copy_clean_case(source, source)
            self.assertEqual(marker.read_text(encoding="utf-8"), "safe")

    def test_mocked_evaluator_runs_the_aemec_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            copy_controls(source)
            shutil.copy2(ROOT / "run/AEMEC/system/blockMeshDict", source / "system/blockMeshDict")
            (source / "mesh.py").write_text(
                """from pathlib import Path
for region in ('', 'anode', 'cathode', 'electrolyte', 'interconnect', 'phiECathode', 'phiEAnode', 'phiAnion'):
    base = Path('constant') if not region else Path('constant') / region
    target = base / 'polyMesh/points'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('mesh')
""",
                encoding="utf-8",
            )
            (source / "solver.py").write_text(
                """for moment in (246, 247, 248, 249, 250):
    print(f'Time = {moment}')
    print('galvanostatic target: -10000 A/m2, raw dV: 0.0001, limited dV: 0.0001')
    print('Controlled boundary current (A) at x: signed = -0.8, magnitude = 0.8, current density = -10000 A/m2, voltage = 1.8')
    print('Hydrogen crossover objective: anode release rate = 2e-5 mol/s')
print('End')
""",
                encoding="utf-8",
            )
            evaluator = AemecOpenFoamEvaluator(source, root / "work/case", root / "logs", (sys.executable, "mesh.py"), (sys.executable, "solver.py"), AemecEvaluationConfig(), 10.0)
            result = evaluator.evaluate(0, 40.0)
            self.assertEqual(result.sample.cell_voltage_v, 1.8)
            self.assertTrue(result.solver_log.is_file())


if __name__ == "__main__":
    unittest.main()
