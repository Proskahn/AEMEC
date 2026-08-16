from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from polarization_curve.polarization_workflow.runner import (
    PolarizationCurveRunner,
    generate_visualizations,
)


ROOT = Path(__file__).resolve().parents[2]

DIAGNOSTIC_LOG = """
Time = 15
AEMEC reaction diagnostic: fluidRegion=anode, phase=gas, zone=anodeCL, reactionCurrent=0.08 A, H2FaradaicRate=0 mol/s, j0=120 A/m3, activationPower=0.016 W, equivalentActivationVoltage=0.2 V, currentWeightedNernst=-1.75 V, eta[min,mean,max]=(0.19,0.2,0.21) V, nernst[min,mean,max]=(-1.76,-1.75,-1.74) V, T[min,mean,max]=(313,313,313) K
AEMEC reaction diagnostic: fluidRegion=cathode, phase=gas, zone=cathodeCL, reactionCurrent=0.08 A, H2FaradaicRate=4.1457e-7 mol/s, j0=1.7e6 A/m3, activationPower=0.008 W, equivalentActivationVoltage=0.1 V, currentWeightedNernst=-2.95 V, eta[min,mean,max]=(-0.11,-0.1,-0.09) V, nernst[min,mean,max]=(-2.96,-2.95,-2.94) V, T[min,mean,max]=(313,313,313) K
AEMEC electric diagnostic: region=phiEAnode, sigma[min,mean,max]=(1,2,3), J[min,mean,max]=(0,0,1) A/m3, dJdPhi[min,mean,max]=(0,0,1) A/(m3 V), ohmicPower=0.004 W, reactionCurrentScale=0.08 A, equivalentOhmicVoltage=0.05 V
AEMEC electric diagnostic: region=phiECathode, sigma[min,mean,max]=(1,2,3), J[min,mean,max]=(0,0,1) A/m3, dJdPhi[min,mean,max]=(0,0,1) A/(m3 V), ohmicPower=0.0024 W, reactionCurrentScale=0.08 A, equivalentOhmicVoltage=0.03 V
AEMEC electric diagnostic: region=phiAnion, sigma[min,mean,max]=(1,2,3), J[min,mean,max]=(-1,0,1) A/m3, dJdPhi[min,mean,max]=(0,0,1) A/(m3 V), ohmicPower=0.008 W, reactionCurrentScale=0.08 A, equivalentOhmicVoltage=0.1 V
Hydrogen crossover objective: anode gas source rate = 4e-8 mol/s
Hydrogen production partition: anion reaction current in cathodeCL = -0.08 A, Faradaic cathode H2 generation = 4.1457e-7 mol/s, initially dissolved = 3e-7 mol/s, direct Faradaic gas = 1.1457e-7 mol/s, cathode dissolved-to-gas transfer = 2.6e-7 mol/s, anode dissolved-to-gas transfer = 4e-8 mol/s, dissolved inventory = 2e-9 mol
Controlled boundary current (A) at interconnect0: signed = -0.08, magnitude = 0.08, current density = -1000 A/m2, voltage = 1.70
End
"""


class PolarizationCurveRunnerTests(unittest.TestCase):
    def test_dry_run_restores_the_isolated_165_second_voltage_scan(self) -> None:
        source = ROOT / "run" / "AEMEC"
        original_controller = (
            source / "constant/phiEAnode/regionProperties"
        ).read_text(encoding="utf-8")
        original_control = (source / "system/controlDict.run").read_text(
            encoding="utf-8"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = PolarizationCurveRunner(
                source_case=source,
                work_dir=root / "work",
                output_dir=root / "results",
                mesh_command=("make", "mesh"),
                solver_command=("openFuelCell",),
                dry_run=True,
            )
            self.assertEqual(runner.run(), 0)
            work = root / "work/case"
            controller = (
                work / "constant/phiEAnode/regionProperties"
            ).read_text(encoding="utf-8")
            control = (work / "system/controlDict.run").read_text(
                encoding="utf-8"
            )
            manifest = json.loads(
                (root / "results/run.json").read_text(encoding="utf-8")
            )

        self.assertIn("galvanostatic\n{\n    active      false;", controller)
        self.assertIn("active                      false;", controller)
        self.assertIn("(165     2.3)", controller)
        self.assertIn("endTime         165;", control)
        self.assertIn("writeInterval   165;", control)
        self.assertEqual(manifest["status"], "configured")
        self.assertEqual(manifest["duration_s"], 165.0)
        self.assertEqual(len(manifest["voltage_holds"]), 11)
        self.assertEqual(
            original_controller,
            (source / "constant/phiEAnode/regionProperties").read_text(
                encoding="utf-8"
            ),
        )
        self.assertEqual(
            original_control,
            (source / "system/controlDict.run").read_text(encoding="utf-8"),
        )

    def test_all_visualization_products_are_generated_from_one_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "solver.log"
            log.write_text(DIAGNOSTIC_LOG, encoding="utf-8")
            products = generate_visualizations(
                log, root / "data", root / "figures"
            )

            self.assertEqual(products["polarization_point_count"], 1)
            self.assertEqual(products["voltage_point_count"], 1)
            self.assertEqual(products["crossover_point_count"], 1)
            for name in (
                "polarization_curve.csv",
                "voltage_decomposition.csv",
                "hydrogen_crossover.csv",
            ):
                self.assertTrue((root / "data" / name).is_file())
            for name in (
                "polarization_curve.png",
                "voltage_decomposition.png",
                "hydrogen_crossover.png",
            ):
                path = root / "figures" / name
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
