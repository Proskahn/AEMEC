from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]


class FixedVoltageDiagnosticTests(unittest.TestCase):
    def test_prepared_case_is_potentiostatic_and_passes_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixed-voltage"
            result = subprocess.run(
                [
                    sys.executable,
                    str(CASE / "prepare_fixed_voltage_diagnostic.py"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            controller = (
                output / "constant/phiEAnode/regionProperties"
            ).read_text(encoding="utf-8")
            self.assertRegex(controller, r"galvanostatic\s*\{\s*active\s+false\s*;")
            self.assertIn("value   1.5;", controller)
            self.assertIn("electrochemicalDiagnostics true;", (
                output / "constant/cathode/combustionProperties.gas"
            ).read_text(encoding="utf-8"))
            self.assertIn("electricDiagnostics true;", (
                output / "constant/phiAnion/regionProperties"
            ).read_text(encoding="utf-8"))

            preflight = subprocess.run(
                [str(output / "check_case.py"), "--static"],
                cwd=output,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(preflight.returncode, 0, preflight.stderr)

    def test_solver_sources_emit_the_required_diagnostics(self) -> None:
        reaction_source = (
            ROOT / "src/libSrc/fuelCellSystems/electroChemicalModel/electroChemicalReaction.C"
        ).read_text(encoding="utf-8")
        electric_source = (
            ROOT / "src/libSrc/fuelCellSystems/regions/electric/electric.C"
        ).read_text(encoding="utf-8")
        crossover_source = (
            ROOT
            / "src/libSrc/fuelCellSystems/hydrogenCrossoverModels/standardH2Crossover/standardH2Crossover.C"
        ).read_text(encoding="utf-8")

        self.assertIn("AEMEC reaction diagnostic:", reaction_source)
        self.assertIn("H2FaradaicRate", reaction_source)
        self.assertIn("AEMEC electric diagnostic:", electric_source)
        self.assertIn("Hydrogen crossover membrane diagnostic:", crossover_source)


if __name__ == "__main__":
    unittest.main()
