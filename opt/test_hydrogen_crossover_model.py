from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HydrogenCrossoverSourceTests(unittest.TestCase):
    def test_faradaic_generation_is_not_a_membrane_source(self) -> None:
        source = (
            ROOT
            / "src/libSrc/fuelCellSystems/hydrogenCrossoverModels"
            / "standardH2Crossover/standardH2Crossover.C"
        ).read_text(encoding="utf-8")

        self.assertIn("volScalarField h2FaradaicGeneration", source)
        self.assertIn(
            "zoneIntegral(h2FaradaicGeneration, sourceZoneName_)", source
        )
        self.assertNotIn("setZoneSource(sourceZoneName_", source)
        self.assertNotIn("- h2Dmdt_", source)
        self.assertIn(
            "setZoneUniformSource(h2CathodeDmdt_, sourceZoneName_, "
            "-h2CrossoverRate)",
            source,
        )
        self.assertIn(
            "setZoneUniformSource(h2AnodeDmdt_, sinkZoneName_, "
            "h2CrossoverRate)",
            source,
        )

    def test_drag_diagnostic_uses_the_transported_concentration(self) -> None:
        source = (
            ROOT
            / "src/libSrc/fuelCellSystems/hydrogenCrossoverModels"
            / "standardH2Crossover/standardH2Crossover.C"
        ).read_text(encoding="utf-8")

        self.assertIn("JH2Drag_ = mag(i)*xi_*cH2_/(F*cElec_);", source)
        self.assertNotIn(
            "JH2Drag_ = mag(i)*xi_*cH2CathodeInterface_/(F*cElec_);",
            source,
        )


if __name__ == "__main__":
    unittest.main()
