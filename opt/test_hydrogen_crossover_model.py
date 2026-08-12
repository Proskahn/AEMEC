from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HydrogenCrossoverSourceTests(unittest.TestCase):
    def test_faradaic_generation_is_partitioned_into_dissolved_field(self) -> None:
        source = (
            ROOT
            / "src/libSrc/fuelCellSystems/hydrogenCrossoverModels"
            / "standardH2Crossover/standardH2Crossover.C"
        ).read_text(encoding="utf-8")

        self.assertIn("volScalarField h2FaradaicGeneration", source)
        self.assertIn(
            "faradaicDissolvedFraction_*h2FaradaicGeneration[cell]", source
        )
        self.assertIn(
            "h2DissolvedToGas_[cell] - h2DissolvedProduction_[cell]",
            source,
        )
        self.assertIn("h2AnodeDmdt_[cell] = h2DissolvedToGas_[cell]", source)

    def test_dissolved_equation_has_storage_and_signed_cl_transfer(self) -> None:
        source = (
            ROOT
            / "src/libSrc/fuelCellSystems/hydrogenCrossoverModels"
            / "standardH2Crossover/standardH2Crossover.C"
        ).read_text(encoding="utf-8")

        self.assertIn("fvm::ddt(epsilonIon_, cH2_)", source)
        self.assertIn("fvm::Sp(h2MassTransferCoeff_, cH2_)", source)
        self.assertIn("h2DissolvedProduction_", source)
        self.assertIn("cH2CathodeInterface_ + cH2AnodeInterface_", source)
        self.assertIn("cH2_\n          - cH2CathodeInterface_", source)
        self.assertNotIn("h2Eqn->setReference", source)
        self.assertNotIn("cH2_.max", source)

    def test_coupling_diagnostic_includes_dissolved_inventory(self) -> None:
        source = (
            ROOT
            / "src/libSrc/fuelCellSystems/hydrogenCrossoverModels"
            / "standardH2Crossover/standardH2Crossover.C"
        ).read_text(encoding="utf-8")

        self.assertIn("cathodeGasRate + anodeGasRate + dissolvedCouplingRate", source)
        self.assertIn("epsilonIon_*cH2_", source)
        self.assertIn("Hydrogen dissolved-gas coupling conservation", source)

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
