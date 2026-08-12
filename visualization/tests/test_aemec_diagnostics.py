from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from visualization.aemec_diagnostics import (
    build_crossover_points,
    build_voltage_points,
    parse_diagnostics,
    plot_hydrogen_crossover,
    plot_voltage_decomposition,
)
from visualization.polarization_curve import Sample


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


class AemecDiagnosticPlotTests(unittest.TestCase):
    def test_voltage_and_crossover_diagnostics_are_decomposed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root/"log.run"
            log_path.write_text(DIAGNOSTIC_LOG, encoding="utf-8")
            diagnostics = parse_diagnostics(log_path)
            sample = Sample(
                time=15.0,
                voltage_v=1.70,
                current_a=-0.08,
                current_density_a_m2=-1000.0,
                target_current_density_a_m2=None,
                source="boundary-current-density",
            )

            voltage_points, voltage_missing = build_voltage_points(
                [sample], diagnostics
            )
            crossover_points, crossover_missing = build_crossover_points(
                [sample], diagnostics
            )

            self.assertFalse(voltage_missing)
            self.assertFalse(crossover_missing)
            self.assertEqual(len(voltage_points), 1)
            self.assertAlmostEqual(voltage_points[0].reversible_voltage_v, 1.2)
            self.assertAlmostEqual(voltage_points[0].electronic_ohmic_v, 0.08)
            self.assertAlmostEqual(voltage_points[0].anion_ohmic_v, 0.1)
            self.assertAlmostEqual(voltage_points[0].unresolved_v, 0.02)
            self.assertEqual(len(crossover_points), 1)
            self.assertAlmostEqual(
                crossover_points[0].crossover_fraction_percent,
                100.0*4e-8/4.1457e-7,
            )
            self.assertAlmostEqual(
                crossover_points[0].cathode_gas_release_rate_mol_s,
                3.7457e-7,
            )
            self.assertAlmostEqual(
                crossover_points[0].dissolved_production_rate_mol_s,
                3e-7,
            )
            self.assertAlmostEqual(
                crossover_points[0].anode_transfer_rate_mol_s,
                4e-8,
            )

            voltage_plot = root/"voltage.png"
            crossover_plot = root/"crossover.png"
            plot_voltage_decomposition(voltage_points, voltage_plot)
            plot_hydrogen_crossover(crossover_points, crossover_plot)
            self.assertGreater(voltage_plot.stat().st_size, 1000)
            self.assertGreater(crossover_plot.stat().st_size, 1000)

    def test_legacy_crossover_log_uses_membrane_diagnostic(self) -> None:
        legacy_log = """
Time = 1
Hydrogen crossover objective: anode gas source rate = 1e-8 mol/s
Hydrogen crossover membrane diagnostic: anion reaction current in cathodeCL = -0.2 A, derived Faradaic H2 source = 1e-6 mol/s, retained/released near anodeCL = 1e-9 mol
"""
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory)/"log.run"
            log_path.write_text(legacy_log, encoding="utf-8")
            diagnostics = parse_diagnostics(log_path)

        self.assertEqual(diagnostics[1.0].crossover_rate_mol_s, 1e-8)
        self.assertEqual(diagnostics[1.0].faradaic_h2_rate_mol_s, 1e-6)

    def test_crossover_fraction_is_undefined_above_instantaneous_production(
        self,
    ) -> None:
        log = """
Time = 1
Hydrogen crossover objective: anode gas source rate = 2e-8 mol/s
Hydrogen crossover membrane diagnostic: anion reaction current in cathodeCL = -0.001 A, derived Faradaic H2 source = 1e-8 mol/s, retained/released near anodeCL = 1e-9 mol
"""
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory)/"log.run"
            log_path.write_text(log, encoding="utf-8")
            diagnostics = parse_diagnostics(log_path)
            sample = Sample(
                time=1.0,
                voltage_v=1.3,
                current_a=-0.001,
                current_density_a_m2=-12.5,
                target_current_density_a_m2=None,
                source="boundary-current-density",
            )
            points, missing = build_crossover_points([sample], diagnostics)

        self.assertFalse(missing)
        self.assertTrue(math.isnan(points[0].crossover_fraction_percent))


if __name__ == "__main__":
    unittest.main()
