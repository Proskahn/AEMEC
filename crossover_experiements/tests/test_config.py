from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from crossover_experiments.config import ExperimentConfig, configure_case
from crossover_experiments.project import DEFAULT_SOURCE_CASE, copy_clean_case


class ExperimentConfigurationTests(unittest.TestCase):
    def test_configures_thickness_current_duration_and_initial_voltage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case_path = Path(temporary) / "case"
            copy_clean_case(DEFAULT_SOURCE_CASE, case_path)
            configure_case(case_path, 40.0, ExperimentConfig(thicknesses_um=(40.0,)))

            mesh_text = (case_path / "system/blockMeshDict").read_text(encoding="utf-8")
            vertices = mesh_text[mesh_text.index("vertices"):mesh_text.index("blocks")]
            z_levels = sorted(
                {
                    float(match.group(3))
                    for match in re.finditer(
                        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
                        vertices,
                    )
                }
            )
            membrane_thickness_mm = min(z for z in z_levels if z > 0) - max(
                z for z in z_levels if z < 0
            )
            self.assertAlmostEqual(membrane_thickness_mm, 0.04)

            region_text = (
                case_path / "constant/phiEAnode/regionProperties"
            ).read_text(encoding="utf-8")
            self.assertRegex(region_text, r"(?s)ibar\s*\{.*?\(0\s+-10000\).*?\(20\s+-10000\)")
            self.assertRegex(region_text, r"targets\s+\(-10000\)\s*;")
            self.assertRegex(region_text, r"minimumHoldDuration\s+20\s*;")

            control_text = (case_path / "system/controlDict.run").read_text(encoding="utf-8")
            self.assertRegex(control_text, r"(?m)^endTime\s+20\s*;")
            self.assertRegex(control_text, r"(?m)^deltaT\s+0\.1\s*;")
            initial_text = (case_path / "0.orig/phiEAnode/phi").read_text(encoding="utf-8")
            self.assertRegex(initial_text, r"internalField\s+uniform\s+2\s*;")

    def test_rejects_duplicate_thicknesses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unique"):
            ExperimentConfig(thicknesses_um=(20.0, 20.0)).validate()


if __name__ == "__main__":
    unittest.main()

