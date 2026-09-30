from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aemec_opt.case import (
    OptimizationError,
    _named_block_span,
    case_fingerprint,
    configure_ptl_porosity,
    copy_clean_case,
)


ROOT = Path(__file__).resolve().parents[2]


def zone_block(text: str, zone: str) -> str:
    start, end = _named_block_span(text, zone)
    return text[start:end]


def porosity(text: str) -> float:
    return float(re.search(r"(?m)^\s*porosity\s+([^;]+);", text).group(1))


def darcy(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in re.search(r"\bd\s+\(([^)]+)\)", text).group(1).split())


class PtlPorosityTests(unittest.TestCase):
    def test_updates_only_selected_gdl_properties_for_each_side(self) -> None:
        for side in ("anode", "cathode", "both"):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as directory:
                case = Path(directory) / "case"
                copy_clean_case(ROOT / "run/AEMEC", case)
                originals = {p: p.read_bytes() for p in case.rglob("*") if p.is_file()}
                selected = ("anode", "cathode") if side == "both" else (side,)
                targets = {}
                for region in selected:
                    electrical = "phiEAnode" if region == "anode" else "phiECathode"
                    for relative in (f"{region}/porousZones", f"{region}/diffusivityModel.gas", f"{electrical}/regionProperties"):
                        targets[case / "constant" / relative] = f"{region}GDL"

                for epsilon in (0.4, 0.8, 0.7):
                    configure_ptl_porosity(case, epsilon, side)
                    for path, original_bytes in originals.items():
                        if path not in targets:
                            self.assertEqual(path.read_bytes(), original_bytes, str(path))
                            continue
                        original = original_bytes.decode()
                        changed = path.read_text()
                        zone = targets[path]
                        old_start, old_end = _named_block_span(original, zone)
                        start, end = _named_block_span(changed, zone)
                        # Every byte outside the chosen GDL block is preserved.
                        self.assertEqual(changed[:start], original[:old_start])
                        self.assertEqual(changed[end:], original[old_end:])
                        block = changed[start:end]
                        original_block = original[old_start:old_end]
                        self.assertEqual(porosity(block), epsilon)
                        if path.name == "porousZones":
                            scale = ((1 - epsilon) / 0.3) ** 2 * (0.7 / epsilon) ** 3
                            for actual, reference in zip(darcy(block), darcy(original_block)):
                                self.assertAlmostEqual(actual / reference, scale, places=9)
                            # No contact angle, surface tension, heat-property,
                            # inertial-coefficient or coordinate-system edits.
                            block = re.sub(r"\bd\s+\([^)]+\)", "d VECTOR", block)
                            original_block = re.sub(r"\bd\s+\([^)]+\)", "d VECTOR", original_block)
                        self.assertEqual(
                            re.sub(r"(?m)^\s*porosity\s+[^;]+;", "porosity EPS;", block),
                            re.sub(r"(?m)^\s*porosity\s+[^;]+;", "porosity EPS;", original_block),
                        )
                    before = case_fingerprint(case)
                    configure_ptl_porosity(case, epsilon, side)
                    self.assertEqual(case_fingerprint(case), before)
                preflight = subprocess.run(
                    [sys.executable, str(case / "check_case.py"), "--static"],
                    cwd=case, capture_output=True, text=True,
                )
                self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)

    def test_permeability_increases_and_solid_conductivity_decreases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            copy_clean_case(ROOT / "run/AEMEC", case)
            permeabilities, conductivities = [], []
            for epsilon in (0.4, 0.7, 0.8):
                configure_ptl_porosity(case, epsilon)
                flow = zone_block((case / "constant/anode/porousZones").read_text(), "anodeGDL")
                solid = zone_block((case / "constant/phiEAnode/regionProperties").read_text(), "anodeGDL")
                permeabilities.append(1 / darcy(flow)[0])
                sigma = float(re.search(r"\bsigma\s+([^;]+);", solid).group(1))
                conductivities.append(sigma * (1 - porosity(solid)) ** 1.5)
            self.assertEqual(permeabilities, sorted(permeabilities))
            self.assertEqual(conductivities, sorted(conductivities, reverse=True))
            self.assertAlmostEqual(permeabilities[1] / 1e-11, 1.0)

    def test_preserves_anisotropy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            copy_clean_case(ROOT / "run/AEMEC", case)
            path = case / "constant/anode/porousZones"
            path.write_text(path.read_text().replace("(1e11 1e11 1e11)", "(1e11 2e11 4e11)", 1))
            configure_ptl_porosity(case, 0.5)
            values = darcy(zone_block(path.read_text(), "anodeGDL"))
            self.assertAlmostEqual(values[1] / values[0], 2)
            self.assertAlmostEqual(values[2] / values[0], 4)

    def test_invalid_porosity_or_side_does_not_mutate_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            copy_clean_case(ROOT / "run/AEMEC", case)
            before = case_fingerprint(case)
            for epsilon in (0, 1, -0.1, 1.1, math.nan, math.inf, 1e-200):
                with self.subTest(epsilon=epsilon), self.assertRaises(OptimizationError):
                    configure_ptl_porosity(case, epsilon)
                self.assertEqual(case_fingerprint(case), before)
            with self.assertRaises(OptimizationError):
                configure_ptl_porosity(case, 0.6, "all")
            self.assertEqual(case_fingerprint(case), before)

    def test_inconsistent_reference_or_unsupported_model_is_rejected_atomically(self) -> None:
        cases = (
            ("cathode/diffusivityModel.gas", "porosity    0.7;", "porosity    0.6;", "Inconsistent reference"),
            ("cathode/porousZones", "f   (0 0 0);", "f   (1 0 0);", "requires f"),
            ("cathode/porousZones", "d   (1e11 1e11 1e11);", "d   (-1 1e11 1e11);", "must be positive"),
            ("phiECathode/regionProperties", "sigmaModel      porousSigma;", "sigmaModel      constantSigma;", "Expected porousSigma"),
        )
        for relative, old, new, message in cases:
            with self.subTest(relative=relative, message=message), tempfile.TemporaryDirectory() as directory:
                case = Path(directory) / "case"
                copy_clean_case(ROOT / "run/AEMEC", case)
                path = case / "constant" / relative
                original = path.read_text()
                if relative.startswith("phiE"):
                    start, end = _named_block_span(original, "cathodeGDL")
                    updated = original[:start] + original[start:end].replace(old, new) + original[end:]
                else:
                    updated = original.replace(old, new, 1)
                self.assertNotEqual(original, updated)
                path.write_text(updated)
                before = case_fingerprint(case)
                with self.assertRaisesRegex(OptimizationError, message):
                    configure_ptl_porosity(case, 0.5, "both")
                self.assertEqual(case_fingerprint(case), before)


if __name__ == "__main__":
    unittest.main()
