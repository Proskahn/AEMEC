#!/usr/bin/env python3
"""Validate the physical-region case contract before starting openFuelCell.

The check is deliberately independent of OpenFOAM utilities.  It detects a
renamed zone that was not propagated through a dictionary or generated mesh,
which would otherwise fail only after the solver has begun initialisation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


FLUID_REGION_ZONES = {
    "anode": ("anodeChannel", "anodeGDL", "anodeMPL", "anodeCL"),
    "cathode": ("cathodeChannel", "cathodeGDL", "cathodeMPL", "cathodeCL"),
}

POROUS_FLUID_ZONES = {
    "anode": ("anodeGDL", "anodeMPL", "anodeCL"),
    "cathode": ("cathodeGDL", "cathodeMPL", "cathodeCL"),
}

ELECTRIC_ZONES = {
    "phiEAnode": ("anodeGDL", "anodeMPL", "anodeCL", "bpp"),
    "phiECathode": ("cathodeGDL", "cathodeMPL", "cathodeCL", "bpp"),
    "phiAnion": ("anodeCL", "cathodeCL", "membrane"),
}


def read(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"missing required file: {path.relative_to(Path.cwd())}")
        return ""


def has_dictionary_block(text: str, name: str) -> bool:
    return re.search(rf"(?m)^\s*{re.escape(name)}\s*\{{", text) is not None


def has_entry(text: str, keyword: str, value: str) -> bool:
    return re.search(
        rf"(?m)^\s*{re.escape(keyword)}\s+{re.escape(value)}\s*;", text
    ) is not None


def check_static(case: Path, errors: list[str]) -> None:
    region_properties = read(case / "constant/regionProperties", errors)
    for region in ("anode", "cathode", "electrolyte", "interconnect", "phiEAnode", "phiECathode", "phiAnion"):
        if region not in region_properties:
            errors.append(f"constant/regionProperties does not declare region '{region}'")

    expected_phases = {"anode": "phases (oxygen water)", "cathode": "phases (hydrogen water)"}
    for region, phase_line in expected_phases.items():
        text = read(case / "constant" / region / "regionProperties", errors)
        if phase_line not in text:
            errors.append(f"constant/{region}/regionProperties must contain '{phase_line}'")

    for region, zones in POROUS_FLUID_ZONES.items():
        porous = read(case / "constant" / region / "porousZones", errors)
        topological = read(case / "system" / region / "topoSetDict", errors)
        for zone in zones:
            if not has_dictionary_block(porous, zone) or not has_entry(porous, "cellZone", zone):
                errors.append(f"constant/{region}/porousZones does not configure cellZone '{zone}'")
        for zone in FLUID_REGION_ZONES[region]:
            if not has_entry(topological, "name", zone):
                errors.append(f"system/{region}/topoSetDict does not generate cellZone '{zone}'")

    diffusion_zones = {"anode": "anodeChannel", "cathode": "cathodeChannel"}
    diffusion_files = {"anode": "diffusivityModel.oxygen", "cathode": "diffusivityModel.hydrogen"}
    for region, zone in diffusion_zones.items():
        text = read(case / "constant" / region / diffusion_files[region], errors)
        if not has_dictionary_block(text, zone):
            errors.append(
                f"constant/{region}/{diffusion_files[region]} does not configure cellZone '{zone}'"
            )

    master_zones = ("electrolyte", "interconnect", *FLUID_REGION_ZONES["anode"], *FLUID_REGION_ZONES["cathode"])
    zone_to_set = read(case / "system/topoSetDict.zoneToSet", errors)
    for zone in master_zones:
        if not has_entry(zone_to_set, "zone", zone):
            errors.append(f"system/topoSetDict.zoneToSet does not import master cellZone '{zone}'")


def check_generated_mesh(case: Path, errors: list[str]) -> None:
    for region, zones in {**FLUID_REGION_ZONES, **ELECTRIC_ZONES}.items():
        path = case / "constant" / region / "polyMesh/cellZones"
        text = read(path, errors)
        for zone in zones:
            if not has_dictionary_block(text, zone):
                errors.append(f"generated mesh {path.relative_to(case)} is missing cellZone '{zone}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static",
        action="store_true",
        help="check dictionaries only; do not require a generated mesh",
    )
    args = parser.parse_args()
    case = Path(__file__).resolve().parent
    errors: list[str] = []
    check_static(case, errors)
    if not args.static:
        check_generated_mesh(case, errors)

    if errors:
        print("AEMEC case preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    scope = "dictionary" if args.static else "dictionary and generated-mesh"
    print(f"AEMEC case preflight passed ({scope} contract).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
