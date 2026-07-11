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

LEGACY_INTERFACE_NAMES = ("electrolyte_to_air", "electrolyte_to_fuel", "interconnect_to_air", "interconnect_to_fuel")


def read(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"missing required file: {path.relative_to(Path.cwd())}")
        return ""


def has_dictionary_block(text: str, name: str) -> bool:
    return re.search(
        rf"(?m)^[ \t]*{re.escape(name)}(?:[ \t]*//[^\n]*)?[ \t]*(?:\n[ \t]*)?\{{",
        text,
    ) is not None


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

    expected_interfaces = {
        "0.orig/electrolyte/T": ("electrolyte_to_anode", "electrolyte_to_cathode"),
        "0.orig/interconnect/T": ("interconnect_to_anode", "interconnect_to_cathode"),
    }
    for relative_path, patches in expected_interfaces.items():
        text = read(case / relative_path, errors)
        for patch in patches:
            if not has_dictionary_block(text, patch):
                errors.append(f"{relative_path} is missing boundary entry '{patch}'")

    for gas, liquid, region in (("oxygen", "water", "anode"), ("hydrogen", "water", "cathode")):
        gas_field = read(case / f"0.orig/{region}/alpha.{gas}", errors)
        liquid_field = read(case / f"0.orig/{region}/alpha.{liquid}", errors)
        if "internalField   uniform 1e-4;" not in gas_field:
            errors.append(f"0.orig/{region}/alpha.{gas} must start with a nonzero 1e-4 gas fraction")
        if "internalField   uniform 0.9999;" not in liquid_field:
            errors.append(f"0.orig/{region}/alpha.{liquid} must complement the gas fraction at startup")

    for field_name in ("cH2", "phi"):
        relative_path = f"0.orig/phiAnion/{field_name}"
        text = read(case / relative_path, errors)
        if not has_entry(text, "object", field_name):
            errors.append(f"{relative_path} must declare 'object {field_name};'")

    anion_properties = read(case / "constant/phiAnion/regionProperties", errors)
    if "lambdaSigma" in anion_properties or "lambdaName" in anion_properties:
        errors.append("constant/phiAnion/regionProperties must not use the PEM hydration-based lambdaSigma model")
    for zone, conductivity in (("anodeCL", "1.10"), ("cathodeCL", "1.10"), ("membrane", "11.4")):
        if not has_dictionary_block(anion_properties, zone) or f"sigma               {conductivity};" not in anion_properties:
            errors.append(f"constant/phiAnion/regionProperties is missing effective conductivity {conductivity} for '{zone}'")

    for entry in (
        "cathodeFluidRegion  cathode;",
        "anodeFluidRegion    anode;",
        "hydrogenSpecies     H2;",
        "diffusivityModel    porosityTortuosity;",
        "dragModel       constant;",
        "UMembrane       (0 0 0);",
    ):
        if entry not in anion_properties:
            errors.append(
                f"constant/phiAnion/regionProperties is missing crossover entry '{entry}'"
            )
    for interface in ("cathodeInterface", "anodeInterface"):
        if not has_dictionary_block(anion_properties, interface) or "henryCoefficient" not in anion_properties:
            errors.append(
                f"constant/phiAnion/regionProperties must configure Henry interface '{interface}'"
            )

    anode_thermo = read(case / "constant/anode/thermophysicalProperties.oxygen", errors)
    if not re.search(r"species\s*\([^)]*\bH2\b", anode_thermo, re.DOTALL):
        errors.append("constant/anode/thermophysicalProperties.oxygen must include H2")
    anode_h2 = read(case / "0.orig/anode/H2.oxygen", errors)
    for patch in ("anodeInlet", "anodeOutlet", "anode_to_electrolyte", "anode_to_interconnect"):
        if not has_dictionary_block(anode_h2, patch):
            errors.append(f"0.orig/anode/H2.oxygen is missing boundary entry '{patch}'")
    anode_water_velocity = read(case / "0.orig/anode/U.water", errors)
    if "uniform (0.01 0 0)" not in anode_water_velocity:
        errors.append("0.orig/anode/U.water must provide the aqueous-KOH feed at (0.01 0 0) m/s")

    for relative_path in ("constant/anode/combustionProperties.oxygen", "constant/cathode/combustionProperties"):
        text = read(case / relative_path, errors)
        if "jMax            5.0e8;" not in text or "exponentLimit   50;" not in text:
            errors.append(f"{relative_path} must define the proof-of-concept Butler-Volmer current bound")

    for field in sorted((case / "0.orig").rglob("*")):
        if not field.is_file():
            continue
        text = read(field, errors)
        for legacy_name in LEGACY_INTERFACE_NAMES:
            if legacy_name in text:
                errors.append(f"{field.relative_to(case)} still references legacy patch '{legacy_name}'")

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

    for region in ("anode", "cathode", "electrolyte", "interconnect", "phiEAnode", "phiECathode", "phiAnion"):
        boundary_path = case / "constant" / region / "polyMesh/boundary"
        mesh_boundary = read(boundary_path, errors)
        patches = re.findall(r"(?m)^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\n[ \t]*\{", mesh_boundary)
        field_directory = case / "0" / region
        if not field_directory.is_dir():
            errors.append(f"generated initial-field directory is missing: 0/{region}")
            continue
        for field in sorted(field_directory.iterdir()):
            if not field.is_file():
                continue
            field_text = read(field, errors)
            has_wildcard = re.search(r'(?m)^[ \t]*"\.\*"[ \t]*\n[ \t]*\{', field_text) is not None
            for patch in patches:
                if not has_wildcard and not has_dictionary_block(field_text, patch):
                    errors.append(
                        f"initial field {field.relative_to(case)} is missing boundary entry '{patch}'"
                    )

    for gas, liquid, region in (("oxygen", "water", "anode"), ("hydrogen", "water", "cathode")):
        gas_field = read(case / f"0/{region}/alpha.{gas}", errors)
        liquid_field = read(case / f"0/{region}/alpha.{liquid}", errors)
        if "internalField   uniform 1e-4;" not in gas_field:
            errors.append(
                f"generated 0/{region}/alpha.{gas} is stale; run 'make mesh' to apply the nonzero gas seed"
            )
        if "internalField   uniform 0.9999;" not in liquid_field:
            errors.append(
                f"generated 0/{region}/alpha.{liquid} is stale; run 'make mesh' to apply the complementary liquid fraction"
            )

    anode_h2 = read(case / "0/anode/H2.oxygen", errors)
    if "internalField   uniform 1e-12;" not in anode_h2:
        errors.append(
            "generated 0/anode/H2.oxygen is stale; run 'make mesh' to apply the anode H2 crossover field"
        )


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
