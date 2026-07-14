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


def without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"(?m)^\s*#.*$", "", text)


def matching_brace(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unterminated dictionary block")


def parse_boundary_field(text: str) -> list[tuple[str, dict[str, str]]]:
    clean = without_comments(text)
    match = re.search(r"\bboundaryField\s*\{", clean)
    if not match:
        raise ValueError("missing boundaryField dictionary")

    opening = clean.find("{", match.start())
    closing = matching_brace(clean, opening)
    body = clean[opening + 1 : closing]
    patches: list[tuple[str, dict[str, str]]] = []
    position = 0

    while position < len(body):
        while position < len(body) and (body[position].isspace() or body[position] == ";"):
            position += 1
        if position >= len(body):
            break

        if body[position] == '"':
            end = body.find('"', position + 1)
            if end == -1:
                raise ValueError("unterminated quoted patch name")
            patch = body[position : end + 1]
            position = end + 1
        else:
            match = re.match(r"[^\s{};]+", body[position:])
            if not match:
                raise ValueError("cannot parse patch name")
            patch = match.group(0)
            position += len(patch)

        while position < len(body) and body[position].isspace():
            position += 1
        if position >= len(body) or body[position] != "{":
            raise ValueError(f"patch '{patch}' is missing its dictionary block")

        patch_closing = matching_brace(body, position)
        patch_body = body[position + 1 : patch_closing]
        entry_matches = list(
            re.finditer(
                r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s+([^;{}]+);",
                patch_body,
            )
        )
        entry_names = [entry.group(1) for entry in entry_matches]
        duplicates = sorted(
            {name for name in entry_names if entry_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(
                f"patch '{patch}' repeats keyword(s): {', '.join(duplicates)}"
            )
        entries = {
            entry.group(1): entry.group(2).strip() for entry in entry_matches
        }
        patches.append((patch, entries))
        position = patch_closing + 1

    return patches


def boundary_map(text: str) -> dict[str, dict[str, str]]:
    return dict(parse_boundary_field(text))


def check_boundary_syntax(case: Path, errors: list[str]) -> None:
    required_entries = {
        "fixedValue": ("value",),
        "calculated": ("value",),
        "inletOutlet": ("inletValue", "value", "phi"),
        "pressureInletOutletVelocity": ("value", "phi"),
        "prghPressure": ("p", "value"),
        "copiedFixedValue": ("sourceFieldName", "value"),
        "zeroGradient": (),
    }
    expected_patches = {
        Path("anode"): {
            "anodeInlet",
            "anodeOutlet",
            "anodeSides",
            "anode_to_electrolyte",
            "anode_to_interconnect",
        },
        Path("cathode"): {
            "cathodeInlet",
            "cathodeOutlet",
            "cathodeSides",
            "cathode_to_electrolyte",
            "cathode_to_interconnect",
        },
        Path("electrolyte"): {
            "electrolyteSides",
            "electrolyte_to_anode",
            "electrolyte_to_cathode",
        },
        Path("interconnect"): {
            "interconnect0",
            "interconnect1",
            "interconnectSides",
            "interconnect_to_anode",
            "interconnect_to_cathode",
        },
    }
    master_patches = {
        "interconnect0",
        "interconnect1",
        "interconnectSides",
        "anodeInlet",
        "anodeOutlet",
        "anodeSides",
        "electrolyteSides",
        "cathodeInlet",
        "cathodeOutlet",
        "cathodeSides",
    }

    for field in sorted((case / "0.orig").rglob("*")):
        if not field.is_file():
            continue
        relative = field.relative_to(case / "0.orig")
        text = read(field, errors)
        try:
            patches = parse_boundary_field(text)
        except ValueError as error:
            errors.append(f"0.orig/{relative}: {error}")
            continue

        names = [name for name, _ in patches]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        for patch in duplicates:
            errors.append(f"0.orig/{relative} repeats boundary entry '{patch}'")

        patch_map = dict(patches)
        expected = master_patches if len(relative.parts) == 1 else expected_patches.get(relative.parent)
        if expected is not None and '".*"' not in patch_map:
            for patch in sorted(expected - patch_map.keys()):
                errors.append(f"0.orig/{relative} is missing boundary entry '{patch}'")

        for patch, entries in patches:
            boundary_type = entries.get("type")
            if boundary_type is None:
                errors.append(f"0.orig/{relative}:{patch} is missing 'type'")
                continue
            if boundary_type not in required_entries:
                errors.append(
                    f"0.orig/{relative}:{patch} uses unsupported boundary type "
                    f"'{boundary_type}'"
                )
            for keyword in required_entries.get(boundary_type, ()):
                if keyword not in entries:
                    errors.append(
                        f"0.orig/{relative}:{patch} type {boundary_type} is missing '{keyword}'"
                    )
            for keyword in ("value", "inletValue", "p", "p0"):
                if entries.get(keyword) == "internalField":
                    errors.append(
                        f"0.orig/{relative}:{patch} must use '$internalField', not 'internalField'"
                    )
            if boundary_type == "copiedFixedValue":
                source = entries.get("sourceFieldName")
                if source and not (field.parent / source).is_file():
                    errors.append(
                        f"0.orig/{relative}:{patch} references missing source field '{source}'"
                    )


def scalar_internal_field(text: str) -> float | None:
    match = re.search(
        r"(?m)^\s*internalField\s+uniform\s+([-+0-9.eE]+)\s*;",
        without_comments(text),
    )
    return float(match.group(1)) if match else None


def scalar_boundary_value(entry: dict[str, str], internal: float | None) -> float | None:
    value = entry.get("value")
    if value == "$internalField":
        return internal
    match = re.fullmatch(r"uniform\s+([-+0-9.eE]+)", value or "")
    return float(match.group(1)) if match else None


def scalar_entry_value(
    entry: dict[str, str], keyword: str, internal: float | None
) -> float | None:
    value = entry.get(keyword)
    if value == "$internalField":
        return internal
    match = re.fullmatch(r"uniform\s+([-+0-9.eE]+)", value or "")
    return float(match.group(1)) if match else None


def vector_internal_field(text: str) -> tuple[float, float, float] | None:
    match = re.search(
        r"(?m)^\s*internalField\s+uniform\s+\(\s*"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*;",
        without_comments(text),
    )
    return tuple(float(match.group(index)) for index in range(1, 4)) if match else None


def check_boundary_contract(case: Path, errors: list[str]) -> None:
    """Check the BC combinations on which this particular AEMEC setup relies."""

    fields: dict[str, tuple[str, dict[str, dict[str, str]]]] = {}

    def field(relative: str) -> tuple[str, dict[str, dict[str, str]]]:
        if relative not in fields:
            text = read(case / relative, errors)
            try:
                patches = boundary_map(text)
            except ValueError:
                patches = {}
            fields[relative] = (text, patches)
        return fields[relative]

    def expect(
        relative: str,
        patch: str,
        keyword: str,
        expected: str,
    ) -> None:
        _, patches = field(relative)
        actual = patches.get(patch, {}).get(keyword)
        if actual != expected:
            errors.append(
                f"{relative}:{patch} must set '{keyword} {expected};' "
                f"(found {actual!r})"
            )

    # Both electrodes are liquid-fed. The gas phase is retained for species
    # transport and crossover, but it must not be given a second bulk inlet.
    for region in ("anode", "cathode"):
        inlet = f"{region}Inlet"
        outlet = f"{region}Outlet"

        gas_path = f"0.orig/{region}/U.gas"
        gas_text, _ = field(gas_path)
        if vector_internal_field(gas_text) != (0.0, 0.0, 0.0):
            errors.append(f"{gas_path} must initialize the stationary gas phase at (0 0 0) m/s")
        expect(gas_path, inlet, "type", "fixedValue")
        gas_inlet_value = field(gas_path)[1].get(inlet, {}).get("value")
        if gas_inlet_value not in ("$internalField", "uniform (0 0 0)"):
            errors.append(f"{gas_path}:{inlet} must impose zero gas velocity")
        expect(gas_path, outlet, "type", "pressureInletOutletVelocity")
        expect(gas_path, outlet, "phi", "phi.gas")

        water_path = f"0.orig/{region}/U.water"
        water_text, _ = field(water_path)
        if vector_internal_field(water_text) != (0.01, 0.0, 0.0):
            errors.append(f"{water_path} must initialize the liquid feed at (0.01 0 0) m/s")
        expect(water_path, inlet, "type", "fixedValue")
        expect(water_path, inlet, "value", "uniform (0.01 0 0)")
        expect(water_path, outlet, "type", "pressureInletOutletVelocity")
        expect(water_path, outlet, "phi", "phi.water")

        alpha_gas_path = f"0.orig/{region}/alpha.gas"
        alpha_water_path = f"0.orig/{region}/alpha.water"
        alpha_gas_text, alpha_gas_patches = field(alpha_gas_path)
        alpha_water_text, alpha_water_patches = field(alpha_water_path)
        alpha_gas = scalar_internal_field(alpha_gas_text)
        alpha_water = scalar_internal_field(alpha_water_text)
        if alpha_gas is None or alpha_water is None or abs(alpha_gas + alpha_water - 1.0) > 1e-12:
            errors.append(f"0.orig/{region}/alpha.gas and alpha.water internal fractions must sum to 1")
        expect(alpha_gas_path, inlet, "type", "fixedValue")
        expect(alpha_water_path, inlet, "type", "calculated")
        gas_inlet = scalar_boundary_value(alpha_gas_patches.get(inlet, {}), alpha_gas)
        water_inlet = scalar_boundary_value(alpha_water_patches.get(inlet, {}), alpha_water)
        if gas_inlet is None or water_inlet is None or abs(gas_inlet + water_inlet - 1.0) > 1e-12:
            errors.append(f"0.orig/{region} inlet phase fractions must sum to 1")
        expect(alpha_gas_path, outlet, "type", "inletOutlet")
        expect(alpha_gas_path, outlet, "phi", "phi.gas")

        pressure_path = f"0.orig/{region}/p"
        expect(pressure_path, inlet, "type", "calculated")
        expect(pressure_path, outlet, "type", "calculated")
        prgh_path = f"0.orig/{region}/p_rgh"
        expect(prgh_path, inlet, "type", "zeroGradient")
        expect(prgh_path, outlet, "type", "prghPressure")

        for phase in ("gas", "water"):
            temperature_path = f"0.orig/{region}/T.{phase}"
            expect(temperature_path, inlet, "type", "fixedValue")
            expect(temperature_path, outlet, "type", "inletOutlet")
            expect(temperature_path, outlet, "phi", f"phi.{phase}")

    species = {
        "anode": ("H2.gas", "H2O.gas", "O2.gas"),
        "cathode": ("H2.gas", "H2O.gas"),
    }
    for region, names in species.items():
        inlet = f"{region}Inlet"
        outlet = f"{region}Outlet"
        internal_sum = 0.0
        inlet_sum = 0.0
        reverse_flow_sum = 0.0
        complete = True
        for name in names:
            relative = f"0.orig/{region}/{name}"
            text, patches = field(relative)
            internal = scalar_internal_field(text)
            inlet_value = scalar_boundary_value(patches.get(inlet, {}), internal)
            reverse_value = scalar_entry_value(
                patches.get(outlet, {}), "inletValue", internal
            )
            if any(value is None or value < 0.0 or value > 1.0 for value in (internal, inlet_value, reverse_value)):
                errors.append(f"{relative} must define species fractions between 0 and 1")
                complete = False
            else:
                internal_sum += internal
                inlet_sum += inlet_value
                reverse_flow_sum += reverse_value
            expect(relative, inlet, "type", "fixedValue")
            expect(relative, outlet, "type", "inletOutlet")
            expect(relative, outlet, "phi", "phi.gas")
        if complete:
            for location, total in (
                ("internal", internal_sum),
                ("inlet", inlet_sum),
                ("outlet reverse-flow", reverse_flow_sum),
            ):
                if abs(total - 1.0) > 1e-9:
                    errors.append(
                        f"0.orig/{region} gas species fractions at the {location} sum to {total:g}, not 1"
                    )

    # The two electronic meshes occupy opposite current collectors. Keep the
    # field, controller, faceSet and created patch names on the same collector.
    collector_contract = {
        "phiEAnode": ("interconnect0", "interconnect1"),
        "phiECathode": ("interconnect1", "interconnect0"),
    }
    for region, (collector, wrong_collector) in collector_contract.items():
        potential = f"0.orig/{region}/phi"
        expect(potential, '".*"', "type", "zeroGradient")
        expect(potential, collector, "type", "fixedValue")
        create_patch_path = f"system/{region}/createPatchDict"
        create_patch = read(case / create_patch_path, errors)
        if not has_entry(create_patch, "name", collector) or not has_entry(
            create_patch, "set", collector
        ):
            errors.append(
                f"{create_patch_path} must create collector patch '{collector}' from its matching faceSet"
            )
        if has_entry(create_patch, "name", wrong_collector) or has_entry(
            create_patch, "set", wrong_collector
        ):
            errors.append(
                f"{create_patch_path} incorrectly references opposite collector '{wrong_collector}'"
            )

    controller = read(case / "constant/phiEAnode/regionProperties", errors)
    if not has_entry(controller, "patchName", "interconnect0"):
        errors.append(
            "constant/phiEAnode/regionProperties must control the interconnect0 collector"
        )

    for relative in ("0.orig/T", "0.orig/interconnect/T"):
        expect(relative, "interconnect0", "type", "fixedValue")
        expect(relative, "interconnect1", "type", "fixedValue")

    physical_boundaries = (
        "interconnect0",
        "interconnect1",
        "interconnectSides",
        "anodeInlet",
        "anodeOutlet",
        "anodeSides",
        "electrolyteSides",
        "cathodeInlet",
        "cathodeOutlet",
        "cathodeSides",
    )
    block_mesh = read(case / "system/blockMeshDict", errors)
    salome_mesh = read(case / "salomeMesh.py", errors)
    for patch in physical_boundaries:
        if not has_dictionary_block(block_mesh, patch):
            errors.append(f"system/blockMeshDict is missing physical boundary '{patch}'")
        if not re.search(
            rf"GroupOnGeom\(\s*{re.escape(patch)}\s*,\s*'{re.escape(patch)}'",
            salome_mesh,
        ):
            errors.append(f"salomeMesh.py is missing physical boundary group '{patch}'")


def check_static(case: Path, errors: list[str]) -> None:
    check_boundary_syntax(case, errors)
    check_boundary_contract(case, errors)

    region_properties = read(case / "constant/regionProperties", errors)
    for region in ("anode", "cathode", "electrolyte", "interconnect", "phiEAnode", "phiECathode", "phiAnion"):
        if region not in region_properties:
            errors.append(f"constant/regionProperties does not declare region '{region}'")

    expected_phases = {"anode": "phases (gas water)", "cathode": "phases (gas water)"}
    for region, phase_line in expected_phases.items():
        text = read(case / "constant" / region / "regionProperties", errors)
        if phase_line not in text:
            errors.append(f"constant/{region}/regionProperties must contain '{phase_line}'")

    for relative_path in (
        "0.orig/anode/H2.gas",
        "0.orig/anode/H2O.gas",
        "0.orig/anode/O2.gas",
        "0.orig/anode/T.gas",
        "0.orig/anode/U.gas",
        "0.orig/anode/alpha.gas",
        "constant/anode/combustionProperties.gas",
        "constant/anode/diffusivityModel.gas",
        "constant/anode/thermophysicalProperties.gas",
        "constant/anode/turbulenceProperties.gas",
        "0.orig/cathode/H2.gas",
        "0.orig/cathode/H2O.gas",
        "0.orig/cathode/T.gas",
        "0.orig/cathode/U.gas",
        "0.orig/cathode/alpha.gas",
        "constant/cathode/diffusivityModel.gas",
        "constant/cathode/thermophysicalProperties.gas",
        "constant/cathode/turbulenceProperties.gas",
    ):
        read(case / relative_path, errors)

    expected_interfaces = {
        "0.orig/electrolyte/T": ("electrolyte_to_anode", "electrolyte_to_cathode"),
        "0.orig/interconnect/T": ("interconnect_to_anode", "interconnect_to_cathode"),
    }
    for relative_path, patches in expected_interfaces.items():
        text = read(case / relative_path, errors)
        for patch in patches:
            if not has_dictionary_block(text, patch):
                errors.append(f"{relative_path} is missing boundary entry '{patch}'")

    for gas, liquid, region in (("gas", "water", "anode"), ("gas", "water", "cathode")):
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

    anion_schemes = read(case / "system/phiAnion/fvSchemes", errors)
    for entry in ("div(phiH2Drag,cH2)", "div(phiH2Conv,cH2)"):
        if entry not in anion_schemes:
            errors.append(
                "system/phiAnion/fvSchemes is missing the hydrogen-transport "
                f"convection scheme '{entry}'"
            )

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
    if anion_properties.count("gasPhase            gas;") != 2:
        errors.append(
            "constant/phiAnion/regionProperties must use the gas phase at both crossover interfaces"
        )
    for interface in ("cathodeInterface", "anodeInterface"):
        if not has_dictionary_block(anion_properties, interface) or "henryCoefficient" not in anion_properties:
            errors.append(
                f"constant/phiAnion/regionProperties must configure Henry interface '{interface}'"
            )

    anode_thermo = read(case / "constant/anode/thermophysicalProperties.gas", errors)
    if not re.search(r"species\s*\([^)]*\bH2\b", anode_thermo, re.DOTALL):
        errors.append("constant/anode/thermophysicalProperties.gas must include H2")
    anode_h2 = read(case / "0.orig/anode/H2.gas", errors)
    for patch in ("anodeInlet", "anodeOutlet", "anode_to_electrolyte", "anode_to_interconnect"):
        if not has_dictionary_block(anode_h2, patch):
            errors.append(f"0.orig/anode/H2.gas is missing boundary entry '{patch}'")
    anode_water_velocity = read(case / "0.orig/anode/U.water", errors)
    if "uniform (0.01 0 0)" not in anode_water_velocity:
        errors.append("0.orig/anode/U.water must provide the aqueous-KOH feed at (0.01 0 0) m/s")

    anode_controller = read(case / "constant/phiEAnode/regionProperties", errors)
    if not has_dictionary_block(anode_controller, "polarizationCurve"):
        errors.append("constant/phiEAnode/regionProperties must define polarizationCurve")
    control_mode = re.search(
        r"(?s)\bgalvanostatic\s*\{\s*active\s+(true|false)\s*;",
        anode_controller,
    )
    if not control_mode:
        errors.append("constant/phiEAnode/regionProperties must declare galvanostatic.active")
    elif control_mode.group(1) == "true":
        for entry in (
            "targets                     (-6000 -9000 -12000 -15000);",
            "minimumHoldDuration         15;",
            "targetCurrentTolerance      0.05;",
            "voltageTolerance            0.002;",
            "currentStabilityTolerance   0.02;",
            "stabilitySamples            5;",
        ):
            if entry not in anode_controller:
                errors.append(
                    f"constant/phiEAnode/regionProperties is missing stable polarization entry '{entry}'"
                )

        if "maxVoltageStep 0.01;" not in anode_controller:
            errors.append(
                "constant/phiEAnode/regionProperties must use maxVoltageStep 0.01 for the POC scan"
            )
    else:
        if "type    constant;" not in anode_controller or not re.search(
            r"(?m)^\s*value\s+[-+0-9.eE]+\s*;", anode_controller
        ):
            errors.append(
                "fixed-voltage diagnostic must define a constant collector voltage"
            )

    control_run = read(case / "system/controlDict.run", errors)
    is_diagnostic = control_mode is not None and control_mode.group(1) == "false"
    if is_diagnostic:
        end_match = re.search(r"(?m)^\s*endTime\s+([-+0-9.eE]+)\s*;", control_run)
        write_match = re.search(r"(?m)^\s*writeInterval\s+([-+0-9.eE]+)\s*;", control_run)
        if not end_match or not write_match or float(end_match.group(1)) <= 0.0:
            errors.append("fixed-voltage diagnostic must define positive endTime and writeInterval")
        elif float(end_match.group(1)) != float(write_match.group(1)):
            errors.append("fixed-voltage diagnostic must write the final endTime state")
    else:
        for entry in ("endTime         120;", "writeInterval   120;"):
            if entry not in control_run:
                errors.append(
                    f"system/controlDict.run is missing expected run setting '{entry}'"
                )

    for relative_path in ("constant/anode/combustionProperties.gas", "constant/cathode/combustionProperties"):
        text = read(case / relative_path, errors)
        if "jMax            5.0e8;" not in text or "exponentLimit   50;" not in text:
            errors.append(f"{relative_path} must define the proof-of-concept Butler-Volmer current bound")
        if is_diagnostic and "electrochemicalDiagnostics true;" not in text:
            errors.append(f"{relative_path} must enable electrochemicalDiagnostics in a fixed-voltage diagnostic")
    if is_diagnostic and "electricDiagnostics true;" not in anion_properties:
        errors.append("constant/phiAnion/regionProperties must enable electricDiagnostics in a fixed-voltage diagnostic")

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
    diffusion_files = {"anode": "diffusivityModel.gas", "cathode": "diffusivityModel.gas"}
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

    for gas, liquid, region in (("gas", "water", "anode"), ("gas", "water", "cathode")):
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

    anode_h2 = read(case / "0/anode/H2.gas", errors)
    if "internalField   uniform 1e-12;" not in anode_h2:
        errors.append(
            "generated 0/anode/H2.gas is stale; run 'make mesh' to apply the anode H2 crossover field"
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
