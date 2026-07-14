#!/usr/bin/env python3
"""Create an isolated fixed-voltage AEMEC diagnostic case.

The copied case holds the physical anode collector at a constant voltage and
enables electrochemical diagnostics without modifying the source case used by
the polarization scan or optimizer.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def matching_brace(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unclosed OpenFOAM dictionary block")


def named_block(text: str, name: str) -> tuple[int, int]:
    match = re.search(rf"\b{re.escape(name)}\b\s*\{{", text)
    if not match:
        raise ValueError(f"Cannot find dictionary block '{name}'")
    opening = text.find("{", match.start())
    return match.start(), matching_brace(text, opening) + 1


def set_first_switch(block: str, value: bool) -> str:
    replacement = "true" if value else "false"
    rewritten, count = re.subn(
        r"(?m)^(\s*active\s+)(?:true|false)(\s*;)",
        rf"\g<1>{replacement}\g<2>",
        block,
        count=1,
    )
    if count != 1:
        raise ValueError("Cannot update an 'active' switch")
    return rewritten


def configure_electrode(text: str, voltage: float) -> str:
    start, end = named_block(text, "galvanostatic")
    block = set_first_switch(text[start:end], False)

    curve_start, curve_end = named_block(block, "polarizationCurve")
    curve = set_first_switch(block[curve_start:curve_end], False)
    block = block[:curve_start] + curve + block[curve_end:]

    voltage_block = (
        "\n    // Fixed-voltage diagnostic; galvanostatic feedback is disabled.\n"
        "    voltage\n"
        "    {\n"
        "        type    constant;\n"
        f"        value   {voltage:g};\n"
        "    }\n"
    )
    rewritten, count = re.subn(
        r"\n(\s*ibar\s*\{)",
        voltage_block + r"\n\1",
        block,
        count=1,
    )
    if count != 1:
        raise ValueError("Cannot insert the fixed-voltage function")

    return text[:start] + rewritten + text[end:]


def configure_control_dict(text: str, end_time: float) -> str:
    def set_scalar(contents: str, name: str, value: float) -> str:
        rewritten, count = re.subn(
            rf"(?m)^(\s*{name}\s+)[^;]+;",
            rf"\g<1>{value:g};",
            contents,
            count=1,
        )
        if count != 1:
            raise ValueError(f"Cannot update '{name}' in controlDict")
        return rewritten

    text = set_scalar(text, "endTime", end_time)
    return set_scalar(text, "writeInterval", end_time)


def enable_diagnostics(text: str) -> str:
    if re.search(r"\belectrochemicalDiagnostics\s+", text):
        return re.sub(
            r"(?m)^(\s*electrochemicalDiagnostics\s+)(?:true|false)(\s*;)",
            r"\g<1>true\g<2>",
            text,
            count=1,
        )

    rewritten, count = re.subn(
        r"(?m)^(\s*combustionModel\s+electroChemicalReaction\s*;)",
        r"\1\n\nelectrochemicalDiagnostics true;",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Cannot enable electrochemical diagnostics")
    return rewritten


def enable_electric_diagnostics(text: str) -> str:
    if re.search(r"\belectricDiagnostics\s+", text):
        return re.sub(
            r"(?m)^(\s*electricDiagnostics\s+)(?:true|false)(\s*;)",
            r"\g<1>true\g<2>",
            text,
            count=1,
        )

    rewritten, count = re.subn(
        r"(?m)^(\s*relax\s+[^;]+;)",
        r"\1\n\nelectricDiagnostics true;",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Cannot enable phiAnion electric diagnostics")
    return rewritten


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("run/AEMEC-fixed-voltage"),
        help="Destination for the copied diagnostic case.",
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=1.5,
        help="Fixed physical-anode collector voltage in V (default: 1.5).",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=12.0,
        help="Diagnostic simulation duration in s (default: 12).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.voltage <= 0.0 or args.end_time <= 0.0:
        raise SystemExit("--voltage and --end-time must be positive")

    source = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output == source:
        raise SystemExit("The diagnostic output must not replace the source case")
    if output.exists():
        if not args.force:
            raise SystemExit(f"Diagnostic output already exists: {output}; use --force")
        shutil.rmtree(output)

    shutil.copytree(
        source,
        output,
        ignore=shutil.ignore_patterns("log.*", "processor*", "postProcessing", "VTK"),
    )

    electrode = output / "constant/phiEAnode/regionProperties"
    electrode.write_text(
        configure_electrode(electrode.read_text(encoding="utf-8"), args.voltage),
        encoding="utf-8",
    )
    for name in ("controlDict.run", "controlDict"):
        control_dict = output / "system" / name
        control_dict.write_text(
            configure_control_dict(
                control_dict.read_text(encoding="utf-8"), args.end_time
            ),
            encoding="utf-8",
        )
    for path in (
        output / "constant/cathode/combustionProperties",
        output / "constant/anode/combustionProperties.gas",
    ):
        path.write_text(enable_diagnostics(path.read_text(encoding="utf-8")), encoding="utf-8")
    anion_properties = output / "constant/phiAnion/regionProperties"
    anion_properties.write_text(
        enable_electric_diagnostics(anion_properties.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    print(f"Created fixed-voltage diagnostic case: {output}")
    print("Next: cd", output)
    print("Then: make clear && make mesh && make srun")


if __name__ == "__main__":
    main()
