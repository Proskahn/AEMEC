from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aemec_case import (
    AemecEvaluationConfig,
    AemecOpenFoamEvaluator,
    OptimizationError,
    VoltageHold,
    VoltageSweep,
    configure_voltage_sweep,
    copy_clean_case,
    interpolate_voltage_objective,
    parse_voltage_sweep_samples,
    rewrite_block_mesh_thickness,
)


ROOT = Path(__file__).resolve().parents[1]


def copy_controls(case: Path) -> None:
    (case / "constant/phiEAnode").mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(exist_ok=True)
    shutil.copy2(
        ROOT / "run/AEMEC/constant/phiEAnode/regionProperties",
        case / "constant/phiEAnode/regionProperties",
    )
    for name in ("controlDict.run", "controlDict"):
        shutil.copy2(ROOT / "run/AEMEC/system" / name, case / "system" / name)


class AemecCaseTests(unittest.TestCase):
    def test_static_case_preflight_passes(self) -> None:
        result = subprocess.run(
            [str(ROOT / "run/AEMEC/check_case.py"), "--static"],
            cwd=ROOT / "run/AEMEC",
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_case_uses_the_cathode_fed_aem_region_contract(self) -> None:
        case = ROOT / "run/AEMEC"
        regions = (case / "constant/regionProperties").read_text(encoding="utf-8")
        cell_properties = (case / "constant/cellProperties").read_text(
            encoding="utf-8"
        )
        cathode = (case / "constant/cathode/regionProperties").read_text(encoding="utf-8")
        anode = (case / "constant/anode/regionProperties").read_text(encoding="utf-8")
        crossover = (case / "constant/phiAnion/regionProperties").read_text(encoding="utf-8")
        cathode_reaction = (
            case / "constant/cathode/combustionProperties.gas"
        ).read_text(encoding="utf-8")
        anode_reaction = (case / "constant/anode/combustionProperties.gas").read_text(encoding="utf-8")
        cathode_diffusivity = (case / "constant/cathode/diffusivityModel.gas").read_text(encoding="utf-8")
        anode_diffusivity = (case / "constant/anode/diffusivityModel.gas").read_text(encoding="utf-8")
        cathode_porous = (case / "constant/cathode/porousZones").read_text(encoding="utf-8")
        anode_porous = (case / "constant/anode/porousZones").read_text(encoding="utf-8")
        cathode_water_velocity = (case / "0.orig/cathode/U.water").read_text(encoding="utf-8")
        cathode_gas_velocity = (case / "0.orig/cathode/U.gas").read_text(encoding="utf-8")
        cathode_hydrogen = (case / "0.orig/cathode/H2.gas").read_text(encoding="utf-8")
        cathode_water_vapour = (case / "0.orig/cathode/H2O.gas").read_text(encoding="utf-8")
        anode_water_velocity = (case / "0.orig/anode/U.water").read_text(encoding="utf-8")
        anode_thermo = (case / "constant/anode/thermophysicalProperties.gas").read_text(encoding="utf-8")
        cathode_thermo = (case / "constant/cathode/thermophysicalProperties.gas").read_text(encoding="utf-8")
        anode_hydrogen = (case / "0.orig/anode/H2.gas").read_text(encoding="utf-8")
        anode_controller = (case / "constant/phiEAnode/regionProperties").read_text(encoding="utf-8")
        anode_collector = (case / "system/phiEAnode/createPatchDict").read_text(encoding="utf-8")
        cathode_collector = (case / "system/phiECathode/createPatchDict").read_text(encoding="utf-8")
        electrolyte_temperature = (case / "0.orig/electrolyte/T").read_text(encoding="utf-8")
        interconnect_temperature = (case / "0.orig/interconnect/T").read_text(encoding="utf-8")
        ionic_potential = (case / "0.orig/phiAnion/phi").read_text(encoding="utf-8")
        anode_solution = (case / "system/anode/fvSolution").read_text(encoding="utf-8")
        cathode_solution = (case / "system/cathode/fvSolution").read_text(encoding="utf-8")

        self.assertIn("fluid (anode cathode)", regions)
        self.assertIn("electric (phiECathode phiEAnode phiAnion)", regions)
        self.assertIn("phases (gas water)", cathode)
        self.assertIn("phases (gas water)", anode)
        self.assertIn("continuous water;", cathode)
        self.assertIn("continuous water;", anode)
        self.assertIn("solveEnergy             true;", cell_properties)
        self.assertIn("thermalEquilibrium true;", cathode)
        self.assertIn("thermalEquilibrium true;", anode)
        self.assertNotIn("residualAlphaEnergy", cathode)
        self.assertNotIn("residualAlphaEnergy", anode)
        self.assertIn("pressureWorkAlphaLimit 0.05;", anode_thermo)
        self.assertIn("pressureWorkAlphaLimit 0.05;", cathode_thermo)
        self.assertRegex(anode_solution, r"\biDmdt\s+0\.5\s*;")
        self.assertRegex(cathode_solution, r"\biDmdt\s+0\.5\s*;")
        self.assertIn("sourceZone      cathodeCL", crossover)
        self.assertIn("sinkZone        anodeCL", crossover)
        self.assertIn("H2O    -2", cathode_reaction)
        self.assertIn("H2O     1", anode_reaction)
        self.assertIn("cathodeChannel\n{", cathode_diffusivity)
        self.assertIn("anodeChannel\n{", anode_diffusivity)
        self.assertIn("cellZone        cathodeMPL", cathode_porous)
        self.assertIn("cellZone        anodeMPL", anode_porous)
        self.assertIn("uniform (0.001 0 0)", cathode_water_velocity)
        self.assertIn("object      U.gas;", cathode_gas_velocity)
        self.assertIn("internalField   uniform (0 0 0);", cathode_gas_velocity)
        self.assertNotRegex(cathode_gas_velocity, r"\bvalue\s+internalField\s*;")
        self.assertIn("phi             phi.gas;", cathode_gas_velocity)
        self.assertIn("object          yH2;", cathode_hydrogen)
        self.assertIn("object          yH2O;", cathode_water_vapour)
        self.assertFalse((case / "0.orig/cathode/H2.hydrogen").exists())
        self.assertFalse((case / "0.orig/cathode/H2O.hydrogen").exists())
        self.assertIn("uniform (0.001 0 0)", anode_water_velocity)
        self.assertIn("H2", anode_thermo)
        self.assertIn("object          yH2;", anode_hydrogen)
        self.assertFalse((case / "0.orig/anode/H2.oxygen").exists())
        self.assertFalse((case / "0.orig/anode/H2O.oxygen").exists())
        self.assertIn("cathodeFluidRegion  cathode;", crossover)
        self.assertIn("anodeFluidRegion    anode;", crossover)
        self.assertIn("diffusivityModel    porosityTortuosity;", crossover)
        self.assertIn("cathodeInterface", crossover)
        self.assertEqual(crossover.count("gasPhase            gas;"), 2)
        self.assertIn("anodeInterface", crossover)
        self.assertIn("UMembrane       (0 0 0);", crossover)
        self.assertRegex(crossover, r"currentBalance\s*\{\s*active\s+true\s*;")
        self.assertIn("potentialRelaxation     0.25;", crossover)
        self.assertIn("maxFieldPotentialStep   0.02;", crossover)
        self.assertIn("internalField   uniform 3.0;", ionic_potential)
        self.assertIn("relax           1.0;", anode_reaction)
        self.assertIn("relax           1.0;", cathode_reaction)
        self.assertIn("polarizationCurve", anode_controller)
        self.assertRegex(
            anode_controller,
            r"galvanostatic\s*\{\s*active\s+false\s*;",
        )
        self.assertRegex(
            anode_controller,
            r"voltage\s*\{\s*type\s+table\s*;",
        )
        self.assertRegex(
            anode_controller,
            r"polarizationCurve\s*\{\s*active\s+false\s*;",
        )
        self.assertIn("(15.001  1.4)", anode_controller)
        self.assertIn("(165     2.3)", anode_controller)
        self.assertIn(
            "targets                     (0 -2000 -4000 -6000 -8000 -10000 -12000 -14000 -16000 -18000 -20000);",
            anode_controller,
        )
        self.assertIn("minimumHoldDuration         15;", anode_controller)
        self.assertIn("stabilitySamples            5;", anode_controller)
        self.assertRegex(anode_collector, r"name\s+interconnect0\s*;")
        self.assertRegex(anode_collector, r"set\s+interconnect0\s*;")
        self.assertRegex(cathode_collector, r"name\s+interconnect1\s*;")
        self.assertRegex(cathode_collector, r"set\s+interconnect1\s*;")
        self.assertIn("electrolyte_to_anode", electrolyte_temperature)
        self.assertIn("electrolyte_to_cathode", electrolyte_temperature)
        self.assertIn("interconnect_to_anode", interconnect_temperature)
        self.assertIn("interconnect_to_cathode", interconnect_temperature)

    def test_polarization_controller_requests_clean_stop_after_last_point(self) -> None:
        controller_source = (
            ROOT
            / "src/libSrc/fuelCellSystems/regions/electric/electric.C"
        ).read_text(encoding="utf-8")
        self.assertIn("time().stopAt(Time::saWriteNow);", controller_source)

    def test_ionic_poisson_solve_uses_one_reference_cell(self) -> None:
        electric_source = (
            ROOT
            / "src/libSrc/fuelCellSystems/regions/electric/electric.C"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "phiEqn->setReference(referenceCell, referenceValue, true);",
            electric_source,
        )
        self.assertIn(
            "shiftPotential(selectedMeanPotential - meanPotential());",
            electric_source,
        )
        self.assertIn("fvm::Sp(dJdPhi_, phi_)", electric_source)
        self.assertIn("Ionic Newton potential update", electric_source)

        butler_volmer = (
            ROOT
            / "src/libSrc/fuelCellSystems/activationOverpotentialModels/ButlerVolmer/ButlerVolmer.C"
        ).read_text(encoding="utf-8")
        self.assertIn("dSIdPhiAnion[anionId] = dSourceDphi;", butler_volmer)
        self.assertNotRegex(
            electric_source,
            r"for\s*\([^)]*nCells\(\)[^)]*\)\s*\{[^}]*setReference",
        )

        constant_sigma = (
            ROOT
            / "src/libSrc/fuelCellSystems/sigmaModels/constantSigma/constantSigma.C"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "dimCurrent*dimCurrent*dimTime/(dimEnergy*dimLength)",
            constant_sigma,
        )
        self.assertNotIn("dimless/dimLength", constant_sigma)

    def test_two_phase_energy_keeps_composition_and_thermo_consistent(self) -> None:
        source = ROOT / "src/libSrc/fuelCellSystems"
        species_equations = (
            source / "solvers/twoPhaseSystem/YEqns.H"
        ).read_text(encoding="utf-8")
        energy_equation = (
            source / "solvers/twoPhaseSystem/EEqn.H"
        ).read_text(encoding="utf-8")
        multicomponent = (
            source
            / "phaseModel/MultiComponentPhaseModel/MultiComponentPhaseModel.C"
        ).read_text(encoding="utf-8")
        heat_transfer = (
            source
            / "PhaseSystems/TwoResistanceHeatTransferPhaseSystem/TwoResistanceHeatTransferPhaseSystem.C"
        ).read_text(encoding="utf-8")
        anisothermal = (
            source / "phaseModel/AnisothermalPhaseModel/AnisothermalPhaseModel.C"
        ).read_text(encoding="utf-8")
        multicomponent_phase = (
            source
            / "phaseModel/MultiComponentPhaseModel/MultiComponentPhaseModel.C"
        ).read_text(encoding="utf-8")
        phase_system = (source / "phaseSystem/phaseSystem.C").read_text(
            encoding="utf-8"
        )
        fluid_region = (source / "regions/fluid/fluid.C").read_text(
            encoding="utf-8"
        )
        global_energy = (ROOT / "src/appSrc/EEqns.H").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("phase1_.correctThermo();", species_equations)
        self.assertNotIn("phase2_.correctThermo();", species_equations)
        self.assertIn("phase1_.correctComposition();", species_equations)
        self.assertIn("phase2_.correctComposition();", species_equations)
        self.assertIn("thermo1.he(thermo1.p(), T1)", species_equations)
        self.assertLess(
            species_equations.index("phase1_.correctComposition();"),
            species_equations.index("thermo1.he(thermo1.p(), T1)"),
        )
        self.assertLess(
            species_equations.index("thermo1.he(thermo1.p(), T1)"),
            species_equations.index("thermo1.correct();"),
        )
        self.assertNotIn("phase1_.correctElectrochemistry();", species_equations)
        self.assertNotIn("phase2_.correctElectrochemistry();", species_equations)
        self.assertIn(
            "leave the two sides of the same Faradaic reaction",
            species_equations,
        )
        self.assertIn("thermo.correct();", energy_equation)
        self.assertIn("!thermalEquilibrium()", energy_equation)
        self.assertIn("correctComposition();", multicomponent)
        self.assertLess(
            multicomponent.index("//- Normalize"),
            multicomponent.index("BasePhaseModel::correctThermo();"),
        )
        self.assertIn("K/Cpv*he - fvm::Sp(K/Cpv, he)", heat_transfer)
        local_energy = anisothermal[
            anisothermal.index("::heEqn()") : anisothermal.index("::heQdot()")
        ]
        self.assertIn("tEEqn.ref() += filterPressureWork", local_energy)
        self.assertIn("this->thermo().p()*fvc::ddt(alpha)", local_energy)
        explicit_capacity = "fvc::ddt(residualAlphaEnergy*rho, he)"
        implicit_capacity = "fvm::ddt(residualAlphaEnergy*rho, he)"
        self.assertIn(explicit_capacity, anisothermal)
        self.assertIn(implicit_capacity, anisothermal)
        self.assertLess(
            anisothermal.index(implicit_capacity),
            anisothermal.index(explicit_capacity),
        )
        self.assertIn('readIfPresent("thermalEquilibrium"', phase_system)
        self.assertIn("phases_->thermalEquilibrium()", fluid_region)
        self.assertIn("rhoCp += tPhaseRhoCp().primitiveField();", fluid_region)
        self.assertIn("rhoCpPhi += tPhaseRhoCpPhi().primitiveField();", fluid_region)
        self.assertIn("kF += tPhaseKappa().primitiveField();", fluid_region)
        self.assertIn("thermo.he() = thermo.he(thermo.p(), T0).ref();", fluid_region)
        self.assertIn("thermo.T() = T0;", fluid_region)
        self.assertIn("if (this->thermalEquilibrium())", heat_transfer)
        self.assertIn("qInterface0.rmap(qInterface, cellMap);", heat_transfer)
        self.assertIn("fvm::Sp(fvc::ddt(rhoCpCell), TCell)", global_energy)
        self.assertIn("fvm::Sp(fvc::div(rhoCpPhiCell), TCell)", global_energy)
        self.assertIn(
            'dimensionedScalar("diff", dimViscosity, SMALL)',
            multicomponent_phase,
        )
        self.assertNotIn(
            "this->muEff()->dimensions()/dimDensity",
            multicomponent_phase,
        )

        clean_script = (ROOT / "src/libSrc/Allwclean").read_text(encoding="utf-8")
        self.assertIn("wclean libso thermoTools", clean_script)
        self.assertNotIn("wclean libso thermalTools", clean_script)

    def test_thickness_rewrite_preserves_other_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "blockMeshDict"
            shutil.copy2(ROOT / "run/AEMEC/system/blockMeshDict", mesh)
            update = rewrite_block_mesh_thickness(mesh, 50.0)
            self.assertEqual(update.old_thickness_um, 30.0)
            self.assertEqual(update.new_thickness_um, 50.0)
            self.assertIn("0.025", mesh.read_text(encoding="utf-8"))

    def test_optimizer_configures_complete_voltage_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            copy_controls(case)
            sweep = configure_voltage_sweep(case, AemecEvaluationConfig())
            self.assertEqual(len(sweep.holds), 11)
            self.assertEqual(sweep.outer_iterations, 1650)
            self.assertEqual(sweep.delta_t_s, 0.1)
            self.assertEqual(sweep.holds[1].voltage_v, 1.4)
            controls = (case / "system/controlDict.run").read_text(encoding="utf-8")
            controller = (case / "constant/phiEAnode/regionProperties").read_text(encoding="utf-8")
            self.assertRegex(controls, r"endTime\s+165\s*;")
            self.assertRegex(controls, r"writeInterval\s+165\s*;")
            self.assertRegex(
                controller,
                r"polarizationCurve\s*\{[\s\S]*?active\s+false\s*;",
            )
            self.assertRegex(
                controller,
                r"galvanostatic\s*\{\s*active\s+false\s*;",
            )

    def test_voltage_sweep_interpolates_at_target_current(self) -> None:
        config = AemecEvaluationConfig(stability_samples=3)
        log = """
Time = 1
Controlled boundary current (A) at x: signed = -0.64, magnitude = 0.64, current density = -8000 A/m2, voltage = 1.7
Hydrogen crossover objective: anode gas source rate = 3.0e-5 mol/s
Time = 2
Controlled boundary current (A) at x: signed = -0.64, magnitude = 0.64, current density = -8010 A/m2, voltage = 1.7
Hydrogen crossover objective: anode gas source rate = 3.01e-5 mol/s
Time = 3
Controlled boundary current (A) at x: signed = -0.64, magnitude = 0.64, current density = -8000 A/m2, voltage = 1.7
Hydrogen crossover objective: anode gas source rate = 3.0e-5 mol/s
Time = 4
Controlled boundary current (A) at x: signed = -0.96, magnitude = 0.96, current density = -12000 A/m2, voltage = 1.8
Hydrogen crossover objective: anode gas source rate = 5.0e-5 mol/s
Time = 5
Controlled boundary current (A) at x: signed = -0.96, magnitude = 0.96, current density = -12010 A/m2, voltage = 1.8
Hydrogen crossover objective: anode gas source rate = 5.01e-5 mol/s
Time = 6
Controlled boundary current (A) at x: signed = -0.96, magnitude = 0.96, current density = -12000 A/m2, voltage = 1.8
Hydrogen crossover objective: anode gas source rate = 5.0e-5 mol/s
End
"""
        samples = parse_voltage_sweep_samples(log)
        self.assertEqual(samples[0].current_density_a_m2, -8000.0)
        sweep = VoltageSweep(
            (VoltageHold(0.0, 3.0, 1.7), VoltageHold(3.001, 6.0, 1.8)),
            delta_t_s=1.0,
            outer_iterations=6,
        )
        objective = interpolate_voltage_objective(log, config, sweep)
        self.assertAlmostEqual(objective.cell_voltage_v, 1.75)
        self.assertAlmostEqual(objective.crossover_rate_mol_s, 4.0e-5)
        self.assertAlmostEqual(objective.interpolation_fraction, 0.5)

    def test_copy_refuses_to_delete_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "case"
            source.mkdir()
            marker = source / "keep"
            marker.write_text("safe", encoding="utf-8")
            with self.assertRaises(OptimizationError):
                copy_clean_case(source, source)
            self.assertEqual(marker.read_text(encoding="utf-8"), "safe")

    def test_mocked_evaluator_runs_the_aemec_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            copy_controls(source)
            shutil.copy2(ROOT / "run/AEMEC/system/blockMeshDict", source / "system/blockMeshDict")
            (source / "mesh.py").write_text(
                """from pathlib import Path
for region in ('', 'anode', 'cathode', 'electrolyte', 'interconnect', 'phiECathode', 'phiEAnode', 'phiAnion'):
    base = Path('constant') if not region else Path('constant') / region
    target = base / 'polyMesh/points'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('mesh')
""",
                encoding="utf-8",
            )
            (source / "solver.py").write_text(
                """voltages = [1.3 + 0.1 * index for index in range(11)]
for index, voltage in enumerate(voltages, start=1):
    end = 15.0 * index
    current_density = -(voltage - 1.3) * 18000.0
    for offset in (0.4, 0.3, 0.2, 0.1, 0.0):
        print(f'Time = {end - offset:.1f}')
        print(f'Controlled boundary current (A) at x: signed = -0.8, magnitude = 0.8, current density = {current_density} A/m2, voltage = {voltage}')
        print(f'Hydrogen crossover objective: anode release rate = {voltage * 1e-5} mol/s')
print('End')
""",
                encoding="utf-8",
            )
            evaluator = AemecOpenFoamEvaluator(source, root / "work/case", root / "logs", (sys.executable, "mesh.py"), (sys.executable, "solver.py"), AemecEvaluationConfig(), 10.0)
            result = evaluator.evaluate(0, 40.0)
            self.assertAlmostEqual(result.objective.cell_voltage_v, 1.8555555556)
            self.assertAlmostEqual(
                result.objective.crossover_rate_mol_s, 1.8555555556e-5
            )
            self.assertTrue(result.solver_log.is_file())


if __name__ == "__main__":
    unittest.main()
