# Cathode-fed AEM electrolyzer case

This case uses terminology and reaction direction for a cathode-fed anion
exchange membrane (AEM) electrolyzer.

```text
liquid water → cathode / H2-product side → OH- through membrane → anode / O2-product side
```

| Physical role | Fluid region | Electron region | Catalyst-layer zone |
| --- | --- | --- | --- |
| Cathode: H2 generation | `cathode` | `phiECathode` | `cathodeCL` |
| Anode: O2 generation | `anode` | `phiEAnode` | `anodeCL` |
| Anion carrier | — | `phiAnion` | cathodeCL + electrolyte + anodeCL |

Both fluid regions are two-phase:

- `cathode` is a liquid-water/gas region and receives liquid water at
  `cathodeInlet` with `U.water = (0.001 0 0) m/s`. The phase named `gas`
  contains the H2 and H2O-vapour species.
- `anode` is a liquid-water/gas region and also receives liquid aqueous KOH at
  `anodeInlet` with `U.water = (0.001 0 0) m/s`. Its `gas` phase contains O2,
  H2O vapour, and crossover H2, which leave through `anodeOutlet`.

At this level of the solver, the water phase is the aqueous-KOH carrier.
There is no independent KOH mass-fraction, electrolyte-density, or solute
transport equation yet, so do not interpret the liquid phase as pure water in
a quantitative electrolyte-balance calculation.

The reaction dictionaries use the AEM-electrolyzer source signs:

```text
cathodeCL:  2 H2O + 2 e- → H2 + 2 OH-
anodeCL:    2 OH- → 1/2 O2 + H2O + 2 e-
```

`phiAnion` is an effective anion-conducting potential region. It uses the
solver's generic ionic-potential formulation with an `anion` dictionary key;
it does **not** solve an OH- concentration field. Calibrate the membrane
conductivity and add explicit hydroxide/water transport before using results
for quantitative design decisions. Its initial effective conductivities are
11.4 for the membrane and 1.10 for each porous catalyst layer; they reproduce
the former initialization at 313.15 K and are not an AEM-material calibration.
The inherited Nafion dissolved-water and hydration (`lambda`) models are
disabled for this case.

For the proof-of-concept run, both Butler–Volmer dictionaries use a numerical
`jMax = 2e9 A/m3` safeguard and a bounded exponential argument. For the 20 um
catalyst layers this corresponds to 4 A/cm2, above the intended curve range.
It prevents an uncalibrated initial potential from destabilizing the flow
solver; it is not a physical limiting-current model and must be replaced by
validated kinetics and transport limitations before quantitative use.

Hydrogen crossover is defined from `cathodeCL` to `anodeCL`; the anode gas
phase therefore contains a trace `H2` component. The crossover model reports
separate `JH2Diff`, `JH2Drag`, and `JH2Cross` fields. A connected
dissolved-H2 field spans `cathodeCL`, the membrane, and `anodeCL`; each CL is
coupled locally and bidirectionally to its Eulerian gas phase through
`kLa*(cH2 - H*p*XH2)`. The default Franz-style partition places Faradaic H2 in
the dissolved field and removes that same local amount from the direct gas
reaction source, preventing double production. See
[the coupled crossover model](docs/membrane-crossover.md) for the equations,
dictionary contract, and remaining limitations. Validate the storage,
diffusion, drag, Henry, and CL mass-transfer parameters plus the printed
conservation line before running an optimization.

The default `phiEAnode` configuration on the `crossover` branch is a 20 s
galvanostatic experiment at `1 A/cm2` (`-10000 A/m2` in the solver's signed
electrolysis convention). The collector voltage starts from a 2 V guess and is
adjusted from the post-solve boundary-current measurement. The optimization
adapter still uses the retained `1.3--2.3 V` table: it explicitly disables
galvanostatic feedback in each scratch case and interpolates the voltage and
crossover rate at `1 A/cm2`. See [polarization control](docs/polarization-control.md).

The current branch uses a non-isothermal, one-temperature
Eulerian--Eulerian model. `constant/cellProperties` enables the global
conjugate-energy equation with `solveEnergy true`, while the anode and cathode
`regionProperties` dictionaries set `thermalEquilibrium true`. Gas and liquid
retain separate phase fractions, velocities, mass fluxes, and species, but
share the spatially varying parent-mesh temperature. Both phases contribute
to thermal storage, convection, conductivity, gravity/pressure/kinetic work,
and reaction heat. The independent dilute-gas enthalpy equation and its
`residualAlphaEnergy` regularization are not used. Both fluid regions use
`basicTwoPhaseSystem`; interfacial H2O evaporation/condensation and its latent
heat source are disabled. Electrochemically consumed or produced water is
still applied directly to the liquid-water phase, while gas products remain
separate dispersed phases. The phase momentum, pressure, continuity, and
gravity physics remain active, but `includeMechanicalWorkInHeatSource false`
prevents their pressure-, kinetic-, and gravitational-work terms from being
deposited as heat in the global temperature equation.

`isothermalTemperature 313.15` remains as the fallback value used only when
`solveEnergy` is explicitly disabled for an electrochemical diagnostic.

## Run

```bash
make mesh
make srun
```

Then, from the repository root, extract the final point from each voltage hold:

```bash
python3 visualization/polarization_curve.py \
    --log run/AEMEC/log.run \
    --scan-mode voltage \
    --hold-duration 15
```

Generate the voltage-loss decomposition and hydrogen-crossover figures from
the same accepted polarization points with:

```bash
python3 visualization/aemec_diagnostics.py \
    --log run/AEMEC/log.run \
    --scan-mode voltage
```

This writes `voltage_decomposition.{csv,png}` and
`hydrogen_crossover.{csv,png}` in `visualization/output/`. The voltage figure
contains
the cell voltage, lowest-current reversible baseline, current-dependent Nernst
shift (a concentration/transport proxy), anode and cathode activation losses,
electronic and anion ohmic losses, and an explicit unresolved closure curve.
Ohmic voltage is evaluated as `integral(i^2/sigma dV)/I`; activation and Nernst
terms are current-weighted over their catalyst layers. The crossover figure
shows Faradaic H2 generation, the initially dissolved share, signed cathode
dissolved-to-gas transfer, net cathode gas release, and anode gas release
together with crossover as a percentage of instantaneous production.
That percentage is omitted near open circuit if crossover is supplied by the
existing dissolved-H2 inventory and exceeds instantaneous production.

After changing C++ source, rebuild once with `./src/Allwmake` before these
case commands. You do not rebuild the solver between optimization trials;
each trial remeshes and runs the case with its selected membrane thickness.

For MPI, set `NPROCS`, then run `make decompose`, `make parallel`, and
`make run`.

Legacy meshes remain incompatible with the renamed regions, and existing
result directories are incompatible with the renamed cathode gas phase.
Regenerate the mesh after rebuilding the solver.

For an isolated single-voltage diagnostic, see
[the diagnostic guide](docs/fixed-voltage-diagnostic.md).
