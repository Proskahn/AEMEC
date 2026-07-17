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
separate `JH2Diff`, `JH2Drag`, `JH2Conv`, and `JH2Cross` fields, then removes
the resulting molar rate from cathode gas H2 and adds the identical rate to
anode gas H2. See [the membrane-crossover model](docs/membrane-crossover.md)
for the assumptions, dictionary contract, and remaining limitations. Validate
diffusion/drag parameters and the printed conservation line at a reference
thickness before running an optimization.

The default `phiEAnode` configuration is a potentiostatic polarization scan.
It holds `1.3, 1.5, 1.7, 1.9, 2.1,` and `2.3 V` for 15 s each and records the
resulting collector current density. The optimizer separately enables
galvanostatic mode for each isolated trial. See
[polarization control](docs/polarization-control.md).

The current branch uses a non-isothermal, one-temperature
Eulerian--Eulerian model. `constant/cellProperties` enables the global
conjugate-energy equation with `solveEnergy true`, while the anode and cathode
`regionProperties` dictionaries set `thermalEquilibrium true`. Gas and liquid
retain separate phase fractions, velocities, mass fluxes, and species, but
share the spatially varying parent-mesh temperature. Both phases contribute
to thermal storage, convection, conductivity, gravity/pressure/kinetic work,
and reaction heat. The independent dilute-gas enthalpy equation and its
`residualAlphaEnergy` regularization are not used. The summed interfacial film
heat supplies the latent-heat source/sink without retaining an internal
gas-to-liquid sensible-heat exchange term.

`isothermalTemperature 313.15` remains as the fallback value used only when
`solveEnergy` is explicitly disabled for an electrochemical diagnostic.

## Run

```bash
make mesh
make srun
```

Then, from the repository root, extract the final point from each voltage hold:

```bash
python3 visualization/polarized_curve.py \
    --log run/AEMEC/log.run \
    --scan-mode voltage \
    --hold-duration 15
```

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
