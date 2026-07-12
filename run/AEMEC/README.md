# Cathode-fed AEM electrolyzer case

This case uses terminology and reaction direction for a cathode-fed anion
exchange membrane (AEM) electrolyzer.

```text
liquid water → cathode / hydrogen side → OH- through membrane → anode / oxygen side
```

| Physical role | Fluid region | Electron region | Catalyst-layer zone |
| --- | --- | --- | --- |
| Cathode: H2 generation | `cathode` | `phiECathode` | `cathodeCL` |
| Anode: O2 generation | `anode` | `phiEAnode` | `anodeCL` |
| Anion carrier | — | `phiAnion` | cathodeCL + electrolyte + anodeCL |

Both fluid regions are two-phase:

- `cathode` is a water/hydrogen region and receives liquid water at
  `cathodeInlet` with `U.water = (0.01 0 0) m/s`.
- `anode` is a water/oxygen region and also receives liquid aqueous KOH at
  `anodeInlet` with `U.water = (0.01 0 0) m/s`; oxygen, water, and any
  crossover H2 leave through `anodeOutlet`.

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
`jMax = 5e8 A/m3` ceiling and a bounded exponential argument. These settings
prevent an uncalibrated initial potential from destabilizing the flow solver;
replace them with validated kinetic parameters before quantitative use.

Hydrogen crossover is defined from `cathodeCL` to `anodeCL`; the anode oxygen
phase therefore contains a trace `H2` component. The crossover model reports
separate `JH2Diff`, `JH2Drag`, `JH2Conv`, and `JH2Cross` fields, then removes
the resulting molar rate from cathode gas H2 and adds the identical rate to
anode gas H2. See [the membrane-crossover model](docs/membrane-crossover.md)
for the assumptions, dictionary contract, and remaining limitations. Validate
diffusion/drag parameters and the printed conservation line at a reference
thickness before running an optimization.

The default `phiEAnode` configuration is a stable-point polarization scan,
not a fixed-time voltage/current ramp. It advances each current target only
after the post-solve collector current is on target and current/voltage are
stable. Points at either voltage bound are flagged and not accepted. See
[polarization control](docs/polarization-control.md). The optimizer disables
this multi-target mode in its isolated trial case and retains one direct target
hold per CFD evaluation.

## Run

```bash
make mesh
make srun
```

After changing C++ source, rebuild once with `./src/Allwmake` before these
case commands. You do not rebuild the solver between optimization trials;
each trial remeshes and runs the case with its selected membrane thickness.

For MPI, set `NPROCS`, then run `make decompose`, `make parallel`, and
`make run`.

All legacy meshes and result directories are incompatible with these renamed
regions. Regenerate the mesh after rebuilding the solver.
