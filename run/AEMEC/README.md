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
- `anode` is a water/oxygen region. Its inlet velocity is zero in this
  cathode-fed configuration; oxygen and water leave through `anodeOutlet`.

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

Hydrogen crossover is defined from `cathodeCL` to `anodeCL`. Validate the
diffusion/drag parameters and mass balance at a reference thickness before
running an optimization.

## Run

```bash
make mesh
make srun
```

For MPI, set `NPROCS`, then run `make decompose`, `make parallel`, and
`make run`.

All legacy meshes and result directories are incompatible with these renamed
regions. Regenerate the mesh after rebuilding the solver.
