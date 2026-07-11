# Membrane hydrogen crossover model

This note documents the AEM-electrolyzer crossover model configured in
`constant/phiAnion/regionProperties`. It is a proof-of-concept transport model
whose parameters need calibration before it is used for quantitative design or
optimization decisions.

## Phases and concentrations

The solver keeps these quantities distinct:

- `H2.hydrogen` is gas-phase H2 in the cathode `hydrogen` phase.
- `H2.oxygen` is gas-phase H2 in the anode `oxygen` phase. It begins as a
  trace component and receives crossover H2.
- `cH2` is dissolved H2 in the membrane, in `mol/m3`.

At each catalyst-layer/membrane interface the configured `henry` mode imposes
local equilibrium between gas partial pressure and the membrane dissolved
concentration:

```text
cH2_interface [mol/m3] = henryCoefficient [mol/(m3 Pa)] * pH2 [Pa]
```

The provided `7.8e-6 mol/(m3 Pa)` is only the former case's nominal scaling
(`0.78 mol/m3` at `1 bar`), not a validated H2-in-KOH solubility. `fixed`
interface mode is available for a prescribed dissolved concentration.

The two-phase model's existing interfacial mass transfer is retained for water
evaporation/condensation. It does not contain a separate dissolved-H2 species
in the liquid water/KOH phase, so this implementation is a local Henry
interface closure rather than a finite-rate gas-to-liquid H2 transfer model.

## Membrane transport and outputs

The membrane concentration equation contains a Faradaic cathode-side source,
diffusion, electro-osmotic-drag advection, optional through-membrane
convection, and an anode-side release. The Faradaic source is

```text
h2Generation = abs(J)/(2 F)    [mol/(m3 s)]
```

where `J` is the local volumetric electrochemical current source in
`cathodeCL`. This is the finite-volume form of the catalyst-layer/membrane
interface source.

`diffusivityModel` selects the effective diffusivity:

- `constant`: `D_eff = DelecH2`
- `porosityTortuosity`: `D_eff = DelecH2 * epsilonM / tau`
- `bruggeman`: `D_eff = DelecH2 * epsilonM^bruggemanExponent`

The current configuration uses `porosityTortuosity`. `DelecH2` is the base
diffusivity in `m2/s`.

The model writes the following fields, all in `mol/(m2 s)` and reported as
non-negative cathode-to-anode magnitudes:

- `JH2Diff`: diffusive contribution
- `JH2Drag`: electro-osmotic-drag contribution,
  `|i|/F * xi * cH2_cathode_interface / cElec`
- `JH2Conv`: configured convective contribution, `|UMembrane| * cH2`
- `JH2Cross`: sum of those three contributions

`xi` is currently a constant. The `dragModel` dictionary switch is deliberately
limited to `constant`; hydration- and temperature-dependent correlations are
future extensions. `UMembrane` is a user-specified, uniform velocity in `m/s`
and defaults to zero. It is an optional convective closure, not a pressure
solver: keep it zero unless a validated pressure/permeability model justifies
the value.

## Conservative species coupling

The anode-side membrane release rate is integrated in `mol/s`. It is then
placed on the cathode and anode catalyst-layer zones as `h2CathodeDmdt` and
`h2AnodeDmdt`, respectively. `electroChemicalReaction` maps these fields to
the two fluid meshes through the shared parent-cell map, multiplies by the H2
molar mass, and adds them to the gas mass-fraction equations. Thus the H2
source is negative in the cathode and positive in the anode with equal global
molar magnitude, including in decomposed runs; it never depends on matching
local cell numbers.

The log prints both the optimizer objective and the balance check:

```text
Hydrogen crossover objective: anode gas source rate = ... mol/s
Hydrogen crossover conservation: cathode H2 source = ... mol/s, ... imbalance = ... mol/s
```

The optimizer parses the first line as the crossover objective. The imbalance
should be close to round-off.

## Deliberate limitations

- No explicit dissolved-H2 equation exists in the liquid water/KOH phase.
- H2/O2 recombination and parasitic electrochemical consumption in the
  membrane are omitted.
- The optional convective term does not yet calculate membrane permeability
  from a pressure equation.
- The membrane is still represented by its resolved through-plane cells; the
  configured mesh thickness must be the swollen thickness selected for the
  case.
- KOH is represented by the fed liquid phase. Explicit KOH concentration,
  hydroxide concentration, and water crossover balances require additional
  species and electrolyte-property models.
