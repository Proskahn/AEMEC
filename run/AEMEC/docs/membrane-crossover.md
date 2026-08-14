# Coupled dissolved-hydrogen and gas-crossover model

This note documents the proof-of-concept AEM-electrolyzer model configured in
`constant/phiAnion/regionProperties`. Its equation structure follows the
dynamic dissolved-gas balance of Franz et al. (2023), while its AEM diffusion
and electro-osmotic-drag closures follow Klinger et al. (2025). All material
and catalyst-layer transfer parameters still require calibration before
quantitative design or optimization.

## Continua and concentrations

The solver keeps three quantities distinct:

- `cathode/H2.gas` is H2 in the cathode Eulerian gas phase.
- `anode/H2.gas` is H2 in the anode Eulerian gas phase.
- `phiAnion/cH2` is dissolved H2 in the connected hydrated-ionomer continuum
  comprising `cathodeCL`, the membrane, and `anodeCL`. Its unit is mol/m3 of
  absorbed electrolyte, not mol/m3 of bulk membrane.

Dissolved H2 is a transported scalar, not a third Eulerian fluid phase. The
gas and liquid phases retain their own Eulerian--Eulerian continuity,
momentum, and species equations in the fluid regions.

## Dissolved-H2 balance

The multidimensional equation is

```text
d(epsilonIon*cH2)/dt
  + div(Udrag*cH2)
  = div(DH2Eff*grad(cH2))
  + beta*GammaFaradaic
  - GammaDissolvedToGas
```

Equivalently, the dissolved-hydrogen molar flux is

```text
N_H2,diss = -DH2Eff*grad(cH2)
           + nDrag*cH2*iIon/(zIon*F*cH2O)
```

There is no independent hydraulic or bulk-convection contribution to this
flux.

where `beta` is `faradaicDissolvedFraction` and

```text
Udrag = nDrag*iIon/(zIon*F*cH2O)
GammaDissolvedToGas = kLa*(cH2 - cSat)
cSat = henryCoefficient*p*XH2
```

`GammaDissolvedToGas` is active only in the two catalyst layers. It is signed:
positive values desorb H2 from ionomer to pore gas, while negative values
absorb H2 from pore gas into the ionomer. Henry's law supplies the equilibrium
target; it is no longer imposed as a catalyst-layer Dirichlet concentration.
There is no production or dissolved-to-gas transfer term in the membrane bulk.

The default charge number is `zIon = -1` because hydroxide and its dragged
water move opposite to conventional ionic current. The dissolved-gas flux
contains only molecular diffusion and electro-osmotic drag; no membrane
convection term is included.

## Storage and diffusion

`epsilonIonH2` is the hydrated-ionomer/electrolyte fraction used in the
transient storage term. Separate values are configured for the membrane,
cathode CL, and anode CL. `diffusivityModel` selects

- `constant`: `DH2Eff = DelecH2`
- `porosityTortuosity`: `DH2Eff = epsilonIonH2*DelecH2/tau`
- `bruggeman`: `DH2Eff = epsilonIonH2^bruggemanExponent*DelecH2`

The current case uses `porosityTortuosity`. The configured volume fractions,
tortuosities, base diffusivity, Henry coefficient, drag coefficient, and both
`kLa` values are provisional.

## Conservative Faradaic partition and gas coupling

The cathode reaction class initially constructs the full Faradaic gas source

```text
GammaFaradaic = abs(J)/(2*F)
```

The crossover model partitions it locally:

```text
dissolved source       =  beta*GammaFaradaic
direct Faradaic gas    = (1-beta)*GammaFaradaic
cathode gas coupling   =  GammaDissolvedToGas - beta*GammaFaradaic
anode gas coupling     =  GammaDissolvedToGas
```

Thus `beta = 1` implements the Franz-style assumption that generated H2 first
enters the dissolved ionomer. `beta = 0` recovers direct Faradaic production
in the gas phase. An intermediate value is an empirical partition and must be
validated. No value generates H2 twice.

The local molar gas sources `h2CathodeDmdt` and `h2AnodeDmdt` are mapped to the
Eulerian gas species equations through shared parent-cell labels and converted
to mass sources using the H2 molar mass. This works for decomposed cases and
does not assume matching local cell indices between region meshes.

At every update the source-only conservation identity is

```text
cathode gas coupling + anode gas coupling + dissolved coupling = 0
```

The log reports this identity and the non-negative anode desorption rate used
by the optimizer:

```text
Hydrogen crossover objective: anode gas source rate = ... mol/s
Hydrogen dissolved-gas coupling conservation: ... imbalance = ... mol/s
```

## Written diagnostics

- `cH2`: dissolved concentration in the connected CL--membrane--CL ionomer.
- `epsilonIonH2`: local storage fraction.
- `DH2Eff`: local effective diffusivity.
- `h2DissolvedProduction`: Faradaic production assigned to dissolved H2.
- `h2MassTransferCoeff`: local CL `kLa`, zero in the membrane.
- `h2DissolvedToGas`: signed CL interphase transfer rate in mol/(m3 s).
- `cH2CathodeInterface`, `cH2AnodeInterface`: local equilibrium targets.
- `JH2Diff`, `JH2Drag`: transport-flux magnitudes in mol/(m2 s).
- `JH2Cross`: magnitude of the signed vector sum of diffusion and drag. It is
  not the sum of their separate magnitudes, so opposing contributions cancel.

## Deliberate limitations

- No dissolved-H2 equation exists in the bulk liquid-water/KOH phase. Transfer
  occurs directly between catalyst-layer ionomer and pore gas.
- The finite-rate transfer does not yet include interfacial-area or gas-volume
  availability factors. Both `kLa` values must therefore be treated as
  effective catalyst-layer coefficients.
- Dissolved O2, H2/O2 recombination, and parasitic electrochemical consumption
  are omitted.
- KOH, OH-, and water concentrations are not solved explicitly, so `cH2O`, ionic
  conductivity, and drag remain calibrated effective properties.
- The configured thickness must be the swollen membrane thickness.

Primary model references:

- T. Franz, G. Papakonstantinou, and K. Sundmacher, *Journal of Power Sources*
  559 (2023) 232582, https://doi.org/10.1016/j.jpowsour.2022.232582.
- A. Klinger et al., *Advanced Materials Interfaces* 12 (2025) 2400515,
  https://doi.org/10.1002/admi.202400515.
