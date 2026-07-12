# Fixed-voltage electrochemistry diagnostic

Use this diagnostic before fitting a polarization curve when the collector
current changes materially during a current-controlled hold. It makes a fresh
copy of the AEMEC case, disables galvanostatic feedback, and applies a fixed
physical-anode collector voltage. The original case remains ready for a later
polarization scan or optimization run.

From the repository root, rebuild once because the diagnostic messages are in
the solver library, then create and run the copied case:

```bash
cd /root/work/AEMEC
./Allwmake

/root/work/AEMEC/.venv/bin/python \
  run/AEMEC/prepare_fixed_voltage_diagnostic.py \
  --output run/AEMEC-fixed-voltage \
  --voltage 1.5 \
  --end-time 12

cd run/AEMEC-fixed-voltage
make clear
make mesh
make srun
```

`1.5 V` and `12 s` are diagnostic defaults, not a calibrated operating point.
Use `--force` only to replace an existing diagnostic copy.

## What to inspect

Each time step writes these records:

- `AEMEC reaction diagnostic` for the cathode and anode catalyst layers:
  integrated Faradaic current, cathode H2 production rate, configured `j0`,
  activation-overpotential, Nernst-potential, temperature, water/H2 mole
  fractions, and gas volume fraction.
- `AEMEC electric diagnostic` for `phiAnion`: effective anion conductivity
  (`sigma`) and the mapped reaction-source field `J`.
- `Hydrogen crossover membrane diagnostic`: the cathode-CL `phiAnion` reaction
  current and the Faradaic H2 source used by the membrane crossover model.

For a consistent coupling, the magnitude of the cathode reaction current and
the `phiAnion` cathode-CL reaction current should agree, and the two derived
Faradaic H2 rates should agree. A nonzero fluid-side cathode H2 rate together
with a zero membrane-source rate identifies a mapping or update-order defect.

This AEM proof-of-concept uses an effective anion-potential field with constant
conductivity; it does not solve an explicit OH- concentration. Consequently,
the conductivity report verifies the configured effective field but cannot
diagnose hydroxide depletion.

If the fixed-voltage current still decays, compare the time histories in this
order: reaction current and clipping count; activation/Nernst potentials;
water and H2 fractions plus gas volume fraction; then `phiAnion` source and
membrane H2 source. Do not resume polarization optimization until those
quantities are stable and mutually consistent.
