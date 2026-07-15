for (int Ecorr=0; Ecorr<nEnergyCorrectors; Ecorr++)
{
    phasePairKey pair(phase1_.name(), phase2_.name());
    phaseModel& phase = phaseModels_[pair.other(continuous_)];

    const volScalarField& alpha = phase;

    volVectorField& U = phase.URef();

    rhoThermo& thermo = phase.thermoRef();

    volScalarField& rho = thermo.rho();

    autoPtr<phaseSystem::heatTransferTable>
        heatTransferPtr(heatTransfer());

    phaseSystem::heatTransferTable&
        heatTransfer = heatTransferPtr();

    if (!forcedIsothermal() && !phase.isothermal())
    {
        fvScalarMatrix EEqn
        (
            phase.heEqn()
         ==
           *heatTransfer[phase.name()]
          + alpha*rho*(U&g)
          + fvOptions(alpha, rho, thermo.he())
        );

        EEqn.relax();
        fvOptions.constrain(EEqn);
        EEqn.solve();
        fvOptions.correct(thermo.he());

        // Keep temperature and the temperature-dependent transport models
        // consistent with the newly solved enthalpy.  Delaying this correction
        // until the next global time step leaves reacting mixtures using a
        // temperature calculated from the previous species composition.
        thermo.correct();

        Info<< phase.thermo().T().name()
            << ": min = " << min(phase.thermo().T()).value()
            << ", mean = " << average(phase.thermo().T()).value()
            << ", max = " << max(phase.thermo().T()).value()
            << endl;
    }
}
