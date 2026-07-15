{
    autoPtr<phaseSystem::massTransferTable>
        massTransferPtr(this->massTransfer());

    phaseSystem::massTransferTable&
        massTransfer(massTransferPtr());

    if (!phase1_.pure())
    {
        UPtrList<volScalarField>& Y1 = phase1_.YActiveRef();

        forAll(Y1, i)
        {
            fvScalarMatrix Y1iEqn
            (
                phase1_.YiEqn(Y1[i])
             ==
               *massTransfer[Y1[i].name()]
              + fvOptions(alpha1, rho1, Y1[i])
            );

            Y1iEqn.relax();
            Y1iEqn.solve(mesh.solver("Yi"));

            Y1[i].max(SMALL);
            Y1[i].min(1.0);

            Info << "Y: " << Y1[i].name() << ":"
                 << gMin(Y1[i]) << " -> " << gMax(Y1[i])
                 << endl;
        }

        // The species equations change the mixture heat capacity while the
        // sensible-energy field still represents the old composition.  Keep
        // the pre-reaction temperature and recompute sensible energy for the
        // new composition before entering the energy equation.  This is the
        // sensible enthalpy carried by the electrochemical species sources.
        rhoThermo& thermo1 = phase1_.thermoRef();
        const volScalarField T1("TBeforeSpeciesEnergy.gas1", thermo1.T());
        phase1_.correctThermo();
        thermo1.he() = thermo1.he(thermo1.p(), T1).ref();
        thermo1.he().correctBoundaryConditions();
        thermo1.correct();
    }

    if (!phase2_.pure())
    {
        UPtrList<volScalarField>& Y2 = phase2_.YActiveRef();

        forAll(Y2, i)
        {
            fvScalarMatrix Y2iEqn
            (
                phase2_.YiEqn(Y2[i])
             ==
               *massTransfer[Y2[i].name()]
              + fvOptions(alpha2, rho2, Y2[i])
            );

            Y2iEqn.relax();
            Y2iEqn.solve(mesh.solver("Yi"));

            Y2[i].max(SMALL);
            Y2[i].min(1.0);

            Info << "Y: " << Y2[i].name() << ":"
                 << gMin(Y2[i]) << " -> " << gMax(Y2[i])
                 << endl;
        }
        rhoThermo& thermo2 = phase2_.thermoRef();
        const volScalarField T2("TBeforeSpeciesEnergy.gas2", thermo2.T());
        phase2_.correctThermo();
        thermo2.he() = thermo2.he(thermo2.p(), T2).ref();
        thermo2.he().correctBoundaryConditions();
        thermo2.correct();
    }
}
