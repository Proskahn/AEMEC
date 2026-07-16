{
    //- Solving temperature
    Info << "Solving for temperature:\n" << endl;

    fvScalarMatrix E1Eqn
    (
        // Write the common-temperature equation in material form.  Both
        // corrections are required because rhoCp changes with phase fraction
        // and composition; without the transient correction, a changing gas
        // holdup produces a spurious temperature jump even for uniform T.
        fvm::ddt(rhoCpCell, TCell)
      - fvm::Sp(fvc::ddt(rhoCpCell), TCell)
      + fvm::div(rhoCpPhiCell, TCell)
      - fvm::Sp(fvc::div(rhoCpPhiCell), TCell)
      - fvm::laplacian(kCell, TCell)
      - QdotCell
    );

    // We have to relax the equation sometimes...
    // But this greatly decreases convergence speed.
    E1Eqn.relax();

    E1Eqn.solve();

Info<< "T"
        << ": min = " << Foam::min(TCell.primitiveField())
        << ", mean = " << Foam::average(TCell.primitiveField())
        << ", max = " << Foam::max(TCell.primitiveField())
        << endl;
}
