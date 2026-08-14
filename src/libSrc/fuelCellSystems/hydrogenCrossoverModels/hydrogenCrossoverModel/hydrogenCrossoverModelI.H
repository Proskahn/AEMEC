/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
\*---------------------------------------------------------------------------*/

inline const Foam::volScalarField& Foam::hydrogenCrossoverModel::cH2() const
{
    return cH2_;
}


inline Foam::volScalarField& Foam::hydrogenCrossoverModel::cH2()
{
    return cH2_;
}


inline const Foam::volScalarField&
Foam::hydrogenCrossoverModel::cH2CathodeInterface() const
{
    return cH2CathodeInterface_;
}


inline Foam::volScalarField&
Foam::hydrogenCrossoverModel::cH2CathodeInterface()
{
    return cH2CathodeInterface_;
}


inline const Foam::volScalarField&
Foam::hydrogenCrossoverModel::cH2AnodeInterface() const
{
    return cH2AnodeInterface_;
}


inline Foam::volScalarField&
Foam::hydrogenCrossoverModel::cH2AnodeInterface()
{
    return cH2AnodeInterface_;
}


inline const Foam::volScalarField& Foam::hydrogenCrossoverModel::DH2Eff() const
{
    return DH2Eff_;
}


inline Foam::volScalarField& Foam::hydrogenCrossoverModel::DH2Eff()
{
    return DH2Eff_;
}


inline const Foam::volScalarField& Foam::hydrogenCrossoverModel::JH2Diff() const
{
    return JH2Diff_;
}


inline Foam::volScalarField& Foam::hydrogenCrossoverModel::JH2Diff()
{
    return JH2Diff_;
}


inline const Foam::volScalarField& Foam::hydrogenCrossoverModel::JH2Drag() const
{
    return JH2Drag_;
}


inline Foam::volScalarField& Foam::hydrogenCrossoverModel::JH2Drag()
{
    return JH2Drag_;
}


inline const Foam::volScalarField& Foam::hydrogenCrossoverModel::JH2Cross() const
{
    return JH2Cross_;
}


inline Foam::volScalarField& Foam::hydrogenCrossoverModel::JH2Cross()
{
    return JH2Cross_;
}


inline const Foam::volScalarField&
Foam::hydrogenCrossoverModel::h2CathodeDmdt() const
{
    return h2CathodeDmdt_;
}


inline Foam::volScalarField& Foam::hydrogenCrossoverModel::h2CathodeDmdt()
{
    return h2CathodeDmdt_;
}


inline const Foam::volScalarField&
Foam::hydrogenCrossoverModel::h2AnodeDmdt() const
{
    return h2AnodeDmdt_;
}


inline Foam::volScalarField& Foam::hydrogenCrossoverModel::h2AnodeDmdt()
{
    return h2AnodeDmdt_;
}


inline const Foam::word& Foam::hydrogenCrossoverModel::cathodeFluidRegion() const
{
    return cathodeFluidRegion_;
}


inline const Foam::word& Foam::hydrogenCrossoverModel::anodeFluidRegion() const
{
    return anodeFluidRegion_;
}


inline const Foam::word& Foam::hydrogenCrossoverModel::hydrogenSpecies() const
{
    return hydrogenSpecies_;
}

// ************************************************************************* //
