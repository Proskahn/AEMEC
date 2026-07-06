/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
\*---------------------------------------------------------------------------*/

#ifndef standardH2Crossover_H
#define standardH2Crossover_H

#include "hydrogenCrossoverModel.H"

namespace Foam
{
namespace hydrogenCrossoverModels
{

class standardH2Crossover
:
    public hydrogenCrossoverModel
{
    const dictionary& dict_;

    word sourceZoneName_;
    word sinkZoneName_;
    word iName_;
    word jName_;

    dimensionedScalar xi_;
    dimensionedScalar cElec_;
    dimensionedScalar DelecH2_;
    dimensionedScalar epsilonM_;
    dimensionedScalar tau_;
    dimensionedScalar sinkCoeff_;
    dimensionedScalar cH2Anode_;
    dimensionedScalar cH2Cathode_;

    scalar dragSign_;
    scalar relax_;

    void setZoneSource
    (
        const word& zoneName,
        const scalarField& source,
        scalar sign
    );

    scalar zoneIntegral(const volScalarField& field, const word& zoneName) const;

public:

    TypeName("standard");

    standardH2Crossover
    (
        const fvMesh& mesh,
        const dictionary& dict
    );

    virtual ~standardH2Crossover();

    virtual void correct();

    virtual void solve();
};

} // End namespace hydrogenCrossoverModels
} // End namespace Foam

#endif

// ************************************************************************* //
