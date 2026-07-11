/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
\*---------------------------------------------------------------------------*/

#ifndef standardH2Crossover_H
#define standardH2Crossover_H

#include "hydrogenCrossoverModel.H"
#include "dimensionedVector.H"

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
    word diffusivityModel_;
    word dragModel_;
    word cathodeInterfaceType_;
    word anodeInterfaceType_;
    word cathodeGasPhase_;
    word anodeGasPhase_;

    dimensionedScalar xi_;
    dimensionedScalar cElec_;
    dimensionedScalar DelecH2_;
    dimensionedScalar epsilonM_;
    dimensionedScalar tau_;
    scalar bruggemanExponent_;
    dimensionedScalar sinkCoeff_;
    dimensionedScalar cH2Anode_;
    dimensionedScalar cH2Cathode_;
    dimensionedScalar henryCathode_;
    dimensionedScalar henryAnode_;
    dimensionedVector UMembrane_;

    scalar dragSign_;
    scalar relax_;

    void setZoneSource
    (
        const word& zoneName,
        const scalarField& source,
        scalar sign
    );

    void setZoneUniformSource
    (
        volScalarField& field,
        const word& zoneName,
        scalar rate
    );

    void updateInterfaceConcentration
    (
        volScalarField& concentration,
        const word& zoneName,
        const word& fluidRegionName,
        const word& gasPhaseName,
        const word& interfaceType,
        const dimensionedScalar& fixedConcentration,
        const dimensionedScalar& henryCoefficient
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
