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
    word diffusivityModel_;
    word dragModel_;
    word cathodeInterfaceType_;
    word anodeInterfaceType_;
    word cathodeGasPhase_;
    word anodeGasPhase_;

    dimensionedScalar nDrag_;
    dimensionedScalar cH2O_;
    scalar zIon_;
    dimensionedScalar DelecH2_;
    dimensionedScalar epsilonMembrane_;
    dimensionedScalar epsilonCathodeCL_;
    dimensionedScalar epsilonAnodeCL_;
    dimensionedScalar tauMembrane_;
    dimensionedScalar tauCathodeCL_;
    dimensionedScalar tauAnodeCL_;
    scalar bruggemanExponent_;
    dimensionedScalar kLaCathode_;
    dimensionedScalar kLaAnode_;
    dimensionedScalar cH2Anode_;
    dimensionedScalar cH2Cathode_;
    dimensionedScalar henryCathode_;
    dimensionedScalar henryAnode_;
    scalar faradaicDissolvedFraction_;
    scalar relax_;

    //- Hydrated-ionomer volume fraction used for dissolved-H2 storage.
    volScalarField epsilonIon_;

    //- Finite-rate CL transfer coefficient and signed dissolved-to-gas rate.
    volScalarField h2MassTransferCoeff_;
    volScalarField h2DissolvedToGas_;

    //- Fraction of Faradaic cathode H2 initially placed in the dissolved field.
    volScalarField h2DissolvedProduction_;

    void setZoneValue
    (
        volScalarField& field,
        const word& zoneName,
        scalar value
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

    void setZoneTransportProperties
    (
        const word& zoneName,
        scalar epsilon,
        scalar tau
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
