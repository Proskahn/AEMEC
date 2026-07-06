/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "hydrogenCrossoverModel.H"
#include "zeroGradientFvPatchFields.H"

const Foam::word Foam::hydrogenCrossoverModel::modelName("hydrogenCrossover");

namespace Foam
{
    defineTypeNameAndDebug(hydrogenCrossoverModel, 0);
    defineRunTimeSelectionTable(hydrogenCrossoverModel, dictionary);
}


Foam::hydrogenCrossoverModel::hydrogenCrossoverModel
(
    const fvMesh& mesh,
    const dictionary& dict
)
:
    regIOobject
    (
        IOobject
        (
            modelName,
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        )
    ),
    mesh_(mesh),
    cH2_
    (
        IOobject
        (
            "cH2",
            mesh.time().timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar
        (
            "cH2",
            dimMoles/dimVol,
            dict.lookupOrDefault<scalar>("cH2Initial", 0.0)
        ),
        zeroGradientFvPatchScalarField::typeName
    ),
    DH2Eff_
    (
        IOobject
        (
            "DH2Eff",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("DH2Eff", sqr(dimLength)/dimTime, 0.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    h2Dmdt_
    (
        IOobject
        (
            "h2Dmdt",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("h2Dmdt", dimMoles/dimVol/dimTime, 0.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    JH2Diff_
    (
        IOobject
        (
            "JH2Diff",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("JH2Diff", dimMoles/sqr(dimLength)/dimTime, 0.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    JH2Drag_
    (
        IOobject
        (
            "JH2Drag",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("JH2Drag", dimMoles/sqr(dimLength)/dimTime, 0.0),
        zeroGradientFvPatchScalarField::typeName
    )
{}


Foam::hydrogenCrossoverModel::~hydrogenCrossoverModel()
{}


bool Foam::hydrogenCrossoverModel::writeData(Ostream& os) const
{
    return true;
}

// ************************************************************************* //
