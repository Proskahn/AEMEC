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
    cathodeFluidRegion_(dict.lookupOrDefault<word>("cathodeFluidRegion", "cathode")),
    anodeFluidRegion_(dict.lookupOrDefault<word>("anodeFluidRegion", "anode")),
    hydrogenSpecies_(dict.lookupOrDefault<word>("hydrogenSpecies", "H2")),
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
    cH2CathodeInterface_
    (
        IOobject
        (
            "cH2CathodeInterface",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar
        (
            "cH2CathodeInterface",
            dimMoles/dimVol,
            dict.lookupOrDefault<scalar>("cH2Cathode", 0.0)
        ),
        zeroGradientFvPatchScalarField::typeName
    ),
    cH2AnodeInterface_
    (
        IOobject
        (
            "cH2AnodeInterface",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar
        (
            "cH2AnodeInterface",
            dimMoles/dimVol,
            dict.lookupOrDefault<scalar>("cH2Anode", 0.0)
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
    ),
    JH2Conv_
    (
        IOobject
        (
            "JH2Conv",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("JH2Conv", dimMoles/sqr(dimLength)/dimTime, 0.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    JH2Cross_
    (
        IOobject
        (
            "JH2Cross",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("JH2Cross", dimMoles/sqr(dimLength)/dimTime, 0.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    h2CathodeDmdt_
    (
        IOobject
        (
            "h2CathodeDmdt",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("h2CathodeDmdt", dimMoles/dimVol/dimTime, 0.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    h2AnodeDmdt_
    (
        IOobject
        (
            "h2AnodeDmdt",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("h2AnodeDmdt", dimMoles/dimVol/dimTime, 0.0),
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
