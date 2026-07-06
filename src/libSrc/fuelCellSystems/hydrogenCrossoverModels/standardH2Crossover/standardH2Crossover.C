/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "standardH2Crossover.H"
#include "constants.H"
#include "addToRunTimeSelectionTable.H"

namespace Foam
{
namespace hydrogenCrossoverModels
{
    defineTypeNameAndDebug(standardH2Crossover, 0);

    addToRunTimeSelectionTable
    (
        hydrogenCrossoverModel,
        standardH2Crossover,
        dictionary
    );
}
}

const Foam::dimensionedScalar F = Foam::constant::physicoChemical::F;


Foam::hydrogenCrossoverModels::standardH2Crossover::standardH2Crossover
(
    const fvMesh& mesh,
    const dictionary& dict
)
:
    hydrogenCrossoverModel(mesh, dict),
    dict_(dict),
    sourceZoneName_(dict_.lookupOrDefault<word>("sourceZone", "ccl")),
    sinkZoneName_(dict_.lookupOrDefault<word>("sinkZone", "acl")),
    iName_(dict_.lookupOrDefault<word>("i", "i")),
    jName_(dict_.lookupOrDefault<word>("j", "j")),
    xi_("xi", dimless, dict_),
    cElec_("cElec", dimMoles/dimVol, dict_),
    DelecH2_("DelecH2", sqr(dimLength)/dimTime, dict_),
    epsilonM_("epsilonM", dimless, dict_),
    tau_("tau", dimless, dict_),
    sinkCoeff_
    (
        "sinkCoeff",
        dimless/dimTime,
        dict_.lookupOrDefault<scalar>("sinkCoeff", 0.0)
    ),
    cH2Anode_
    (
        "cH2Anode",
        dimMoles/dimVol,
        dict_.lookupOrDefault<scalar>("cH2Anode", 0.0)
    ),
    cH2Cathode_
    (
        "cH2Cathode",
        dimMoles/dimVol,
        dict_.lookupOrDefault<scalar>("cH2Cathode", -1.0)
    ),
    dragSign_(dict_.lookupOrDefault<scalar>("dragSign", 1.0)),
    relax_(dict_.lookupOrDefault<scalar>("relax", 1.0))
{}


Foam::hydrogenCrossoverModels::standardH2Crossover::~standardH2Crossover()
{}


void Foam::hydrogenCrossoverModels::standardH2Crossover::setZoneSource
(
    const word& zoneName,
    const scalarField& source,
    scalar sign
)
{
    if (zoneName == word::null)
    {
        return;
    }

    const label zoneId = mesh_.cellZones().findZoneID(zoneName);

    if (zoneId == -1)
    {
        FatalErrorInFunction
            << "Cannot find hydrogen crossover cellZone " << zoneName
            << exit(FatalError);
    }

    const labelList& cells = mesh_.cellZones()[zoneId];

    forAll(cells, i)
    {
        const label cellI = cells[i];
        h2Dmdt_[cellI] += sign*source[cellI];
    }
}


Foam::scalar Foam::hydrogenCrossoverModels::standardH2Crossover::zoneIntegral
(
    const volScalarField& field,
    const word& zoneName
) const
{
    if (zoneName == word::null)
    {
        return 0.0;
    }

    const label zoneId = mesh_.cellZones().findZoneID(zoneName);

    if (zoneId == -1)
    {
        return 0.0;
    }

    const labelList& cells = mesh_.cellZones()[zoneId];
    scalar sum = 0.0;

    forAll(cells, i)
    {
        const label cellI = cells[i];
        sum += field[cellI]*mesh_.V()[cellI];
    }

    reduce(sum, sumOp<scalar>());

    return sum;
}


void Foam::hydrogenCrossoverModels::standardH2Crossover::correct()
{
    DH2Eff_ = epsilonM_/tau_*DelecH2_;
    DH2Eff_.correctBoundaryConditions();

    JH2Diff_ = mag(DH2Eff_*fvc::grad(cH2_));

    const volVectorField& i = mesh_.lookupObject<volVectorField>(iName_);
    JH2Drag_ = mag(i)*xi_*cH2_/(F*cElec_);

    JH2Diff_.correctBoundaryConditions();
    JH2Drag_.correctBoundaryConditions();
}


void Foam::hydrogenCrossoverModels::standardH2Crossover::solve()
{
    const volVectorField& i = mesh_.lookupObject<volVectorField>(iName_);
    const volScalarField& j = mesh_.lookupObject<volScalarField>(jName_);

    h2Dmdt_ *= 0.0;

    volScalarField h2Generation
    (
        IOobject
        (
            "h2Generation",
            mesh_.time().timeName(),
            mesh_
        ),
        mag(j)/(2.0*F)
    );

    setZoneSource(sourceZoneName_, h2Generation, 1.0);

    volScalarField h2SinkCoeff
    (
        IOobject
        (
            "h2SinkCoeff",
            mesh_.time().timeName(),
            mesh_
        ),
        mesh_,
        dimensionedScalar("h2SinkCoeff", dimless/dimTime, 0.0)
    );

    if (sinkCoeff_.value() > 0.0 && sinkZoneName_ != word::null)
    {
        const label zoneId = mesh_.cellZones().findZoneID(sinkZoneName_);

        if (zoneId == -1)
        {
            FatalErrorInFunction
                << "Cannot find hydrogen crossover sink cellZone "
                << sinkZoneName_ << exit(FatalError);
        }

        const labelList& cells = mesh_.cellZones()[zoneId];

        forAll(cells, iCell)
        {
            h2SinkCoeff[cells[iCell]] = sinkCoeff_.value();
        }
    }

    surfaceScalarField phiDrag
    (
        IOobject
        (
            "phiH2Drag",
            mesh_.time().timeName(),
            mesh_
        ),
        dragSign_*xi_/(F*cElec_)*(fvc::interpolate(i) & mesh_.Sf())
    );

    tmp<fvScalarMatrix> h2Eqn
    (
        fvm::ddt(cH2_)
      + fvm::div(phiDrag, cH2_, "div(phiH2Drag,cH2)")
      - fvm::laplacian(DH2Eff_, cH2_, "laplacian(DH2Eff,cH2)")
      + fvm::Sp(h2SinkCoeff, cH2_)
      - h2SinkCoeff*cH2Anode_
      - h2Dmdt_
    );

    if (cH2Cathode_.value() >= 0.0)
    {
        const label zoneId = mesh_.cellZones().findZoneID(sourceZoneName_);

        if (zoneId != -1)
        {
            const labelList& cells = mesh_.cellZones()[zoneId];

            forAll(cells, iCell)
            {
                const label cellI = cells[iCell];
                h2Eqn->setReference(cellI, cH2Cathode_.value(), true);
            }
        }
    }

    h2Eqn->relax(relax_);
    h2Eqn->solve();
    cH2_.max(dimensionedScalar("zero", cH2_.dimensions(), 0.0));
    cH2_.correctBoundaryConditions();

    correct();

    Info<< "Hydrogen crossover: generated in " << sourceZoneName_
        << " = " << zoneIntegral(h2Dmdt_, sourceZoneName_) << " mol/s";

    if (sinkCoeff_.value() > 0.0 && sinkZoneName_ != word::null)
    {
        Info<< ", retained/released near " << sinkZoneName_
            << " = " << zoneIntegral(cH2_, sinkZoneName_) << " mol";
    }

    Info<< endl;
}

// ************************************************************************* //
