/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "standardH2Crossover.H"
#include "constants.H"
#include "phaseModel.H"
#include "regionType.H"
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


namespace
{

Foam::word interfaceWord
(
    const Foam::dictionary& dict,
    const Foam::word& interfaceName,
    const Foam::word& entry,
    const Foam::word& defaultValue
)
{
    if (!dict.found(interfaceName))
    {
        return defaultValue;
    }

    return dict.subDict(interfaceName).lookupOrDefault<Foam::word>
    (
        entry,
        defaultValue
    );
}


Foam::scalar interfaceScalar
(
    const Foam::dictionary& dict,
    const Foam::word& interfaceName,
    const Foam::word& entry,
    Foam::scalar defaultValue
)
{
    if (!dict.found(interfaceName))
    {
        return defaultValue;
    }

    return dict.subDict(interfaceName).lookupOrDefault<Foam::scalar>
    (
        entry,
        defaultValue
    );
}

} // End anonymous namespace


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
    diffusivityModel_
    (
        dict_.lookupOrDefault<word>("diffusivityModel", "porosityTortuosity")
    ),
    dragModel_(dict_.lookupOrDefault<word>("dragModel", "constant")),
    cathodeInterfaceType_
    (
        interfaceWord(dict_, "cathodeInterface", "type", "fixed")
    ),
    anodeInterfaceType_
    (
        interfaceWord(dict_, "anodeInterface", "type", "fixed")
    ),
    cathodeGasPhase_
    (
        interfaceWord(dict_, "cathodeInterface", "gasPhase", "hydrogen")
    ),
    anodeGasPhase_
    (
        interfaceWord(dict_, "anodeInterface", "gasPhase", "oxygen")
    ),
    xi_("xi", dimless, dict_),
    cElec_("cElec", dimMoles/dimVol, dict_),
    DelecH2_("DelecH2", sqr(dimLength)/dimTime, dict_),
    epsilonM_("epsilonM", dimless, dict_),
    tau_("tau", dimless, dict_),
    bruggemanExponent_
    (
        dict_.lookupOrDefault<scalar>("bruggemanExponent", 1.5)
    ),
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
    henryCathode_
    (
        "henryCoefficient",
        dimMoles/dimVol/dimPressure,
        interfaceScalar
        (
            dict_,
            "cathodeInterface",
            "henryCoefficient",
            0.0
        )
    ),
    henryAnode_
    (
        "henryCoefficient",
        dimMoles/dimVol/dimPressure,
        interfaceScalar
        (
            dict_,
            "anodeInterface",
            "henryCoefficient",
            0.0
        )
    ),
    UMembrane_
    (
        "UMembrane",
        dimVelocity,
        dict_.lookupOrDefault<vector>("UMembrane", vector::zero)
    ),
    dragSign_(dict_.lookupOrDefault<scalar>("dragSign", 1.0)),
    relax_(dict_.lookupOrDefault<scalar>("relax", 1.0))
{
    if
    (
        diffusivityModel_ != "constant"
     && diffusivityModel_ != "porosityTortuosity"
     && diffusivityModel_ != "bruggeman"
    )
    {
        FatalErrorInFunction
            << "Unknown hydrogen diffusivityModel " << diffusivityModel_
            << ". Valid options are constant, porosityTortuosity, bruggeman"
            << exit(FatalError);
    }

    if (dragModel_ != "constant")
    {
        FatalErrorInFunction
            << "Unsupported hydrogen dragModel " << dragModel_
            << ". Only constant is currently implemented"
            << exit(FatalError);
    }

    if
    (
        (cathodeInterfaceType_ != "fixed" && cathodeInterfaceType_ != "henry")
     || (anodeInterfaceType_ != "fixed" && anodeInterfaceType_ != "henry")
    )
    {
        FatalErrorInFunction
            << "Hydrogen interface type must be fixed or henry"
            << exit(FatalError);
    }
}


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


void Foam::hydrogenCrossoverModels::standardH2Crossover::setZoneUniformSource
(
    volScalarField& field,
    const word& zoneName,
    scalar rate
)
{
    if (zoneName == word::null || mag(rate) <= VSMALL)
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
    scalar volume = 0.0;

    forAll(cells, i)
    {
        volume += mesh_.V()[cells[i]];
    }
    reduce(volume, sumOp<scalar>());

    if (volume <= VSMALL)
    {
        FatalErrorInFunction
            << "Hydrogen crossover cellZone " << zoneName
            << " has zero volume" << exit(FatalError);
    }

    const scalar volumetricRate = rate/volume;
    forAll(cells, i)
    {
        field[cells[i]] += volumetricRate;
    }
}


void Foam::hydrogenCrossoverModels::standardH2Crossover::updateInterfaceConcentration
(
    volScalarField& concentration,
    const word& zoneName,
    const word& fluidRegionName,
    const word& gasPhaseName,
    const word& interfaceType,
    const dimensionedScalar& fixedConcentration,
    const dimensionedScalar& henryCoefficient
)
{
    concentration = fixedConcentration;

    if (interfaceType == "fixed")
    {
        concentration.correctBoundaryConditions();
        return;
    }

    const label zoneId = mesh_.cellZones().findZoneID(zoneName);
    if (zoneId == -1)
    {
        FatalErrorInFunction
            << "Cannot find hydrogen interface cellZone " << zoneName
            << exit(FatalError);
    }

    const regionType& fluidRegion = mesh_.time().lookupObject<regionType>
    (
        fluidRegionName
    );
    const word alphaName = IOobject::groupName("alpha", gasPhaseName);

    if (!fluidRegion.foundObject<phaseModel>(alphaName))
    {
        FatalErrorInFunction
            << "Cannot find gas phase " << gasPhaseName
            << " in fluid region " << fluidRegionName
            << exit(FatalError);
    }

    const phaseModel& gas = fluidRegion.lookupObject<phaseModel>(alphaName);
    const label hydrogenI = gas.thermo().composition().species()[hydrogenSpecies_];
    if (hydrogenI == -1)
    {
        FatalErrorInFunction
            << "Gas phase " << gasPhaseName << " in region " << fluidRegionName
            << " does not contain crossover species " << hydrogenSpecies_
            << exit(FatalError);
    }

    const scalarField& XH2 = gas.X(hydrogenSpecies_);
    const volScalarField& p = gas.thermo().p();
    const Map<label>& fluidCells = fluidRegion.cellMap();
    const labelList& cells = mesh_.cellZones()[zoneId];

    forAll(cells, i)
    {
        const label membraneCell = cells[i];
        const label masterCell = mesh_.cellMapIO()[membraneCell];

        if (!fluidCells.found(masterCell))
        {
            FatalErrorInFunction
                << "Cannot map membrane cell " << membraneCell
                << " in zone " << zoneName << " to fluid region "
                << fluidRegionName
                << ". Region mappings, not local cell ordering, are required"
                << exit(FatalError);
        }

        const label fluidCell = fluidCells[masterCell];
        concentration[membraneCell] = henryCoefficient.value()
          * max(p[fluidCell], scalar(0))
          * max(XH2[fluidCell], scalar(0));
    }

    concentration.correctBoundaryConditions();
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
    if (diffusivityModel_ == "constant")
    {
        DH2Eff_ = DelecH2_;
    }
    else if (diffusivityModel_ == "porosityTortuosity")
    {
        DH2Eff_ = epsilonM_/tau_*DelecH2_;
    }
    else
    {
        DH2Eff_ = pow(epsilonM_.value(), bruggemanExponent_)*DelecH2_;
    }
    DH2Eff_.correctBoundaryConditions();

    JH2Diff_ = mag(DH2Eff_*fvc::grad(cH2_));

    const volVectorField& i = mesh_.lookupObject<volVectorField>(iName_);
    JH2Drag_ = mag(i)*xi_*cH2CathodeInterface_/(F*cElec_);
    JH2Conv_ = mag(UMembrane_)*cH2_;
    JH2Cross_ = JH2Diff_ + JH2Drag_ + JH2Conv_;

    JH2Diff_.correctBoundaryConditions();
    JH2Drag_.correctBoundaryConditions();
    JH2Conv_.correctBoundaryConditions();
    JH2Cross_.correctBoundaryConditions();
}


void Foam::hydrogenCrossoverModels::standardH2Crossover::solve()
{
    const volVectorField& i = mesh_.lookupObject<volVectorField>(iName_);
    const volScalarField& j = mesh_.lookupObject<volScalarField>(jName_);

    h2Dmdt_ *= 0.0;
    h2CathodeDmdt_ *= 0.0;
    h2AnodeDmdt_ *= 0.0;

    updateInterfaceConcentration
    (
        cH2CathodeInterface_,
        sourceZoneName_,
        cathodeFluidRegion_,
        cathodeGasPhase_,
        cathodeInterfaceType_,
        cH2Cathode_,
        henryCathode_
    );
    updateInterfaceConcentration
    (
        cH2AnodeInterface_,
        sinkZoneName_,
        anodeFluidRegion_,
        anodeGasPhase_,
        anodeInterfaceType_,
        cH2Anode_,
        henryAnode_
    );

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

    surfaceScalarField phiConv
    (
        IOobject
        (
            "phiH2Conv",
            mesh_.time().timeName(),
            mesh_
        ),
        UMembrane_ & mesh_.Sf()
    );

    tmp<fvScalarMatrix> h2Eqn
    (
        fvm::ddt(cH2_)
      + fvm::div(phiDrag, cH2_, "div(phiH2Drag,cH2)")
      + fvm::div(phiConv, cH2_, "div(phiH2Conv,cH2)")
      - fvm::laplacian(DH2Eff_, cH2_, "laplacian(DH2Eff,cH2)")
      + fvm::Sp(h2SinkCoeff, cH2_)
      - h2SinkCoeff*cH2AnodeInterface_
      - h2Dmdt_
    );

    if (cathodeInterfaceType_ == "henry" || cH2Cathode_.value() >= 0.0)
    {
        const label zoneId = mesh_.cellZones().findZoneID(sourceZoneName_);

        if (zoneId != -1)
        {
            const labelList& cells = mesh_.cellZones()[zoneId];

            forAll(cells, iCell)
            {
                const label cellI = cells[iCell];
                h2Eqn->setReference
                (
                    cellI,
                    cH2CathodeInterface_[cellI],
                    true
                );
            }
        }
    }

    h2Eqn->relax(relax_);
    h2Eqn->solve();
    cH2_.max(dimensionedScalar("zero", cH2_.dimensions(), 0.0));
    cH2_.correctBoundaryConditions();

    correct();

    scalar h2CrossoverRate = 0.0;
    if (sinkCoeff_.value() > 0.0 && sinkZoneName_ != word::null)
    {
        // The implicit anode-side release represents H2 transferred from the
        // membrane into the anode.  The same positive molar rate is removed
        // from cathode H2 and added to anode H2 below.
        tmp<volScalarField> tH2ReleaseRate
        (
            h2SinkCoeff
           *max
            (
                cH2_ - cH2AnodeInterface_,
                dimensionedScalar("zero", cH2_.dimensions(), 0.0)
            )
        );
        h2CrossoverRate = zoneIntegral(tH2ReleaseRate(), sinkZoneName_);
    }

    setZoneUniformSource(h2CathodeDmdt_, sourceZoneName_, -h2CrossoverRate);
    setZoneUniformSource(h2AnodeDmdt_, sinkZoneName_, h2CrossoverRate);
    h2CathodeDmdt_.correctBoundaryConditions();
    h2AnodeDmdt_.correctBoundaryConditions();

    const scalar cathodeGasRate =
        zoneIntegral(h2CathodeDmdt_, sourceZoneName_);
    const scalar anodeGasRate = zoneIntegral(h2AnodeDmdt_, sinkZoneName_);

    Info<< "Hydrogen crossover objective: anode gas source rate = "
        << anodeGasRate << " mol/s" << endl;
    Info<< "Hydrogen crossover conservation: cathode H2 source = "
        << cathodeGasRate << " mol/s, anode H2 source = "
        << anodeGasRate << " mol/s, imbalance = "
        << cathodeGasRate + anodeGasRate << " mol/s" << endl;

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
