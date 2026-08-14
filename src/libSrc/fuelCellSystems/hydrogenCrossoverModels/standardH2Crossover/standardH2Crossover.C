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
        interfaceWord(dict_, "cathodeInterface", "gasPhase", "gas")
    ),
    anodeGasPhase_
    (
        interfaceWord(dict_, "anodeInterface", "gasPhase", "gas")
    ),
    nDrag_("nDrag", dimless, dict_),
    cH2O_("cH2O", dimMoles/dimVol, dict_),
    zIon_(dict_.lookupOrDefault<scalar>("zIon", -1.0)),
    DelecH2_("DelecH2", sqr(dimLength)/dimTime, dict_),
    epsilonMembrane_
    (
        "epsilonMembrane",
        dimless,
        dict_.lookupOrDefault<scalar>
        (
            "epsilonMembrane",
            dict_.lookupOrDefault<scalar>("epsilonM", 0.2)
        )
    ),
    epsilonCathodeCL_
    (
        "epsilonCathodeCL",
        dimless,
        dict_.lookupOrDefault<scalar>
        (
            "epsilonCathodeCL",
            epsilonMembrane_.value()
        )
    ),
    epsilonAnodeCL_
    (
        "epsilonAnodeCL",
        dimless,
        dict_.lookupOrDefault<scalar>
        (
            "epsilonAnodeCL",
            epsilonMembrane_.value()
        )
    ),
    tauMembrane_
    (
        "tauMembrane",
        dimless,
        dict_.lookupOrDefault<scalar>
        (
            "tauMembrane",
            dict_.lookupOrDefault<scalar>("tau", 1.0)
        )
    ),
    tauCathodeCL_
    (
        "tauCathodeCL",
        dimless,
        dict_.lookupOrDefault<scalar>("tauCathodeCL", tauMembrane_.value())
    ),
    tauAnodeCL_
    (
        "tauAnodeCL",
        dimless,
        dict_.lookupOrDefault<scalar>("tauAnodeCL", tauMembrane_.value())
    ),
    bruggemanExponent_
    (
        dict_.lookupOrDefault<scalar>("bruggemanExponent", 1.5)
    ),
    kLaCathode_
    (
        "massTransferCoefficient",
        dimless/dimTime,
        interfaceScalar
        (
            dict_,
            "cathodeInterface",
            "massTransferCoefficient",
            0.0
        )
    ),
    kLaAnode_
    (
        "massTransferCoefficient",
        dimless/dimTime,
        interfaceScalar
        (
            dict_,
            "anodeInterface",
            "massTransferCoefficient",
            dict_.lookupOrDefault<scalar>("sinkCoeff", 0.0)
        )
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
    faradaicDissolvedFraction_
    (
        dict_.lookupOrDefault<scalar>("faradaicDissolvedFraction", 1.0)
    ),
    relax_(dict_.lookupOrDefault<scalar>("relax", 1.0)),
    epsilonIon_
    (
        IOobject
        (
            "epsilonIonH2",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        epsilonMembrane_,
        zeroGradientFvPatchScalarField::typeName
    ),
    h2MassTransferCoeff_
    (
        IOobject
        (
            "h2MassTransferCoeff",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar
        (
            "h2MassTransferCoeff",
            dimless/dimTime,
            0.0
        ),
        zeroGradientFvPatchScalarField::typeName
    ),
    h2DissolvedToGas_
    (
        IOobject
        (
            "h2DissolvedToGas",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar
        (
            "h2DissolvedToGas",
            dimMoles/dimVol/dimTime,
            0.0
        ),
        zeroGradientFvPatchScalarField::typeName
    ),
    h2DissolvedProduction_
    (
        IOobject
        (
            "h2DissolvedProduction",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar
        (
            "h2DissolvedProduction",
            dimMoles/dimVol/dimTime,
            0.0
        ),
        zeroGradientFvPatchScalarField::typeName
    )
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

    if
    (
        epsilonMembrane_.value() <= 0.0
     || epsilonMembrane_.value() > 1.0
     || epsilonCathodeCL_.value() <= 0.0
     || epsilonCathodeCL_.value() > 1.0
     || epsilonAnodeCL_.value() <= 0.0
     || epsilonAnodeCL_.value() > 1.0
     || tauMembrane_.value() <= 0.0
     || tauCathodeCL_.value() <= 0.0
     || tauAnodeCL_.value() <= 0.0
    )
    {
        FatalErrorInFunction
            << "Hydrated-ionomer volume fractions must be in (0,1] and "
            << "tortuosities must be positive" << exit(FatalError);
    }

    if
    (
        kLaCathode_.value() < 0.0
     || kLaAnode_.value() < 0.0
     || faradaicDissolvedFraction_ < 0.0
     || faradaicDissolvedFraction_ > 1.0
     || nDrag_.value() < 0.0
     || cH2O_.value() <= 0.0
     || mag(zIon_) <= SMALL
    )
    {
        FatalErrorInFunction
            << "CL mass-transfer coefficients must be non-negative and "
            << "faradaicDissolvedFraction must be in [0,1]; nDrag must be "
            << "non-negative, cH2O positive, and zIon non-zero"
            << exit(FatalError);
    }
}


Foam::hydrogenCrossoverModels::standardH2Crossover::~standardH2Crossover()
{}


void Foam::hydrogenCrossoverModels::standardH2Crossover::setZoneValue
(
    volScalarField& field,
    const word& zoneName,
    scalar value
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
        field[cells[i]] = value;
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
    concentration = dimensionedScalar
    (
        "zero",
        concentration.dimensions(),
        0.0
    );

    const label zoneId = mesh_.cellZones().findZoneID(zoneName);
    if (zoneId == -1)
    {
        FatalErrorInFunction
            << "Cannot find hydrogen interface cellZone " << zoneName
            << exit(FatalError);
    }

    const labelList& cells = mesh_.cellZones()[zoneId];

    if (interfaceType == "fixed")
    {
        forAll(cells, i)
        {
            concentration[cells[i]] = fixedConcentration.value();
        }
        concentration.correctBoundaryConditions();
        return;
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
    // phaseModel::X validates the requested component.  Its thermo() facade
    // is rhoThermo here and does not expose composition() directly.
    const scalarField& XH2 = gas.X(hydrogenSpecies_);
    const volScalarField& p = gas.thermo().p();
    const Map<label>& fluidCells = fluidRegion.cellMap();
    // The crossover mesh is the registered electric region (for example
    // phiAnion). Looking it up by name avoids refCast on fvMesh, whose
    // multiple OpenFOAM type registries make its diagnostic type() ambiguous.
    const regionType& membraneRegion = mesh_.time().lookupObject<regionType>
    (
        mesh_.name()
    );
    forAll(cells, i)
    {
        const label membraneCell = cells[i];
        const label masterCell = membraneRegion.cellMapIO()[membraneCell];

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


void Foam::hydrogenCrossoverModels::standardH2Crossover::
setZoneTransportProperties
(
    const word& zoneName,
    scalar epsilon,
    scalar tau
)
{
    setZoneValue(epsilonIon_, zoneName, epsilon);

    scalar diffusivity = DelecH2_.value();
    if (diffusivityModel_ == "porosityTortuosity")
    {
        diffusivity *= epsilon/tau;
    }
    else if (diffusivityModel_ == "bruggeman")
    {
        diffusivity *= pow(epsilon, bruggemanExponent_);
    }

    setZoneValue(DH2Eff_, zoneName, diffusivity);
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
    epsilonIon_ = epsilonMembrane_;

    if (diffusivityModel_ == "constant")
    {
        DH2Eff_ = DelecH2_;
    }
    else if (diffusivityModel_ == "porosityTortuosity")
    {
        DH2Eff_ = epsilonMembrane_/tauMembrane_*DelecH2_;
    }
    else
    {
        DH2Eff_ =
            pow(epsilonMembrane_.value(), bruggemanExponent_)*DelecH2_;
    }

    setZoneTransportProperties
    (
        sourceZoneName_,
        epsilonCathodeCL_.value(),
        tauCathodeCL_.value()
    );
    setZoneTransportProperties
    (
        sinkZoneName_,
        epsilonAnodeCL_.value(),
        tauAnodeCL_.value()
    );

    epsilonIon_.correctBoundaryConditions();
    DH2Eff_.correctBoundaryConditions();

    JH2Diff_ = mag(DH2Eff_*fvc::grad(cH2_));

    const volVectorField& i = mesh_.lookupObject<volVectorField>(iName_);
    JH2Drag_ = mag(i)*nDrag_*cH2_/(mag(zIon_)*F*cH2O_);
    JH2Cross_ = mag
    (
       -DH2Eff_*fvc::grad(cH2_)
      + nDrag_*cH2_/(zIon_*F*cH2O_)*i
    );

    JH2Diff_.correctBoundaryConditions();
    JH2Drag_.correctBoundaryConditions();
    JH2Cross_.correctBoundaryConditions();
}


void Foam::hydrogenCrossoverModels::standardH2Crossover::solve()
{
    const volVectorField& i = mesh_.lookupObject<volVectorField>(iName_);
    const volScalarField& j = mesh_.lookupObject<volScalarField>(jName_);

    h2CathodeDmdt_ *= 0.0;
    h2AnodeDmdt_ *= 0.0;
    h2DissolvedProduction_ *= 0.0;
    h2DissolvedToGas_ *= 0.0;
    h2MassTransferCoeff_ *= 0.0;

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

    // The electrochemical reaction class initially assembles the full
    // Faradaic H2 source in the cathode gas equation. The gas coupling below
    // subtracts the selected dissolved fraction locally, so the same H2 is
    // not generated twice.
    volScalarField h2FaradaicGeneration
    (
        IOobject
        (
            "h2FaradaicGeneration",
            mesh_.time().timeName(),
            mesh_
        ),
        mesh_,
        dimensionedScalar
        (
            "h2FaradaicGeneration",
            dimMoles/dimVol/dimTime,
            0.0
        )
    );

    const label cathodeZoneId =
        mesh_.cellZones().findZoneID(sourceZoneName_);
    const label anodeZoneId =
        mesh_.cellZones().findZoneID(sinkZoneName_);

    if (cathodeZoneId == -1 || anodeZoneId == -1)
    {
        FatalErrorInFunction
            << "Cannot find coupled dissolved-H2 catalyst zones "
            << sourceZoneName_ << " and " << sinkZoneName_
            << exit(FatalError);
    }

    const labelList& cathodeCells = mesh_.cellZones()[cathodeZoneId];
    const labelList& anodeCells = mesh_.cellZones()[anodeZoneId];

    forAll(cathodeCells, cellI)
    {
        const label cell = cathodeCells[cellI];
        h2FaradaicGeneration[cell] = mag(j[cell])/(2.0*F.value());
        h2DissolvedProduction_[cell] =
            faradaicDissolvedFraction_*h2FaradaicGeneration[cell];
        h2MassTransferCoeff_[cell] = kLaCathode_.value();
    }

    forAll(anodeCells, cellI)
    {
        h2MassTransferCoeff_[anodeCells[cellI]] = kLaAnode_.value();
    }

    h2DissolvedProduction_.correctBoundaryConditions();
    h2MassTransferCoeff_.correctBoundaryConditions();

    // Update the region-dependent ionomer storage and diffusivity before
    // assembling the dissolved-species equation.
    correct();

    surfaceScalarField phiDrag
    (
        IOobject
        (
            "phiH2Drag",
            mesh_.time().timeName(),
            mesh_
        ),
        nDrag_/(zIon_*F*cH2O_)*(fvc::interpolate(i) & mesh_.Sf())
    );

    tmp<fvScalarMatrix> h2Eqn
    (
        fvm::ddt(epsilonIon_, cH2_)
      + fvm::div(phiDrag, cH2_, "div(phiH2Drag,cH2)")
      - fvm::laplacian(DH2Eff_, cH2_, "laplacian(DH2Eff,cH2)")
      + fvm::Sp(h2MassTransferCoeff_, cH2_)
     ==
        h2DissolvedProduction_
      + h2MassTransferCoeff_
       *(cH2CathodeInterface_ + cH2AnodeInterface_)
    );

    h2Eqn->relax(relax_);
    h2Eqn->solve();
    cH2_.correctBoundaryConditions();

    correct();

    // Positive values are desorption from hydrated ionomer to pore gas;
    // negative values are absorption from pore gas into the ionomer.
    h2DissolvedToGas_ =
        h2MassTransferCoeff_
       *(
            cH2_
          - cH2CathodeInterface_
          - cH2AnodeInterface_
        );

    forAll(cathodeCells, cellI)
    {
        const label cell = cathodeCells[cellI];
        h2CathodeDmdt_[cell] =
            h2DissolvedToGas_[cell] - h2DissolvedProduction_[cell];
    }

    forAll(anodeCells, cellI)
    {
        const label cell = anodeCells[cellI];
        h2AnodeDmdt_[cell] = h2DissolvedToGas_[cell];
    }

    h2DissolvedToGas_.correctBoundaryConditions();
    h2CathodeDmdt_.correctBoundaryConditions();
    h2AnodeDmdt_.correctBoundaryConditions();

    const scalar faradaicH2Rate =
        zoneIntegral(h2FaradaicGeneration, sourceZoneName_);
    const scalar dissolvedProductionRate =
        zoneIntegral(h2DissolvedProduction_, sourceZoneName_);
    const scalar cathodeTransferRate =
        zoneIntegral(h2DissolvedToGas_, sourceZoneName_);
    const scalar anodeTransferRate =
        zoneIntegral(h2DissolvedToGas_, sinkZoneName_);
    const scalar cathodeGasRate =
        zoneIntegral(h2CathodeDmdt_, sourceZoneName_);
    const scalar anodeGasRate = zoneIntegral(h2AnodeDmdt_, sinkZoneName_);
    const scalar cathodeAnionReactionCurrent =
        zoneIntegral(j, sourceZoneName_);
    const scalar dissolvedCouplingRate =
        dissolvedProductionRate - cathodeTransferRate - anodeTransferRate;
    const scalar couplingImbalance =
        cathodeGasRate + anodeGasRate + dissolvedCouplingRate;
    const scalar h2CrossoverRate = max(anodeTransferRate, scalar(0));

    tmp<volScalarField> tDissolvedInventory(epsilonIon_*cH2_);
    const scalar dissolvedInventory =
        fvc::domainIntegrate(tDissolvedInventory()).value();

    Info<< "Hydrogen crossover objective: anode gas source rate = "
        << h2CrossoverRate << " mol/s" << endl;
    Info<< "Hydrogen dissolved-gas coupling conservation: cathode gas = "
        << cathodeGasRate << " mol/s, anode gas = " << anodeGasRate
        << " mol/s, dissolved field = " << dissolvedCouplingRate
        << " mol/s, imbalance = " << couplingImbalance << " mol/s"
        << endl;

    Info<< "Hydrogen production partition: anion reaction current in "
        << sourceZoneName_ << " = " << cathodeAnionReactionCurrent << " A"
        << ", Faradaic cathode H2 generation = " << faradaicH2Rate
        << " mol/s, initially dissolved = " << dissolvedProductionRate
        << " mol/s, direct Faradaic gas = "
        << faradaicH2Rate - dissolvedProductionRate
        << " mol/s, cathode dissolved-to-gas transfer = "
        << cathodeTransferRate
        << " mol/s, anode dissolved-to-gas transfer = "
        << anodeTransferRate
        << " mol/s, dissolved inventory = " << dissolvedInventory << " mol"
        << endl;
}

// ************************************************************************* //
