/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright held by the original author
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "electric.H"
#include "IOdictionary.H"
#include "zeroGradientFvPatchFields.H"
#include "addToRunTimeSelectionTable.H"

#include "sigmaModelList.H"
#include "dissolvedModel.H"
#include "hydrogenCrossoverModel.H"
#include "activationOverpotentialModel.H"

#include "fuelCellSystem.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace regionTypes
{
    defineTypeNameAndDebug(electric, 0);

    addToRunTimeSelectionTable
    (
        regionType,
        electric,
        dictionary
    );
}
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::regionTypes::electric::electric
(
    const fvMesh& mesh,
    const word& regionName
)
:
    regionType(mesh, regionName),

    mesh_(mesh),
    i_
    (
        IOobject
        (
            "i",
            this->time().timeName(),
            *this,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        *this,
        dimensionedVector(dimCurrent/dimArea, Zero),
        zeroGradientFvPatchVectorField::typeName
    ),
    phi_
    (
        IOobject
        (
            activationOverpotentialModel::phiName,
            this->time().timeName(),
            *this,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        *this
    ),
    j_
    (
        IOobject
        (
            activationOverpotentialModel::jName,
            this->time().timeName(),
            *this,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        *this,
        dimensionedScalar(dimCurrent/dimVol, Zero),
        zeroGradientFvPatchScalarField::typeName
    ),
    sigmaField_
    (
        IOobject
        (
            "sigma",
            this->time().timeName(),
            *this,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        *this,
        dimensionedScalar(j_.dimensions()/phi_.dimensions()*dimArea, 1.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    T_
    (
        IOobject
        (
            "T",
            this->time().timeName(),
            *this,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        *this,
        dimensionedScalar(dimTemperature, dict_.getOrDefault<scalar>("T0", 293.15)),
        zeroGradientFvPatchScalarField::typeName
    ),
    relax_(dict_.lookupOrDefault<scalar>("relax", 0.0)),
    maxVoltageStep_(GREAT),
    minVoltage_(-GREAT),
    maxVoltage_(GREAT),
    control_(dict_.lookupOrDefault<Switch>("control", false)),
    dissolveOnOff_(dict_.lookupOrDefault<Switch>("dissolveOnOff", false)),
    hydrogenCrossoverOnOff_
    (
        dict_.lookupOrDefault<Switch>
        (
            "hydrogenCrossoverOnOff",
            dict_.found("hydrogenCrossover")
        )
    ),
    patchName_(word::null),
    zoneName_(word::null),
    cellZoneIDs_(),
    galvanostatic_(true),
    ibar_(nullptr),
    voltage_(nullptr)
{
    Info<< "Electric region " << name()
        << ": dissolveOnOff=" << dissolveOnOff_
        << ", hydrogenCrossoverOnOff=" << hydrogenCrossoverOnOff_
        << ", hydrogenCrossoverDict=" << dict_.found("hydrogenCrossover")
        << endl;

    if (dissolveOnOff_)
    {
        dissolved_ = dissolvedModel::New(*this, dict_.subDict("dissolved"));
    }

    if (hydrogenCrossoverOnOff_)
    {
        hydrogenCrossover_ =
            hydrogenCrossoverModel::New(*this, dict_.subDict("hydrogenCrossover"));
    }

    sigma_.reset
    (
        new sigmaModelList
        (
            *this,
            dict_.subDict("sigma")
        )
    );

    if (control_)
    {
        const dictionary& controlDict = dict_.subDict("galvanostatic");
        patchName_ = controlDict.get<word>("patchName");
        galvanostatic_ = controlDict.get<Switch>("active");
        maxVoltageStep_ =
            controlDict.lookupOrDefault<scalar>("maxVoltageStep", GREAT);
        minVoltage_ = controlDict.lookupOrDefault<scalar>("minVoltage", -GREAT);
        maxVoltage_ = controlDict.lookupOrDefault<scalar>("maxVoltage", GREAT);

        if (maxVoltageStep_ <= 0 || minVoltage_ >= maxVoltage_)
        {
            FatalErrorInFunction
                << "Invalid galvanostatic voltage limits: maxVoltageStep="
                << maxVoltageStep_ << ", minVoltage=" << minVoltage_
                << ", maxVoltage=" << maxVoltage_ << exit(FatalError);
        }

        if (galvanostatic_)
        {
            ibar_.reset(Function1<scalar>::New("ibar", controlDict, this));
        }
        else
        {
            voltage_.reset(Function1<scalar>::New("voltage", controlDict, this));
        }

        if (controlDict.found("zoneName"))
        {
            zoneName_ = controlDict.get<word>("zoneName");
            cellZoneIDs_ = this->cellZones().indices(zoneName_);
        }
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::regionTypes::electric::~electric()
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::regionTypes::electric::solve()
{
    Info << "\nSolve for region " << name() << ":\n" << endl;

    tmp<fvScalarMatrix> phiEqn
    (
      - fvm::laplacian(sigmaField_, phi_, "laplacian(sigma,phi)")
      - j_
    );

    //- Set reference values
    if (phi_.needReference())
    {
        //- Update the potential field.
        //- Electro-Neutral, the total electron+proton flux should be zero
        const scalarField& source = j_;
        const scalarField& volume = this->V();
        scalarField sum = source*volume;
        scalar iDot(Foam::gSum(sum)/Foam::gSum(volume));

        for (label id = 0; id < nCells(); id++)
        {
            //- The option has to be true (forceReference)
            //- otherwise, errors happen in parallel simulations
            phiEqn->setReference(id, phi_[id] - iDot*relax_, true);
        }
    }

    //phiEqn->relax();

    phiEqn->solve();

    i_ = -sigmaField_ * fvc::grad(phi_);
    i_.correctBoundaryConditions();

    if (control_)
    {
        const label patchID = this->boundaryMesh().findPatchID(patchName_);

        if (patchID == -1)
        {
            FatalErrorInFunction
                << "Cannot find " << patchName_
                << "please check the patchName" << exit(FatalError);
        }

        const scalar appliedVoltage =
            Foam::gAverage(phi_.boundaryField()[patchID]);

        const scalar signedCurrent =
            Foam::gSum
            (
              - sigmaField_.boundaryField()[patchID]
              * phi_.boundaryField()[patchID].snGrad()
              * this->magSf().boundaryField()[patchID]
            );

        Info << "Controlled boundary current (A) at " << patchName_
            << ": signed = " << signedCurrent
            << ", magnitude = " << mag(signedCurrent)
            << ", voltage = " << appliedVoltage << endl;
    }

    if (dissolved_.valid())
    {
        dissolved_->solve();
    }

    if (hydrogenCrossover_.valid())
    {
        hydrogenCrossover_->solve();
    }
}


void Foam::regionTypes::electric::setRDeltaT()
{
    //- Do nothing, add if necessary.
}


void Foam::regionTypes::electric::correct()
{
    sigma_->correct(sigmaField_);
    sigmaField_.correctBoundaryConditions();

    if (control_)
    {
        //- Update the potential field.
        const scalarField& source = j_;
        const scalarField& volume = this->V();
        scalar iDot(0.0);

        if (cellZoneIDs_.size())
        {
            forAll(cellZoneIDs_, zoneI)
            {
                const labelList& cells = this->cellZones()[cellZoneIDs_[zoneI]];

                forAll(cells, i)
                {
                    const label cellI = cells[i];

                    iDot += source[cellI] * volume[cellI];
                }
            }

            reduce(iDot, sumOp<scalar>());
        }
        else
        {
            iDot = Foam::gSum(source*volume);
        }

        {
            label patchID = this->boundaryMesh().findPatchID(patchName_);

            if (patchID == -1)
            {
                FatalErrorInFunction
                    << "Cannot find " << patchName_
                    << "please check the patchName" << exit(FatalError);
            }

            scalarField phiBoundary = phi_.boundaryFieldRef()[patchID];

            scalar ibar0 = iDot/Foam::gSum(this->magSf().boundaryField()[patchID]);

            if (galvanostatic_)
            {
                const scalar targetIbar = ibar_->value(time().value());
                const scalar rawVoltageStep = relax_*(ibar0 - targetIbar);
                const scalar voltageStep = Foam::max
                (
                    -maxVoltageStep_,
                    Foam::min(maxVoltageStep_, rawVoltageStep)
                );
                const scalar oldVoltage = Foam::gAverage(phiBoundary);
                const scalar newVoltage = Foam::max
                (
                    minVoltage_,
                    Foam::min(maxVoltage_, oldVoltage + voltageStep)
                );

                phi_.boundaryFieldRef()[patchID] == newVoltage;

                Info << "galvanostatic target: " << targetIbar
                    << " A/m2, raw dV: " << rawVoltageStep
                    << ", limited dV: " << voltageStep << endl;
            }
            else
            {
                phi_.boundaryFieldRef()[patchID] == voltage_->value(time().value());
            }

            const scalar appliedVoltage =
                Foam::gAverage(phi_.boundaryField()[patchID]);

            Info << "ibar: " << ibar0
                << "\tvoltage: " << appliedVoltage << endl;
        }
    }

    if (dissolved_.valid())
    {
        dissolved_->correct();
    }

    if (hydrogenCrossover_.valid())
    {
        hydrogenCrossover_->correct();
    }
}


void Foam::regionTypes::electric::mapToCell
(
    fuelCellSystem& fuelCell
)
{
    Info << "Map " << name() << " to Cell " << nl << endl;

    //- heat source
    volScalarField heatSource = (i_ & i_)/sigmaField_;
    volScalarField heatSource0
    (
        IOobject
        (
            "heatSource",
            mesh_.time().timeName(),
            mesh_
        ),
        mesh_,
        dimensionedScalar(dimEnergy/dimVolume/dimTime, Zero)
    );

    heatSource0.rmap(heatSource, cellMapIO_);

    fuelCell.Qdot() += heatSource0;
}


void Foam::regionTypes::electric::mapFromCell
(
    fuelCellSystem& fuelCell
)
{
    Info << "Map " << name() << " from Cell " << nl << endl;

    scalarField& T = T_;

    forAll(T, cellI)
    {
        T[cellI] = fuelCell.T()[cellMapIO_[cellI]];
    }

    T_.correctBoundaryConditions();
}

// ************************************************************************* //
