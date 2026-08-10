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
    dJdPhi_
    (
        IOobject
        (
            "dJdPhi",
            this->time().timeName(),
            *this,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        *this,
        dimensionedScalar
        (
            j_.dimensions()/phi_.dimensions(),
            Zero
        ),
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
    currentBalanceActive_(false),
    currentBalanceAbsoluteTolerance_(1e-12),
    currentBalanceRelativeTolerance_(1e-10),
    currentBalanceInitialStep_(0.01),
    currentBalanceMaxStep_(0.25),
    currentBalanceMaxIterations_(80),
    currentBalanceKineticsCorrectors_(1),
    currentBalanceFailOnNonConvergence_(true),
    currentBalancePotentialRelaxation_(0.25),
    currentBalanceMaxFieldPotentialStep_(0.02),
    polarizationActive_(false),
    polarizationTargets_(),
    minimumHoldDuration_(0.0),
    targetCurrentTolerance_(0.05),
    currentScale_(1.0),
    voltageTolerance_(0.005),
    currentStabilityTolerance_(0.02),
    requiredStabilitySamples_(5),
    targetIndex_(0),
    stabilitySamples_(0),
    holdStartTime_(0.0),
    previousMeasuredCurrentDensity_(0.0),
    previousAppliedVoltage_(0.0),
    previousTargetCurrentDensity_(GREAT),
    havePreviousControlSample_(false),
    polarizationComplete_(false),
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

    if (dict_.found("currentBalance"))
    {
        const dictionary& balanceDict = dict_.subDict("currentBalance");
        currentBalanceActive_ =
            balanceDict.lookupOrDefault<Switch>("active", false);
        currentBalanceAbsoluteTolerance_ =
            balanceDict.lookupOrDefault<scalar>("absoluteTolerance", 1e-12);
        currentBalanceRelativeTolerance_ =
            balanceDict.lookupOrDefault<scalar>("relativeTolerance", 1e-10);
        currentBalanceInitialStep_ =
            balanceDict.lookupOrDefault<scalar>("initialPotentialStep", 0.01);
        currentBalanceMaxStep_ =
            balanceDict.lookupOrDefault<scalar>("maxPotentialStep", 0.25);
        currentBalanceMaxIterations_ =
            balanceDict.lookupOrDefault<label>("maxIterations", 80);
        currentBalanceKineticsCorrectors_ =
            balanceDict.lookupOrDefault<label>("kineticsCorrectors", 1);
        currentBalanceFailOnNonConvergence_ =
            balanceDict.lookupOrDefault<Switch>("failOnNonConvergence", true);
        currentBalancePotentialRelaxation_ =
            balanceDict.lookupOrDefault<scalar>("potentialRelaxation", 0.25);
        currentBalanceMaxFieldPotentialStep_ =
            balanceDict.lookupOrDefault<scalar>
            (
                "maxFieldPotentialStep",
                0.02
            );

        if
        (
            currentBalanceAbsoluteTolerance_ < 0
         || currentBalanceRelativeTolerance_ < 0
         || currentBalanceInitialStep_ <= 0
         || currentBalanceMaxStep_ < currentBalanceInitialStep_
         || currentBalanceMaxIterations_ <= 0
         || currentBalanceKineticsCorrectors_ <= 0
         || currentBalancePotentialRelaxation_ <= 0
         || currentBalancePotentialRelaxation_ > 1
         || currentBalanceMaxFieldPotentialStep_ <= 0
        )
        {
            FatalIOErrorInFunction(balanceDict)
                << "Invalid currentBalance configuration for region "
                << name() << exit(FatalIOError);
        }

        if (currentBalanceActive_ && !phi_.needReference())
        {
            FatalIOErrorInFunction(balanceDict)
                << "currentBalance requires an all-Neumann potential field in "
                << name() << exit(FatalIOError);
        }
    }

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

            if (controlDict.found("polarizationCurve"))
            {
                const dictionary& polarizationDict =
                    controlDict.subDict("polarizationCurve");
                polarizationActive_ = polarizationDict.lookupOrDefault<Switch>
                (
                    "active",
                    false
                );

                if (polarizationActive_)
                {
                    polarizationTargets_ =
                        polarizationDict.get<List<scalar>>("targets");
                    minimumHoldDuration_ =
                        polarizationDict.lookupOrDefault<scalar>
                        ("minimumHoldDuration", 30.0);
                    targetCurrentTolerance_ =
                        polarizationDict.lookupOrDefault<scalar>
                        ("targetCurrentTolerance", 0.05);
                    currentScale_ = polarizationDict.lookupOrDefault<scalar>
                    (
                        "currentScale",
                        100.0
                    );
                    voltageTolerance_ =
                        polarizationDict.lookupOrDefault<scalar>
                        ("voltageTolerance", 0.005);
                    currentStabilityTolerance_ =
                        polarizationDict.lookupOrDefault<scalar>
                        ("currentStabilityTolerance", 0.02);
                    requiredStabilitySamples_ =
                        polarizationDict.lookupOrDefault<label>
                        ("stabilitySamples", 5);

                    if
                    (
                        !polarizationTargets_.size()
                     || minimumHoldDuration_ < 0.0
                     || targetCurrentTolerance_ < 0.0
                     || currentScale_ <= 0.0
                     || voltageTolerance_ < 0.0
                     || currentStabilityTolerance_ < 0.0
                     || requiredStabilitySamples_ <= 0
                    )
                    {
                        FatalErrorInFunction
                            << "Invalid polarizationCurve configuration in "
                            << this->name() << exit(FatalError);
                    }
                }
            }
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

        holdStartTime_ = time().value();
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::regionTypes::electric::~electric()
{}


Foam::scalar Foam::regionTypes::electric::integratedSource() const
{
    return Foam::gSum(j_.primitiveField()*this->V());
}


Foam::scalar Foam::regionTypes::electric::integratedSourceMagnitude() const
{
    return 0.5*Foam::gSum(Foam::mag(j_.primitiveField())*this->V());
}


Foam::scalar Foam::regionTypes::electric::meanPotential() const
{
    return
        Foam::gSum(phi_.primitiveField()*this->V())
       /Foam::max(Foam::gSum(this->V()), VSMALL);
}


void Foam::regionTypes::electric::shiftPotential(const scalar deltaPhi)
{
    phi_.primitiveFieldRef() += deltaPhi;
    phi_.correctBoundaryConditions();
}


Foam::scalar Foam::regionTypes::electric::targetCurrentDensity() const
{
    if (polarizationActive_)
    {
        return polarizationTargets_[targetIndex_];
    }

    return ibar_->value(time().value());
}


void Foam::regionTypes::electric::updateGalvanostaticControl
(
    const label patchID,
    const scalar measuredCurrentDensity,
    const scalar appliedVoltage
)
{
    const scalar targetIbar = targetCurrentDensity();

    if (mag(targetIbar - previousTargetCurrentDensity_) > SMALL)
    {
        holdStartTime_ = time().value();
        stabilitySamples_ = 0;
        havePreviousControlSample_ = false;
        previousTargetCurrentDensity_ = targetIbar;
    }

    const scalar currentError = measuredCurrentDensity - targetIbar;
    const scalar currentScale = max(mag(targetIbar), currentScale_);
    const scalar currentRelativeError = mag(currentError)/currentScale;
    const scalar rawVoltageStep = relax_*currentError;
    const scalar voltageStep = Foam::max
    (
        -maxVoltageStep_,
        Foam::min(maxVoltageStep_, rawVoltageStep)
    );
    const scalar requestedVoltage = appliedVoltage + voltageStep;
    const scalar nextVoltage = Foam::max
    (
        minVoltage_,
        Foam::min(maxVoltage_, requestedVoltage)
    );
    const bool voltageStepClipped = mag(rawVoltageStep - voltageStep) > SMALL;
    const bool voltageClipped = mag(nextVoltage - requestedVoltage) > SMALL;
    const bool voltageAtLimit =
        mag(appliedVoltage - minVoltage_) <= SMALL
     || mag(appliedVoltage - maxVoltage_) <= SMALL;
    const bool currentOnTarget = currentRelativeError <= targetCurrentTolerance_;
    const bool voltageStable = !havePreviousControlSample_
     || mag(appliedVoltage - previousAppliedVoltage_) <= voltageTolerance_;
    const bool currentStable = !havePreviousControlSample_
     || mag(measuredCurrentDensity - previousMeasuredCurrentDensity_)
        <= currentStabilityTolerance_*currentScale;

    const bool stableNow = currentOnTarget
     && voltageStable
     && currentStable
     && !voltageAtLimit
     && !voltageClipped;
    stabilitySamples_ = stableNow ? stabilitySamples_ + 1 : 0;

    const scalar holdEndTime = holdStartTime_ + minimumHoldDuration_;
    const bool holdComplete = time().value() + SMALL >= holdEndTime;
    const bool pointAccepted = polarizationActive_
     && !polarizationComplete_
     && holdComplete
     && stabilitySamples_ >= requiredStabilitySamples_;

    // This changes the boundary value for the next outer iteration.  The
    // current/voltage printed below are the paired post-solve values from the
    // present iteration, before the controller applies this correction.
    if (!polarizationComplete_)
    {
        phi_.boundaryFieldRef()[patchID] == nextVoltage;
    }

    Info<< "galvanostatic target: " << targetIbar
        << " A/m2, requested target: " << mag(targetIbar)
        << " A/m2, signed target: " << targetIbar
        << " A/m2, measured current density: " << measuredCurrentDensity
        << " A/m2, current error: " << currentError
        << " A/m2, current relative error: " << currentRelativeError
        << ", voltage: " << appliedVoltage
        << ", raw dV: " << rawVoltageStep
        << ", limited dV: " << voltageStep
        << ", voltageStepClipped: " << voltageStepClipped
        << ", voltageClipped: " << voltageClipped
        << ", voltageAtLimit: " << voltageAtLimit
        << ", currentClipped: false"
        << ", hold start: " << holdStartTime_
        << ", hold end: " << holdEndTime
        << ", stability samples: " << stabilitySamples_
        << "/" << requiredStabilitySamples_
        << ", accepted: " << pointAccepted
        << ", polarizationComplete: " << polarizationComplete_
        << endl;

    previousMeasuredCurrentDensity_ = measuredCurrentDensity;
    previousAppliedVoltage_ = appliedVoltage;
    havePreviousControlSample_ = true;

    if (pointAccepted)
    {
        if (targetIndex_ + 1 < polarizationTargets_.size())
        {
            ++targetIndex_;
            previousTargetCurrentDensity_ = polarizationTargets_[targetIndex_];
            holdStartTime_ = time().value();
            stabilitySamples_ = 0;
            havePreviousControlSample_ = false;
        }
        else
        {
            polarizationComplete_ = true;
            Info<< "Polarization curve complete: accepted "
                << polarizationTargets_.size()
                << " target points; stopping after this time step" << endl;

            // Let openFuelCell finish this complete multi-region time step.
            // Its ordinary runTime.write() then writes every region together
            // before Time::run() exits the outer loop.
            time().stopAt(Time::saWriteNow);
        }
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::regionTypes::electric::solve()
{
    Info << "\nSolve for region " << name() << ":\n" << endl;

    const bool implicitKinetics =
        Foam::gMax(Foam::mag(dJdPhi_.primitiveField())) > VSMALL;

    // With no kinetic derivative, an all-Neumann potential is defined only
    // up to an additive constant. Keep the gauge selected by the scalar
    // current-balance solve instead of allowing the algebraic reference cell
    // to select a new mean. A non-zero kinetic derivative makes the
    // linearized reaction/charge-conservation equation non-singular and
    // determines the absolute potential directly.
    const bool preserveMeanGauge =
        phi_.needReference() && currentBalanceActive_ && !implicitKinetics;
    const scalar selectedMeanPotential =
        preserveMeanGauge ? meanPotential() : 0.0;
    const scalarField previousPotential(phi_.primitiveField());

    tmp<fvScalarMatrix> phiEqn
    (
      - fvm::laplacian(sigmaField_, phi_, "laplacian(sigma,phi)")
      - fvm::Sp(dJdPhi_, phi_)
      - j_
      + dJdPhi_*phi_
    );

    //- Set reference values
    if (phi_.needReference() && !implicitKinetics)
    {
        //- Update the potential field.
        //- Electro-neutrality: the total electron plus ionic-carrier flux is
        //- zero.  The carrier can be an anion in an AEM case.
        // The source compatibility is enforced by regionTypeList's scalar
        // current-balance solve.  Pin one cell only to preserve the selected
        // gauge while allowing the Poisson equation to determine spatial
        // potential gradients.  The former all-cell reference suppressed
        // those gradients.
        const label referenceCell = Pstream::master() ? 0 : -1;
        const scalar referenceValue =
            referenceCell >= 0 ? phi_[referenceCell] : 0.0;
        phiEqn->setReference(referenceCell, referenceValue, true);
    }

    //phiEqn->relax();

    phiEqn->solve();

    if (currentBalanceActive_ && implicitKinetics)
    {
        const scalarField potentialCorrection
        (
            phi_.primitiveField() - previousPotential
        );
        const scalar maxPotentialCorrection =
            Foam::gMax(Foam::mag(potentialCorrection));
        scalar correctionScale = currentBalancePotentialRelaxation_;

        if (maxPotentialCorrection > currentBalanceMaxFieldPotentialStep_)
        {
            correctionScale = Foam::min
            (
                correctionScale,
                currentBalanceMaxFieldPotentialStep_/maxPotentialCorrection
            );
        }

        phi_.primitiveFieldRef() =
            previousPotential + correctionScale*potentialCorrection;
        phi_.correctBoundaryConditions();

        Info<< "Ionic Newton potential update: region=" << name()
            << ", rawMaxCorrection=" << maxPotentialCorrection << " V"
            << ", scale=" << correctionScale
            << ", appliedMaxCorrection="
            << correctionScale*maxPotentialCorrection << " V" << endl;
    }

    if (preserveMeanGauge)
    {
        shiftPotential(selectedMeanPotential - meanPotential());
    }

    i_ = -sigmaField_ * fvc::grad(phi_);
    i_.correctBoundaryConditions();

    if (dict_.lookupOrDefault<Switch>("electricDiagnostics", false))
    {
        scalar ohmicPower = 0.0;
        scalar positiveReactionCurrent = 0.0;
        scalar negativeReactionCurrent = 0.0;
        const vectorField& currentDensity = i_.primitiveField();
        const scalarField& conductivity = sigmaField_.primitiveField();
        const scalarField& reactionSource = j_.primitiveField();
        const scalarField& cellVolumes = this->V();

        forAll(cellVolumes, cellI)
        {
            ohmicPower +=
                magSqr(currentDensity[cellI])
               /max(conductivity[cellI], VSMALL)
               *cellVolumes[cellI];
            positiveReactionCurrent +=
                max(reactionSource[cellI], scalar(0))*cellVolumes[cellI];
            negativeReactionCurrent -=
                min(reactionSource[cellI], scalar(0))*cellVolumes[cellI];
        }

        reduce(ohmicPower, sumOp<scalar>());
        reduce(positiveReactionCurrent, sumOp<scalar>());
        reduce(negativeReactionCurrent, sumOp<scalar>());

        const scalar reactionCurrentScale = max
        (
            positiveReactionCurrent,
            negativeReactionCurrent
        );
        const scalar equivalentOhmicVoltage =
            reactionCurrentScale > VSMALL
          ? ohmicPower/reactionCurrentScale
          : 0.0;

        Info<< "AEMEC electric diagnostic: region=" << name()
            << ", sigma[min,mean,max]=(" << min(sigmaField_).value() << ","
            << sigmaField_.weightedAverage(this->V()).value() << ","
            << max(sigmaField_).value() << ")"
            << ", J[min,mean,max]=(" << min(j_).value() << ","
            << j_.weightedAverage(this->V()).value() << ","
            << max(j_).value() << ") A/m3"
            << ", dJdPhi[min,mean,max]=(" << min(dJdPhi_).value() << ","
            << dJdPhi_.weightedAverage(this->V()).value() << ","
            << max(dJdPhi_).value() << ") A/(m3 V)"
            << ", ohmicPower=" << ohmicPower << " W"
            << ", reactionCurrentScale=" << reactionCurrentScale << " A"
            << ", equivalentOhmicVoltage=" << equivalentOhmicVoltage
            << " V" << endl;
    }

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

        const scalar patchArea =
            Foam::gSum(this->magSf().boundaryField()[patchID]);

        if (patchArea <= VSMALL)
        {
            FatalErrorInFunction
                << "Controlled boundary patch " << patchName_
                << " has zero area" << exit(FatalError);
        }

        const scalar signedCurrentDensity = signedCurrent/patchArea;

        Info << "Controlled boundary current (A) at " << patchName_
            << ": signed = " << signedCurrent
            << ", magnitude = " << mag(signedCurrent)
            << ", current density = " << signedCurrentDensity << " A/m2"
            << ", voltage = " << appliedVoltage << endl;

        if (galvanostatic_)
        {
            updateGalvanostaticControl
            (
                patchID,
                signedCurrentDensity,
                appliedVoltage
            );
        }
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

    if (control_ && !galvanostatic_)
    {
        const label patchID = this->boundaryMesh().findPatchID(patchName_);

        if (patchID == -1)
        {
            FatalErrorInFunction
                << "Cannot find " << patchName_
                << "please check the patchName" << exit(FatalError);
        }

        // Potentiostatic operation prescribes the voltage before the solve.
        // Galvanostatic feedback is deliberately applied in solve(), after
        // the post-solve collector current has been measured.
        phi_.boundaryFieldRef()[patchID] == voltage_->value(time().value());
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
