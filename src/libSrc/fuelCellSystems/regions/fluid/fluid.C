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
#include "fluid.H"
#include "fuelCellSystem.H"
#include "patchToPatchInterpolation.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //
namespace Foam
{
namespace regionTypes
{
    defineTypeNameAndDebug(fluid, 0);

    addToRunTimeSelectionTable
    (
        regionType,
        fluid,
        dictionary
    );
}
}

// * * * * * * * * * * * * * * Private functions * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::regionTypes::fluid::fluid
(
    const fvMesh& mesh,
    const word& regionName
)
:
    regionType(mesh, regionName),

    mesh_(mesh)
{
    //- correct fluid, solve the mass and momentum equations, singlePhase and twoPhase.
    phases_ = phaseSystem::New
    (
        *this
    );
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //
Foam::regionTypes::fluid::~fluid()
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
void Foam::regionTypes::fluid::solve()
{
    phases_->solve();
}


void Foam::regionTypes::fluid::setRDeltaT()
{
    phases_->setRDeltaT();
}


void Foam::regionTypes::fluid::correct()
{
    phases_->correctEnergyTransport();
    phases_->correctThermo();
    phases_->correct();
}


void Foam::regionTypes::fluid::correctElectrochemistry()
{
    phases_->correctElectrochemistry();
}


void Foam::regionTypes::fluid::mapToCell
(
    fuelCellSystem& fuelCell
)
{
    Info << "Map " << name() << " to Cell" << nl << endl;

    const uniformDimensionedVectorField& g =
        this->time().lookupObject<uniformDimensionedVectorField>("g");

    //- Continuous phase name
    const word& continuous = phases_->continuous();

    // Phase model
    phaseModel& phase = phases_->phases()[continuous];
    const bool thermalEquilibrium = phases_->thermalEquilibrium();

    if (phase.isothermal())
    {
        return;
    }

    //- Gravity effect heat, alpha*rho*(U&g)
    volScalarField heatSource
    (
        phase
      * phase.thermo().rho()
      * (phase.U()&g)
    );

    if (thermalEquilibrium)
    {
        heatSource += phase.heQdot();
    }

    forAll(phases_->phases(), phasei)
    {
        phaseModel& phaseiModel = phases_->phases()[phasei];

        if (thermalEquilibrium && phaseiModel.name() != continuous)
        {
            // In local thermal equilibrium the parent equation is the sum of
            // every phase enthalpy equation, including kinetic/pressure work.
            heatSource +=
                phaseiModel
              * phaseiModel.thermo().rho()
              * (phaseiModel.U()&g)
              + phaseiModel.heQdot();
        }

        // Electrochemical heat is deposited in the global conjugate energy
        // equation.  It is not assigned entirely to a dilute product-gas
        // phase, which would create an artificial temperature spike.
        heatSource += phaseiModel.Qdot().ref();
    }

    // Heat transfer field in parent mesh
    volScalarField heatSource0
    (
        IOobject
        (
            "heatSource",
            mesh_.time().timeName(),
            mesh_
        ),
        mesh_,
        dimensionedScalar("zero", dimEnergy/dimVolume/dimTime, 0.0)
    );

    heatSource0.rmap(heatSource, cellMapIO_);

    // Map to parent mesh
    fuelCell.Qdot() += heatSource0;
    fuelCell.Qdot() += phases_->heatTransfer(fuelCell.T(), cellMapIO_)();

    // Rho*Cp
    scalarField rhoCp
    (
        phase
      * phase.thermo().Cp()
      * phase.thermo().rho()
    );

    // contErr*Cp
    scalarField contErrCp
    (
        phase.continuityError()
      * phase.thermo().Cp()
    );

    if (thermalEquilibrium)
    {
        forAll(phases_->phases(), phasei)
        {
            phaseModel& phaseiModel = phases_->phases()[phasei];

            if (phaseiModel.name() == continuous)
            {
                continue;
            }

            tmp<volScalarField> tPhaseRhoCp =
                phaseiModel
              * phaseiModel.thermo().Cp()
              * phaseiModel.thermo().rho();

            rhoCp += tPhaseRhoCp().primitiveField();

            tmp<volScalarField> tPhaseContErrCp =
                phaseiModel.continuityError()
              * phaseiModel.thermo().Cp();

            contErrCp += tPhaseContErrCp().primitiveField();
        }
    }

    // Perform reverse mapping
    fuelCell.rhoCp().rmap(rhoCp, cellMapIO_);

    fuelCell.contErrCp().rmap(contErrCp, cellMapIO_);

    // Map air fluxes
    labelList internalFaceMap
    (
        SubList<label>(faceMap_, this->nInternalFaces())
    );

    scalarField internalFaceMask
    (
        scalarField::subField(faceMask_, this->nInternalFaces())
    );

    //
    // ** recall phi already incorporates rho **
    //
    surfaceScalarField& continuousRhoPhi = phase.alphaRhoPhiRef();

    scalarField rhoPhi(continuousRhoPhi.primitiveField());

    tmp<surfaceScalarField> tRhoCpPhi =
        linearInterpolate(phase.thermo().Cp())*continuousRhoPhi;

    scalarField rhoCpPhi(tRhoCpPhi().primitiveField());

    if (thermalEquilibrium)
    {
        forAll(phases_->phases(), phasei)
        {
            phaseModel& phaseiModel = phases_->phases()[phasei];

            if (phaseiModel.name() == continuous)
            {
                continue;
            }

            surfaceScalarField& phaseRhoPhi =
                phaseiModel.alphaRhoPhiRef();

            rhoPhi += phaseRhoPhi.primitiveField();

            tmp<surfaceScalarField> tPhaseRhoCpPhi =
                linearInterpolate(phaseiModel.thermo().Cp())*phaseRhoPhi;

            rhoCpPhi += tPhaseRhoCpPhi().primitiveField();
        }
    }

    fuelCell.phi().rmap
    (
        rhoPhi*internalFaceMask,
        internalFaceMap
    );

    fuelCell.rhoCpPhi().rmap
    (
        rhoCpPhi*internalFaceMask,
        internalFaceMap
    );

    // Do flux boundary conditions
    forAll (patchesMapIO_, patchI)
    {
        if
        (
            patchesMapIO_[patchI] > -1
         && patchesMapIO_[patchI] < mesh_.boundary().size()
        )
        {
            // Patch maps
            labelField curFpm
            (
                labelField::subField
                (
                    faceMap_,
                    this->boundary()[patchI].size(),
                    this->boundary()[patchI].patch().start()
                )
            );

            scalarField curMask
            (
                scalarField::subField
                (
                    faceMask_,
                    this->boundary()[patchI].size(),
                    this->boundary()[patchI].patch().start()
                )
            );

            curFpm -= mesh_.boundary()
                        [patchesMapIO_[patchI]].patch().start();

            scalarField rhoPhiPatch
            (
                continuousRhoPhi.boundaryField()[patchI]
            );

            tmp<volScalarField> tContinuousCp = phase.thermo().Cp();
            scalarField rhoCpPhiPatch
            (
                tContinuousCp().boundaryField()[patchI]
               *continuousRhoPhi.boundaryField()[patchI]
            );

            if (thermalEquilibrium)
            {
                forAll(phases_->phases(), phasei)
                {
                    phaseModel& phaseiModel = phases_->phases()[phasei];

                    if (phaseiModel.name() == continuous)
                    {
                        continue;
                    }

                    surfaceScalarField& phaseRhoPhi =
                        phaseiModel.alphaRhoPhiRef();
                    tmp<volScalarField> tPhaseCp =
                        phaseiModel.thermo().Cp();

                    rhoPhiPatch += phaseRhoPhi.boundaryField()[patchI];
                    rhoCpPhiPatch +=
                        tPhaseCp().boundaryField()[patchI]
                       *phaseRhoPhi.boundaryField()[patchI];
                }
            }

            fuelCell.phi().boundaryFieldRef()[patchesMapIO_[patchI]].
                scalarField::rmap
                (
                    rhoPhiPatch*curMask,
                    curFpm
                );

            fuelCell.rhoCpPhi().boundaryFieldRef()[patchesMapIO_[patchI]].
                scalarField::rmap
                (
                    rhoCpPhiPatch*curMask,
                    curFpm
                );
        }
    }

    //- effective thermal conductivities

    scalarField kF(nCells(), 0.0);

    kF = phase.kappa()*phase;

    if (thermalEquilibrium)
    {
        forAll(phases_->phases(), phasei)
        {
            phaseModel& phaseiModel = phases_->phases()[phasei];

            if (phaseiModel.name() == continuous)
            {
                continue;
            }

            tmp<volScalarField> tPhaseKappa =
                phaseiModel.kappa()*phaseiModel;

            kF += tPhaseKappa().primitiveField();
        }
    }

    forAll(phases_->porousZone(), iz)
    {
        label znId =
                this->cellZones().findZoneID(phases_->porousZone()[iz].zoneName());

        scalar CpZn, kZn;
        phases_->porousZone()[iz].dict().lookup("Cp") >> CpZn;
        phases_->porousZone()[iz].dict().lookup("k") >> kZn;

        scalar porZn = phases_->porousZone()[iz].porosity();

        labelList znCells(this->cellZones()[znId]);

        forAll(znCells, cellI)
        {
            label cellId = znCells[cellI];

            kF[cellId] =
                    kZn*(scalar(1) - porZn) + kF[cellId]*porZn;
        }
    }

    // Perform reverse mapping
    fuelCell.k().rmap(kF, cellMapIO_);
}


void Foam::regionTypes::fluid::mapFromCell
(
    fuelCellSystem& fuelCell
)
{
    Info << "Map " << name() << " from Cell" << nl << endl;

    if (!fuelCell.solveEnergy())
    {
        phases_->setIsothermalTemperature(fuelCell.isothermalTemperature());
        return;
    }

    if (phases_->thermalEquilibrium())
    {
        forAll(phases_->phases(), phasei)
        {
            phaseModel& phase = phases_->phases()[phasei];
            rhoThermo& thermo = phase.thermoRef();

            // Species changes alter mixture enthalpy. Normalize composition
            // first, then reconstruct every phase enthalpy from the one
            // temperature solved in the parent conjugate-energy equation.
            phase.correctComposition();

            volScalarField T0(thermo.T());

            forAll(T0, cellI)
            {
                T0[cellI] = fuelCell.T()[cellMapIO_[cellI]];
            }

            thermo.he() = thermo.he(thermo.p(), T0).ref();
            thermo.he().correctBoundaryConditions();
            thermo.correct();

            // The he-to-T inversion has a finite convergence tolerance.
            // Re-impose the shared temperature exactly so both phase models
            // enter the flow/species solve with the same thermal state.
            thermo.T() = T0;
        }

        return;
    }

    const word& continuous = phases_->continuous();

    phaseModel& phase = phases_->phases()[continuous];

    volScalarField& he = phase.thermoRef().he();
    volScalarField& p = phase.thermoRef().p();

    volScalarField T0 = phase.thermoRef().T();

    forAll(phase.thermo().T(), cellI)
    {
        T0[cellI] = fuelCell.T()[cellMapIO_[cellI]];
    }

    he = phase.thermoRef().he(p, T0).ref();

    he.correctBoundaryConditions();
}

// ************************************************************************* //
