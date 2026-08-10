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

#include "electroChemicalReaction.H"

#include "phaseSystem.H"
#include "activationOverpotentialModel.H"
#include "dissolvedModel.H"
#include "hydrogenCrossoverModel.H"

#include "constants.H"
#include "PstreamReduceOps.H"

const Foam::dimensionedScalar Rgas = Foam::constant::physicoChemical::R;
const Foam::dimensionedScalar dimF = Foam::constant::physicoChemical::F;

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //
template<class ReactionThermo>
Foam::combustionModels::electroChemicalReaction<ReactionThermo>::electroChemicalReaction
(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turb,
    const word& electroChemicalReactionProperties
)
:
    ThermoCombustion<ReactionThermo>(modelType, thermo, turb),
    thermo_(thermo),
    saturation_(saturationModel::New(this->subDict("saturation"), this->mesh())),
    dissolved_(this->template lookupOrDefault<Switch>("dissolved", true))
{
    //- Get current phase Model
    const phaseModel& phase = this->mesh().template
        lookupObject<phaseModel>
        (
            this->thermo_.phasePropertyName("alpha")
        );

    //- Activation overpotential model
    eta_ = activationOverpotentialModel::New(phase, this->subDict("activationOverpotentialModel"));
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::electroChemicalReaction<ReactionThermo>::~electroChemicalReaction()
{}

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //
template<class ReactionThermo>
void Foam::combustionModels::electroChemicalReaction<ReactionThermo>::correct()
{
    //- Activation overpotential model update
    eta_->correct();

    //- Get sub regions
    //- Refer to regionType
    //- Including: fluid, electron (BPP + GDL + CL), and anion/legacy-ion
    //- carrier (CLs + membrane).
    const regionType& fluidPhase = eta_->region
    (
        word(eta_->regions().subDict("fluid").lookup("name"))
    );
    const regionType& anionPhase = eta_->region
    (
        word(eta_->regions().subDict(eta_->regions().found("anion") ? "anion" : "ion").lookup("name"))
    );

     // Get the phase System
     const phaseSystem& phaseSys = this->mesh().template
     lookupObject<phaseSystem>(phaseSystem::propertiesName);

    //- Get the present phase Model from fluid phase
    const phaseModel& phase = this->mesh().template
        lookupObject<phaseModel>(thermo_.phasePropertyName("alpha"));

    word water(phaseModel::water);

    //- Lable of water (H2O)
    const label specieI =
        thermo_.composition().species()[water];

    //- In case no water exists
    if (specieI == -1)
    {
        return;
    }

    //- Get the specieStoichCoeff
    dimensionedScalar specieStoichCoeff
    (
        "stoichCoeff",
        dimless,
        eta_->nernst().rxnList().found(water)
      ? eta_->nernst().rxnList()[water]/mag(eta_->nernst().rxnList()["e"])
      : 0.0
    );

    //- kg/kmol -> kg/mol
    const dimensionedScalar Wi
    (
        "W",
        dimMass/dimMoles,
        thermo_.composition().W(specieI)/1000
    );

    //- water production
    volScalarField wSpecie
    (
        eta_->j()*Wi*specieStoichCoeff/dimF
    );

    // A PEM/Nafion-style dissolved-water model is optional.  AEM cases may
    // intentionally omit it and apply the reaction water source directly to
    // their two-phase fluid model.
    const bool useDissolvedWater =
        dissolved_
     && anionPhase.foundObject<dissolvedModel>(dissolvedModel::modelName);
    scalarField* dissolvedWaterRate = nullptr;

    //- Only consider catalyst zone
    label znId = fluidPhase.cellZones().findZoneID(eta_->zoneName());
    const labelList& cells = fluidPhase.cellZones()[znId];

    if
    (
        this->template lookupOrDefault<Switch>
        ("electrochemicalDiagnostics", false)
    )
    {
        const scalarField& reactionCurrent = eta_->j();
        const scalarField& activationOverpotential = eta_->eta();
        const scalarField& nernstPotential = eta_->nernst()();
        const scalarField& temperature = thermo_.T();
        const scalarField& waterMoleFraction = phase.X(water);
        const scalarField& gasVolumeFraction = phase;

        const label hydrogenSpecieI =
            thermo_.composition().species()["H2"];
        const scalarField* hydrogenMoleFraction = nullptr;
        if (hydrogenSpecieI != -1)
        {
            hydrogenMoleFraction = &phase.X("H2");
        }

        scalar catalystVolume = 0.0;
        scalar integratedCurrent = 0.0;
        scalar activationPower = 0.0;
        scalar currentWeightedNernst = 0.0;
        scalar etaWeightedSum = 0.0;
        scalar nernstWeightedSum = 0.0;
        scalar temperatureWeightedSum = 0.0;
        scalar waterWeightedSum = 0.0;
        scalar hydrogenWeightedSum = 0.0;
        scalar gasFractionWeightedSum = 0.0;
        scalar etaMin = GREAT;
        scalar etaMax = -GREAT;
        scalar nernstMin = GREAT;
        scalar nernstMax = -GREAT;
        scalar temperatureMin = GREAT;
        scalar temperatureMax = -GREAT;
        scalar waterMin = GREAT;
        scalar waterMax = -GREAT;
        scalar hydrogenMin = GREAT;
        scalar hydrogenMax = -GREAT;
        scalar gasFractionMin = GREAT;
        scalar gasFractionMax = -GREAT;
        const scalar reactionSign =
            eta_->nernst().rxnList()["e"]
           /mag(eta_->nernst().rxnList()["e"]);

        forAll(cells, cellI)
        {
            const label fluidId = cells[cellI];
            const scalar volume = fluidPhase.V()[fluidId];

            catalystVolume += volume;
            integratedCurrent += reactionCurrent[fluidId]*volume;
            activationPower +=
                reactionCurrent[fluidId]
               *reactionSign
               *activationOverpotential[fluidId]
               *volume;
            currentWeightedNernst +=
                reactionCurrent[fluidId]*nernstPotential[fluidId]*volume;
            etaWeightedSum += activationOverpotential[fluidId]*volume;
            nernstWeightedSum += nernstPotential[fluidId]*volume;
            temperatureWeightedSum += temperature[fluidId]*volume;
            waterWeightedSum += waterMoleFraction[fluidId]*volume;
            gasFractionWeightedSum += gasVolumeFraction[fluidId]*volume;
            etaMin = min(etaMin, activationOverpotential[fluidId]);
            etaMax = max(etaMax, activationOverpotential[fluidId]);
            nernstMin = min(nernstMin, nernstPotential[fluidId]);
            nernstMax = max(nernstMax, nernstPotential[fluidId]);
            temperatureMin = min(temperatureMin, temperature[fluidId]);
            temperatureMax = max(temperatureMax, temperature[fluidId]);
            waterMin = min(waterMin, waterMoleFraction[fluidId]);
            waterMax = max(waterMax, waterMoleFraction[fluidId]);
            gasFractionMin = min(gasFractionMin, gasVolumeFraction[fluidId]);
            gasFractionMax = max(gasFractionMax, gasVolumeFraction[fluidId]);

            if (hydrogenMoleFraction)
            {
                hydrogenWeightedSum += (*hydrogenMoleFraction)[fluidId]*volume;
                hydrogenMin = min(hydrogenMin, (*hydrogenMoleFraction)[fluidId]);
                hydrogenMax = max(hydrogenMax, (*hydrogenMoleFraction)[fluidId]);
            }
        }

        reduce(catalystVolume, sumOp<scalar>());
        reduce(integratedCurrent, sumOp<scalar>());
        reduce(activationPower, sumOp<scalar>());
        reduce(currentWeightedNernst, sumOp<scalar>());
        reduce(etaWeightedSum, sumOp<scalar>());
        reduce(nernstWeightedSum, sumOp<scalar>());
        reduce(temperatureWeightedSum, sumOp<scalar>());
        reduce(waterWeightedSum, sumOp<scalar>());
        reduce(hydrogenWeightedSum, sumOp<scalar>());
        reduce(gasFractionWeightedSum, sumOp<scalar>());
        reduce(etaMin, minOp<scalar>());
        reduce(etaMax, maxOp<scalar>());
        reduce(nernstMin, minOp<scalar>());
        reduce(nernstMax, maxOp<scalar>());
        reduce(temperatureMin, minOp<scalar>());
        reduce(temperatureMax, maxOp<scalar>());
        reduce(waterMin, minOp<scalar>());
        reduce(waterMax, maxOp<scalar>());
        reduce(hydrogenMin, minOp<scalar>());
        reduce(hydrogenMax, maxOp<scalar>());
        reduce(gasFractionMin, minOp<scalar>());
        reduce(gasFractionMax, maxOp<scalar>());

        const scalar electronCount = mag(eta_->nernst().rxnList()["e"]);
        const scalar hydrogenStoich = eta_->nernst().rxnList().found("H2")
          ? eta_->nernst().rxnList()["H2"]
          : 0.0;
        const scalar hydrogenFaradaicRate =
            integratedCurrent*hydrogenStoich/(electronCount*dimF.value());
        const scalar safeVolume = max(catalystVolume, VSMALL);
        const scalar safeCurrent = max(integratedCurrent, VSMALL);

        Info<< "AEMEC reaction diagnostic: fluidRegion=" << this->mesh().name()
            << ", phase=" << phase.name()
            << ", zone=" << eta_->zoneName()
            << ", reactionCurrent=" << integratedCurrent << " A"
            << ", H2FaradaicRate=" << hydrogenFaradaicRate << " mol/s"
            << ", j0=" << eta_->j0().value() << " A/m3"
            << ", activationPower=" << activationPower << " W"
            << ", equivalentActivationVoltage="
            << activationPower/safeCurrent << " V"
            << ", currentWeightedNernst="
            << currentWeightedNernst/safeCurrent << " V"
            << ", eta[min,mean,max]=(" << etaMin << ","
            << etaWeightedSum/safeVolume << "," << etaMax << ") V"
            << ", nernst[min,mean,max]=(" << nernstMin << ","
            << nernstWeightedSum/safeVolume << "," << nernstMax << ") V"
            << ", T[min,mean,max]=(" << temperatureMin << ","
            << temperatureWeightedSum/safeVolume << "," << temperatureMax << ") K"
            << ", XH2O[min,mean,max]=(" << waterMin << ","
            << waterWeightedSum/safeVolume << "," << waterMax << ")"
            << ", XH2[min,mean,max]=(";

        if (hydrogenMoleFraction)
        {
            Info<< hydrogenMin << "," << hydrogenWeightedSum/safeVolume
                << "," << hydrogenMax;
        }
        else
        {
            Info<< "not-present";
        }

        Info<< "), alphaGas[min,mean,max]=(" << gasFractionMin << ","
            << gasFractionWeightedSum/safeVolume << "," << gasFractionMax
            << ")" << endl;
    }

    if (useDissolvedWater)
    {
        dissolvedModel& dW = const_cast<dissolvedModel&>
        (
            anionPhase.template
            lookupObject<dissolvedModel>(dissolvedModel::modelName)
        );

        scalarField& act = const_cast<volScalarField&>(dW.act());
        scalarField& dmdt = const_cast<volScalarField&>(dW.dmdt());
        const scalarField& xH2O = phase.X(water);
        const scalarField act0 =
            xH2O
          * this->thermo_.p()
          / saturation_->pSat(thermo_.T()).ref()
          + 2.*(scalar(1) - phase.primitiveField());

        forAll(cells, cellI)
        {
            const label fluidId = cells[cellI];
            const label anionId =
                anionPhase.cellMap()[fluidPhase.cellMapIO()[fluidId]];
            act[anionId] = act0[fluidId];
        }

        dW.update(eta_->zoneName());
        dissolvedWaterRate = &dmdt;
    }

    //- Update the water production in phase model
    forAll(cells, cellI)
    {
        //- get cell IDs
        label fluidId = cells[cellI];
        label anionId = anionPhase.cellMap()[fluidPhase.cellMapIO()[fluidId]];

        scalar dissolvedMassRate = 0.0;
        if (dissolvedWaterRate)
        {
            (*dissolvedWaterRate)[anionId] += wSpecie[fluidId]/Wi.value();
            dissolvedMassRate = (*dissolvedWaterRate)[anionId]*Wi.value();
        }

        if (phaseSys.isSinglePhase() || !eta_->phaseChange())
        {
            //- Water is transferred between current phase and dissolved phase
            volScalarField& iDmdtWater = const_cast<volScalarField&>
                (phase.iDmdt(water));

            iDmdtWater[fluidId] = wSpecie[fluidId] - dissolvedMassRate;
        }
        else
        {
            //- Get the name of the other phase
            const word name1 = Pair<word>
            (
                phaseSys.phases()[0].name(),
                phaseSys.phases()[1].name()
            ).other(phase.name());

            //- Water is transferred between the other phase and dissolved phase
            scalarField& iDmdtWater = const_cast<volScalarField&>
                (phaseSys.phases()[name1].iDmdt(water));

            iDmdtWater[fluidId] = wSpecie[fluidId] - dissolvedMassRate;
        }
    }
}


template<class ReactionThermo>
Foam::tmp<Foam::fvScalarMatrix>
Foam::combustionModels::electroChemicalReaction<ReactionThermo>::R
(
    volScalarField& Y
) const
{
    //- Get current phase Model
    const phaseModel& phase = this->mesh().template
        lookupObject<phaseModel>
        (
            thermo_.phasePropertyName("alpha")
        );

    word water(phaseModel::water);

    if (Y.member() == water)
    {
        return fvm::Sp(phase.iDmdt(water)/(Y + SMALL), Y);      // TODO: is this a good approach.
    }

    const label specieI =
        thermo_.composition().species()[Y.member()];

    //- kg/kmol -> kg/mol
    const dimensionedScalar Wi
    (
        "W",
        dimMass/dimMoles,
        thermo_.composition().W(specieI)/1000
    );

    dimensionedScalar specieStoichCoeff
    (
        "stoichCoeff",
        dimless,
        eta_->nernst().rxnList().found(Y.member())
      ? eta_->nernst().rxnList()[Y.member()]/mag(eta_->nernst().rxnList()["e"])
      : 0.0
    );

    volScalarField wSpecie
    (
        eta_->j()*Wi*specieStoichCoeff/dimF
    );

    volScalarField& iDmdt = const_cast<volScalarField&>(phase.iDmdt(Y.member()));

    iDmdt = wSpecie;

    // Hydrogen that leaves the cathode membrane interface and reaches the
    // anode must be coupled to both gas-phase species equations.  The source
    // fields live on the anion/electrolyte mesh, so map via the common master
    // cell labels.  This is valid for decomposed cases and deliberately does
    // not assume local cell indices agree between regions.
    const word anionRegionKey =
        eta_->regions().found("anion") ? "anion" : "ion";
    const regionType& anionPhase = eta_->region
    (
        word(eta_->regions().subDict(anionRegionKey).lookup("name"))
    );

    if
    (
        anionPhase.foundObject<hydrogenCrossoverModel>
        (hydrogenCrossoverModel::modelName)
    )
    {
        const hydrogenCrossoverModel& crossover =
            anionPhase.lookupObject<hydrogenCrossoverModel>
            (hydrogenCrossoverModel::modelName);

        if (Y.member() == crossover.hydrogenSpecies())
        {
            const volScalarField* crossoverDmdt = nullptr;

            if (this->mesh().name() == crossover.cathodeFluidRegion())
            {
                crossoverDmdt = &crossover.h2CathodeDmdt();
            }
            else if (this->mesh().name() == crossover.anodeFluidRegion())
            {
                crossoverDmdt = &crossover.h2AnodeDmdt();
            }

            if (crossoverDmdt)
            {
                const word fluidRegionKey = "fluid";
                const regionType& fluidPhase = eta_->region
                (
                    word
                    (
                        eta_->regions().subDict(fluidRegionKey).lookup("name")
                    )
                );
                const Map<label>& anionCells = anionPhase.cellMap();

                forAll(iDmdt, fluidCell)
                {
                    const label masterCell = fluidPhase.cellMapIO()[fluidCell];

                    if (anionCells.found(masterCell))
                    {
                        const label anionCell = anionCells[masterCell];
                        iDmdt[fluidCell] +=
                            (*crossoverDmdt)[anionCell]*Wi.value();
                    }
                }
            }
        }
    }

    return fvm::Sp(iDmdt/(Y + SMALL), Y);       // TODO: is this a good approach.
}


template<class ReactionThermo>
Foam::tmp<Foam::volScalarField>
Foam::combustionModels::electroChemicalReaction<ReactionThermo>::Qdot() const
{
    return eta_->Qdot();
}


template<class ReactionThermo>
bool Foam::combustionModels::electroChemicalReaction<ReactionThermo>::read()
{
    if (ThermoCombustion<ReactionThermo>::read())
    {
        return true;
    }
    else
    {
        return false;
    }
}

// ************************************************************************* //
