/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright held by the original author
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "activationOverpotentialModelTemplate.H"
#include "phaseModel.H"
#include "rhoThermo.H"

template<class Thermo>
Foam::ActivationOverpotentialModel<Thermo>::ActivationOverpotentialModel
(
    const phaseModel& phase,
    const dictionary& dict
)
:
    activationOverpotentialModel(phase, dict),
    thermo_
    (
        phase.mesh().lookupObject<Thermo>
        (
            IOobject::groupName(basicThermo::dictName, phase.name())
        )
    )
{}


// ************************************************************************* //
