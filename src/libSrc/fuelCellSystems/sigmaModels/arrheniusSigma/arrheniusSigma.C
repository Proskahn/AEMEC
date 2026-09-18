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

#include "arrheniusSigma.H"
#include "addToRunTimeSelectionTable.H"
#include "constants.H"

// * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace sigmaModels
{
    defineTypeNameAndDebug(arrheniusSigma, 0);

    addToRunTimeSelectionTable
    (
        sigmaModel,
        arrheniusSigma,
        dictionary
    );
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::sigmaModels::arrheniusSigma::arrheniusSigma
(
    const fvMesh& mesh,
    const dictionary& sigmaDictionary
)
:
    sigmaModel(mesh, sigmaDictionary),
    TName_(sigmaDictionary_.lookupOrDefault<word>("T", "T")),
    sigmaRef_
    (
        "sigmaRef",
        dimCurrent*dimCurrent*dimTime/(dimEnergy*dimLength),
        sigmaDictionary_
    ),
    TRef_("TRef", dimTemperature, sigmaDictionary_),
    Ea_("Ea", dimEnergy/dimMoles, sigmaDictionary_)
{
    if (sigmaRef_.value() <= 0)
    {
        FatalIOErrorInFunction(sigmaDictionary_)
            << "sigmaRef must be greater than zero" << exit(FatalIOError);
    }

    if (TRef_.value() <= 0)
    {
        FatalIOErrorInFunction(sigmaDictionary_)
            << "TRef must be greater than zero Kelvin" << exit(FatalIOError);
    }

    if (Ea_.value() < 0)
    {
        FatalIOErrorInFunction(sigmaDictionary_)
            << "Ea must be non-negative" << exit(FatalIOError);
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::sigmaModels::arrheniusSigma::~arrheniusSigma()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::sigmaModels::arrheniusSigma::correct
(
    volScalarField& sigmaField
) const
{
    const scalarField& T = mesh_.lookupObject<volScalarField>(TName_);
    const scalar activationTemperature =
        Ea_.value()/constant::physicoChemical::R.value();

    forAll(cellZoneIDs_, zoneI)
    {
        const labelList& cells = mesh_.cellZones()[cellZoneIDs_[zoneI]];

        forAll(cells, i)
        {
            const label cellI = cells[i];

            if (T[cellI] <= 0)
            {
                FatalErrorInFunction
                    << "Temperature field '" << TName_ << "' contains "
                    << T[cellI] << " K in cell " << cellI
                    << exit(FatalError);
            }

            sigmaField[cellI] = sigmaRef_.value()*Foam::exp
            (
                -activationTemperature
               *(1.0/T[cellI] - 1.0/TRef_.value())
            );
        }
    }
}

// ************************************************************************* //
