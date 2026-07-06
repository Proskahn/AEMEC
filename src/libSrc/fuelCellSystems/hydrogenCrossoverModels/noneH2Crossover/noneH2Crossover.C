/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
\*---------------------------------------------------------------------------*/

#include "noneH2Crossover.H"
#include "addToRunTimeSelectionTable.H"

namespace Foam
{
namespace hydrogenCrossoverModels
{
    defineTypeNameAndDebug(noneH2Crossover, 0);

    addToRunTimeSelectionTable
    (
        hydrogenCrossoverModel,
        noneH2Crossover,
        dictionary
    );
}
}


Foam::hydrogenCrossoverModels::noneH2Crossover::noneH2Crossover
(
    const fvMesh& mesh,
    const dictionary& dict
)
:
    hydrogenCrossoverModel(mesh, dict),
    dict_(dict)
{}


Foam::hydrogenCrossoverModels::noneH2Crossover::~noneH2Crossover()
{}


void Foam::hydrogenCrossoverModels::noneH2Crossover::correct()
{}


void Foam::hydrogenCrossoverModels::noneH2Crossover::solve()
{}

// ************************************************************************* //
