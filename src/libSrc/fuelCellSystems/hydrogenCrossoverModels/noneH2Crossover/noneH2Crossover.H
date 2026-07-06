/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
\*---------------------------------------------------------------------------*/

#ifndef noneH2Crossover_H
#define noneH2Crossover_H

#include "hydrogenCrossoverModel.H"

namespace Foam
{
namespace hydrogenCrossoverModels
{

class noneH2Crossover
:
    public hydrogenCrossoverModel
{
    const dictionary& dict_;

public:

    TypeName("none");

    noneH2Crossover
    (
        const fvMesh& mesh,
        const dictionary& dict
    );

    virtual ~noneH2Crossover();

    virtual void correct();

    virtual void solve();
};

} // End namespace hydrogenCrossoverModels
} // End namespace Foam

#endif

// ************************************************************************* //
