/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
\*---------------------------------------------------------------------------*/

#include "hydrogenCrossoverModel.H"

Foam::autoPtr<Foam::hydrogenCrossoverModel> Foam::hydrogenCrossoverModel::New
(
    const fvMesh& mesh,
    const dictionary& dict
)
{
    word modelType(dict.getOrDefault<word>("type", "none"));

    Info<< "Selecting hydrogen crossover Type: " << modelType << endl;

    auto* ctorPtr = dictionaryConstructorTable(modelType);

    if (!ctorPtr)
    {
        FatalErrorIn("hydrogenCrossoverModel::New")
           << "Unknown hydrogenCrossoverModel type " << modelType << endl
           << "Valid hydrogenCrossoverModel types are:" << endl
           << dictionaryConstructorTablePtr_->sortedToc()
           << exit(FatalError);
    }

    return ctorPtr
    (
        mesh,
        modelType == "none"
      ? dict
      : dict.subDict(modelType + "Coeffs")
    );
}

// ************************************************************************* //
