/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef hydrogenCrossoverModel_H
#define hydrogenCrossoverModel_H

#include "dictionary.H"
#include "dimensionedScalar.H"
#include "fvMesh.H"
#include "regIOobject.H"
#include "runTimeSelectionTables.H"
#include "surfaceFields.H"
#include "volFields.H"

namespace Foam
{

class hydrogenCrossoverModel
:
    public regIOobject
{
    hydrogenCrossoverModel(const hydrogenCrossoverModel&);
    void operator=(const hydrogenCrossoverModel&);

protected:

    const fvMesh& mesh_;

    volScalarField cH2_;
    volScalarField DH2Eff_;
    volScalarField h2Dmdt_;
    volScalarField JH2Diff_;
    volScalarField JH2Drag_;

public:

    TypeName("hydrogenCrossoverModel");

    static const word modelName;

    declareRunTimeSelectionTable
    (
        autoPtr,
        hydrogenCrossoverModel,
        dictionary,
        (
            const fvMesh& mesh,
            const dictionary& dict
        ),
        (mesh, dict)
    );

    hydrogenCrossoverModel
    (
        const fvMesh& mesh,
        const dictionary& dict
    );

    static autoPtr<hydrogenCrossoverModel> New
    (
        const fvMesh& mesh,
        const dictionary& dict
    );

    virtual ~hydrogenCrossoverModel();

    virtual void correct() = 0;

    virtual void solve() = 0;

    inline const volScalarField& cH2() const;
    inline volScalarField& cH2();

    inline const volScalarField& DH2Eff() const;
    inline volScalarField& DH2Eff();

    inline const volScalarField& h2Dmdt() const;
    inline volScalarField& h2Dmdt();

    inline const volScalarField& JH2Diff() const;
    inline volScalarField& JH2Diff();

    inline const volScalarField& JH2Drag() const;
    inline volScalarField& JH2Drag();

    virtual bool writeData(Ostream& os) const;
};

} // End namespace Foam

#include "hydrogenCrossoverModelI.H"

#endif

// ************************************************************************* //
