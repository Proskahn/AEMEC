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

    //- Fluid-region contract used to map membrane crossover sources without
    //  relying on cell ordering between regions.
    word cathodeFluidRegion_;
    word anodeFluidRegion_;
    word hydrogenSpecies_;

    //- Dissolved hydrogen concentration in the hydrated ionomer of the
    //  catalyst layers and membrane [mol/m3 of absorbed electrolyte].
    volScalarField cH2_;
    //- Equilibrium dissolved H2 concentration at each membrane interface.
    volScalarField cH2CathodeInterface_;
    volScalarField cH2AnodeInterface_;
    volScalarField DH2Eff_;
    volScalarField JH2Diff_;
    volScalarField JH2Drag_;
    volScalarField JH2Cross_;

    //- Molar source terms mapped into the cathode/anode H2 gas species
    //  equations. Together with the dissolved-field source/sink, they form a
    //  conservative local partition and interphase-transfer coupling.
    volScalarField h2CathodeDmdt_;
    volScalarField h2AnodeDmdt_;

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

    inline const volScalarField& cH2CathodeInterface() const;
    inline volScalarField& cH2CathodeInterface();

    inline const volScalarField& cH2AnodeInterface() const;
    inline volScalarField& cH2AnodeInterface();

    inline const volScalarField& DH2Eff() const;
    inline volScalarField& DH2Eff();

    inline const volScalarField& JH2Diff() const;
    inline volScalarField& JH2Diff();

    inline const volScalarField& JH2Drag() const;
    inline volScalarField& JH2Drag();

    inline const volScalarField& JH2Cross() const;
    inline volScalarField& JH2Cross();

    inline const volScalarField& h2CathodeDmdt() const;
    inline volScalarField& h2CathodeDmdt();

    inline const volScalarField& h2AnodeDmdt() const;
    inline volScalarField& h2AnodeDmdt();

    inline const word& cathodeFluidRegion() const;
    inline const word& anodeFluidRegion() const;
    inline const word& hydrogenSpecies() const;

    virtual bool writeData(Ostream& os) const;
};

} // End namespace Foam

#include "hydrogenCrossoverModelI.H"

#endif

// ************************************************************************* //
