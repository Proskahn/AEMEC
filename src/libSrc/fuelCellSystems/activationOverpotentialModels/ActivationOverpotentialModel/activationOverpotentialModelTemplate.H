/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright held by the original author
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef activationOverpotentialModelTemplate_H
#define activationOverpotentialModelTemplate_H

#include "activationOverpotentialModel.H"

namespace Foam
{

class phaseModel;

template<class Thermo>
class ActivationOverpotentialModel
:
    public activationOverpotentialModel
{
protected:

        const Thermo& thermo_;


public:

        ActivationOverpotentialModel
        (
            const phaseModel& phase,
            const dictionary& dict
        );

        virtual ~ActivationOverpotentialModel()
        {}
};

} // End namespace Foam

#define makeActivationOverpotentialType(Type, Thermo, Comp, Mix, Phys)         \
                                                                               \
    typedef Thermo<Comp, SpecieMixture<Mix<Phys> > >                           \
        Type##Thermo##Comp##Mix##Phys;                                         \
                                                                               \
    addActivationOverpotentialToRunTimeSelectionTable                          \
    (                                                                          \
        Type,                                                                  \
        Type##Thermo##Comp##Mix##Phys                                          \
    )

#define addActivationOverpotentialToRunTimeSelectionTable(Type, Thermo)        \
                                                                               \
    typedef Type<Thermo> Type##Thermo;                                         \
                                                                               \
    defineTemplateTypeNameAndDebugWithName                                     \
    (                                                                          \
        Type##Thermo,                                                          \
        (                                                                      \
            word(Type##Thermo::typeName_()) + "<"                              \
          + word(Thermo::typeName) + ">"                                       \
        ).c_str(),                                                             \
        0                                                                      \
    );                                                                         \
                                                                               \
    addToRunTimeSelectionTable                                                 \
    (                                                                          \
        activationOverpotentialModel,                                          \
        Type##Thermo,                                                          \
        dictionary                                                             \
    )

#ifdef NoRepository
    #include "activationOverpotentialModelTemplate.C"
#endif

#endif

// ************************************************************************* //
