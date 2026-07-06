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

#ifndef nernstModelTemplate_H
#define nernstModelTemplate_H

#include "nernstModel.H"

namespace Foam
{

class phaseModel;
template<class ThermoType> class pureMixture;
template<class ThermoType> class multiComponentMixture;

template<class Thermo, class OtherThermo>
class NernstModel
:
    public nernstModel
{
protected:

        const Thermo& thermo_;
        const OtherThermo& otherThermo_;

        template<class ThermoType>
        const typename pureMixture<ThermoType>::thermoType& getLocalThermo
        (
            const word& speciesName,
            const pureMixture<ThermoType>& globalThermo
        ) const;

        template<class ThermoType>
        const typename multiComponentMixture<ThermoType>::thermoType&
        getLocalThermo
        (
            const word& speciesName,
            const multiComponentMixture<ThermoType>& globalThermo
        ) const;


public:

        NernstModel
        (
            const phaseModel& phase1,
            const phaseModel& phase2,
            const dictionary& dict
        );

        virtual ~NernstModel()
        {}
};

} // End namespace Foam

#define makeNernstType(Type, Thermo, Comp, Mix, Phys, OtherThermo, OtherComp, OtherMix, OtherPhys)\
                                                                               \
    typedef Thermo<Comp, SpecieMixture<Mix<Phys> > >                           \
        Type##Thermo##Comp##Mix##Phys;                                         \
                                                                               \
    typedef OtherThermo<OtherComp, OtherMix<OtherPhys> >                       \
        Type##Other##OtherThermo##OtherComp##OtherMix##OtherPhys;              \
                                                                               \
    addNernstToRunTimeSelectionTable                                           \
    (                                                                          \
        Type,                                                                  \
        Type##Thermo##Comp##Mix##Phys,                                         \
        Type##Other##OtherThermo##OtherComp##OtherMix##OtherPhys               \
    )

#define makeSpecieNernstType(Type, Thermo, Comp, Mix, Phys, OtherThermo, OtherComp, OtherMix, OtherPhys)\
                                                                               \
    typedef Thermo<Comp, SpecieMixture<Mix<Phys> > >                           \
        Type##Thermo##Comp##Mix##Phys;                                         \
                                                                               \
    typedef OtherThermo<OtherComp, SpecieMixture<OtherMix<OtherPhys> > >       \
        Type##Other##OtherThermo##OtherComp##OtherMix##OtherPhys;              \
                                                                               \
    addNernstToRunTimeSelectionTable                                           \
    (                                                                          \
        Type,                                                                  \
        Type##Thermo##Comp##Mix##Phys,                                         \
        Type##Other##OtherThermo##OtherComp##OtherMix##OtherPhys               \
    )

#define addNernstToRunTimeSelectionTable(Type, Thermo, OtherThermo)            \
                                                                               \
    typedef Type<Thermo, OtherThermo>                                          \
        Type##Thermo##OtherThermo;                                             \
                                                                               \
    defineTemplateTypeNameAndDebugWithName                                     \
    (                                                                          \
        Type##Thermo##OtherThermo,                                             \
        (                                                                      \
            word(Type##Thermo##OtherThermo::typeName_()) + "<"                 \
          + word(Thermo::typeName) + ","                                       \
          + word(OtherThermo::typeName) + ">"                                  \
        ).c_str(),                                                             \
        0                                                                      \
    );                                                                         \
                                                                               \
    addToRunTimeSelectionTable                                                 \
    (                                                                          \
        nernstModel,                                                           \
        Type##Thermo##OtherThermo,                                             \
        dictionary                                                             \
    )

#ifdef NoRepository
    #include "nernstModelTemplate.C"
#endif

#endif

// ************************************************************************* //
