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

Class
    Foam::InterfaceCompositionModel

Description
    Base class for interface composition models, templated on the two
    thermodynamic models either side of the interface.

SourceFiles
    interfaceCompositionModelTemplate.C

\*---------------------------------------------------------------------------*/

#ifndef interfaceCompositionModelTemplate_H
#define interfaceCompositionModelTemplate_H

#include "interfaceCompositionModel.H"

namespace Foam
{

class phaseModel;
class phasePair;
template<class ThermoType> class pureMixture;
template<class ThermoType> class multiComponentMixture;

template<class Thermo, class OtherThermo>
class InterfaceCompositionModel
:
    public interfaceCompositionModel
{
protected:

        const Thermo& thermo_;
        const OtherThermo& otherThermo_;
        const dimensionedScalar Le_;

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

        InterfaceCompositionModel
        (
            const dictionary& dict,
            const phasePair& pair
        );

        virtual ~InterfaceCompositionModel()
        {}

        virtual tmp<volScalarField> dY
        (
            const word& speciesName,
            const volScalarField& Tf
        ) const;

        virtual tmp<volScalarField> D(const word& speciesName) const;

        virtual tmp<volScalarField> L
        (
            const word& speciesName,
            const volScalarField& Tf
        ) const;

        virtual void addMDotL
        (
            const volScalarField& K,
            const volScalarField& Tf,
            volScalarField& mDotL,
            volScalarField& mDotLPrime
        ) const;
};

} // End namespace Foam

#define makeInterfaceCompositionType(Type, Thermo, Comp, Mix, Phys, OtherThermo, OtherComp, OtherMix, OtherPhys)\
                                                                               \
    typedef Thermo<Comp, SpecieMixture<Mix<Phys> > >                           \
        Type##Thermo##Comp##Mix##Phys;                                         \
                                                                               \
    typedef OtherThermo<OtherComp, OtherMix<OtherPhys> >                       \
        Type##Other##OtherThermo##OtherComp##OtherMix##OtherPhys;              \
                                                                               \
    addInterfaceCompositionToRunTimeSelectionTable                             \
    (                                                                          \
        Type,                                                                  \
        Type##Thermo##Comp##Mix##Phys,                                         \
        Type##Other##OtherThermo##OtherComp##OtherMix##OtherPhys               \
    )

#define makeSpecieInterfaceCompositionType(Type, Thermo, Comp, Mix, Phys, OtherThermo, OtherComp, OtherMix, OtherPhys)\
                                                                               \
    typedef Thermo<Comp, SpecieMixture<Mix<Phys> > >                           \
        Type##Thermo##Comp##Mix##Phys;                                         \
                                                                               \
    typedef OtherThermo<OtherComp, SpecieMixture<OtherMix<OtherPhys> > >       \
        Type##Other##OtherThermo##OtherComp##OtherMix##OtherPhys;              \
                                                                               \
    addInterfaceCompositionToRunTimeSelectionTable                             \
    (                                                                          \
        Type,                                                                  \
        Type##Thermo##Comp##Mix##Phys,                                         \
        Type##Other##OtherThermo##OtherComp##OtherMix##OtherPhys               \
    )

#define addInterfaceCompositionToRunTimeSelectionTable(Type, Thermo, OtherThermo)\
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
        interfaceCompositionModel,                                             \
        Type##Thermo##OtherThermo,                                             \
        dictionary                                                             \
    )

#ifdef NoRepository
    #include "interfaceCompositionModelTemplate.C"
#endif

#endif

// ************************************************************************* //
