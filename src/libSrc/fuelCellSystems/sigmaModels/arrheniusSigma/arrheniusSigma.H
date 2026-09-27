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
    Foam::sigmaModels::arrheniusSigma

Description
    Temperature-dependent electric conductivity using the reference-state
    Arrhenius relation

    \f[
        \sigma(T) = \sigma_{ref}
        \exp\left[-\frac{E_a}{R}
        \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right].
    \f]

    The required coefficients are \c sigmaRef [S/m], \c TRef [K], and
    \c Ea [J/mol].  The optional \c T entry selects the temperature field
    and defaults to \c T.

    The underlying Arrhenius form is given in Wang et al., Nature
    Communications 16, 934 (2025), DOI: 10.1038/s41467-025-56262-6.
    The reference-state form is obtained by dividing that relation by its
    value at TRef.

SourceFiles
    arrheniusSigma.C

\*---------------------------------------------------------------------------*/

#ifndef arrheniusSigma_H
#define arrheniusSigma_H

#include "sigmaModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace sigmaModels
{

/*---------------------------------------------------------------------------*\
                         Class arrheniusSigma Declaration
\*---------------------------------------------------------------------------*/

class arrheniusSigma
:
    public sigmaModel
{
    // Private data

        //- Name of the temperature field
        word TName_;

        //- Conductivity at the reference temperature
        dimensionedScalar sigmaRef_;

        //- Reference temperature
        dimensionedScalar TRef_;

        //- Activation energy for ionic conduction
        dimensionedScalar Ea_;


public:

    //- Runtime type information
    TypeName("arrheniusSigma");


    // Constructors

        //- Construct from components
        arrheniusSigma
        (
            const fvMesh& mesh,
            const dictionary& sigmaDictionary
        );


    //- Destructor
    virtual ~arrheniusSigma();


    // Member Functions

        //- Update conductivity in the configured cell zone(s)
        virtual void correct(volScalarField& sigmaField) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace sigmaModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
