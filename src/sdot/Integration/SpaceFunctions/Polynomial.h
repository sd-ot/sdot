#pragma once

#include "../../Support/Assert.h"
#include "../../Support/N.h"
#include "../../Support/N.h"
#include "../FunctionEnum.h"
#include "Constant.h"
#include <cstdint>


namespace sdot {
namespace SpaceFunctions {

/**
 * (canonical)
*/
template<class TF,int nb_coeffs>
class Polynomial {
public:
    operator bool() const { return coeffs[ 0 ]; } // hum

    TF coeffs[ nb_coeffs ];
};

template<class TF,class Func>
auto apply_poly( N<1>, const TF *coeffs, std::size_t delta_coeffs, std::size_t nb_coeffs, Func &&func ) {
    if ( nb_coeffs == 1 )
        return func( Constant<TF>{ coeffs[ 0 ] } );
    TODO;
}

template<class TF,class Func>
auto apply_poly( N<2>, const TF *coeffs, std::size_t delta_coeffs, std::size_t nb_coeffs, Func &&func ) {
    if ( nb_coeffs == 1 )
        return func( Constant<TF>{ coeffs[ 0 ] } );
    if ( nb_coeffs == 6 )  // 1 x y xx xy yy
        return func( Polynomial<TF,6>{ .coeffs = {
            coeffs[ 0 * delta_coeffs ],
            coeffs[ 1 * delta_coeffs ],
            coeffs[ 2 * delta_coeffs ],
            coeffs[ 3 * delta_coeffs ],
            coeffs[ 4 * delta_coeffs ],
            coeffs[ 5 * delta_coeffs ]
        } } );
    TODO;
}

template<class TF,class Func>
auto apply_poly( N<3>, const TF *coeffs, std::size_t delta_coeffs, std::size_t nb_coeffs, Func &&func ) {
    if ( nb_coeffs == 1 )
        return func( Constant<TF>{ coeffs[ 0 ] } );
    TODO;
}

} // namespace SpaceFunctions
} // namespace sdot
