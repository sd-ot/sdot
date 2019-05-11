#include "../Integration/SpaceFunctions/Constant.h"
#include "../Support/CrossProdOfRanges.h"
#include "ScaledImage.h"

#include "../Support/Stream.h"

namespace sdot {

template<class Pc>
ScaledImage<Pc>::ScaledImage( Pt min_pt, Pt max_pt, const TF *data, std::array<TI,dim> sizes ) : min_pt( min_pt ), max_pt( max_pt ), sizes( sizes ), data( data, data + nb_pixels() ) {
    englobing_polyheron = typename CP::Box{ min_pt, max_pt };
}

template<class Pc>
const typename ScaledImage<Pc>::CP& ScaledImage<Pc>::englobing_convex_polyhedron() const {
    return englobing_polyheron;
}

template<class Pc>
typename ScaledImage<Pc>::Pt ScaledImage<Pc>::min_position() const {
    return min_pt;
}

template<class Pc>
typename ScaledImage<Pc>::Pt ScaledImage<Pc>::max_position() const {
    return max_pt;
}

template<class Pc>
typename ScaledImage<Pc>::TI ScaledImage<Pc>::nb_pixels() const {
    TI res = 1;
    for( std::size_t i = 0; i < dim; ++i )
        res *= sizes[ i ];
    return res;
}

template<class Pc>
typename ScaledImage<Pc>::TF ScaledImage<Pc>::measure() const {
    TF res = 1;
    for( std::size_t i = 0; i < dim; ++i )
        res *= max_pt[ i ] - min_pt[ i ];
    return res;
}

template<class Pc>
typename ScaledImage<Pc>::TF ScaledImage<Pc>::coeff_at( const Pt &pos ) const {
    TF res = 0;
    TODO;
    return res;
}

template<class Pc> template<class F>
void ScaledImage<Pc>::for_each_intersection( CP &cp, const F &f ) const {
    using std::min;
    using std::max;

    if ( nb_pixels() == 1 )
        return f( cp, SpaceFunctions::Constant<TF>{ data[ 0 ] } );


    // find min_y, max_y
    Pt ps;
    std::array<TI,dim> min_i;
    std::array<TI,dim> max_i;
    Pt min_pos = cp.min_position();
    Pt max_pos = cp.max_position();
    for( std::size_t d = 0; d < dim; ++d ) {
        min_i[ d ] = ( min_pos[ d ] - min_pt[ d ] ) * sizes[ d ] / ( max_pt[ d ] - min_pt[ d ] );
        max_i[ d ] = ( max_pos[ d ] - min_pt[ d ] ) * sizes[ d ] / ( max_pt[ d ] - min_pt[ d ] );
        min_i[ d ] = max( TI( 0 ), min_i[ d ] );
        max_i[ d ] = min( sizes[ d ], max_i[ d ] + 1 );
        ps[ d ] = ( max_pt[ d ] - min_pt[ d ] ) / sizes[ d ];
    }

    //
    CP ccp;
    CrossProdOfRanges<TI,dim> cr( min_i, max_i );
    cr.for_each( [&]( auto p ) {
        Pt pf;
        TI num_pix = 0;
        for( std::size_t d = 0, acc = 1; d < dim; ++d ) {
            num_pix += acc * p[ d ];
            acc *= sizes[ d ];
            pf[ d ] = p[ d ];
        }

        ccp = { typename CP::Box{
            min_pt + ps * ( pf + TF( 0 ) ),
            min_pt + ps * ( pf + TF( 1 ) )
        }, typename Pc::CI(-1 ) };
        ccp.intersect_with( cp );

        f( ccp, SpaceFunctions::Constant<TF>{ data[ num_pix ] } );
    } );
}

template<class Pc> template<class V>
void ScaledImage<Pc>::display_boundaries( V &vtk_output ) const {
    englobing_polyheron.display( vtk_output, { TF( 0 ) }, false );
}

template<class Pc> template<class V>
void ScaledImage<Pc>::display_coeffs( V &vtk_output ) const {
    display_boundaries( vtk_output );
}

} // namespace sdot
