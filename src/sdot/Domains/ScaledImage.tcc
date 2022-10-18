#include "../Integration/SpaceFunctions/Polynomial.h"
#include "../Support/CrossProdOfRanges.h"
#include "ScaledImage.h"

#include "../Support/Stream.h"

namespace sdot {

template<class Pc>
ScaledImage<Pc>::ScaledImage( Pt min_pt, Pt max_pt, const TF *data, std::array<TI,dim> sizes, TI nb_coeffs ) : nb_coeffs( nb_coeffs ), min_pt( min_pt ), max_pt( max_pt ), sizes( sizes ), data( data, data + nb_pixels() * nb_coeffs ) {
    englobing_polyheron = { typename CP::Box{ min_pt, max_pt }, typename Pc::CI( -1 ) };
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
    TF res = 0;
    if ( nb_coeffs == 1 ) {
        for( const TF &v : data )
            res += v;
        res /= data.size();
    } else if ( dim == 2 && nb_coeffs == 6 ) {
        TF dx = ( max_pt[ 0 ] - min_pt[ 0 ] ) / sizes[ 0 ];
        TF dy = ( max_pt[ 1 ] - min_pt[ 1 ] ) / sizes[ 1 ];
        for( TI yi = 0, o = 0; yi < sizes[ 1 ]; ++yi ) {
            for( TI xi = 0; xi < sizes[ 0 ]; ++xi, ++o ) {
                TF x = min_pt[ 0 ] + ( max_pt[ 0 ] - min_pt[ 0 ] ) / sizes[ 0 ] * xi;
                TF y = min_pt[ 1 ] + ( max_pt[ 1 ] - min_pt[ 1 ] ) / sizes[ 1 ] * yi;

                TF R_0 = data[ nb_pixels() * 3 + o ]; TF R_1 = data[ nb_pixels() * 1 + o ]; TF R_2 = data[ nb_pixels() * 0 + o ]; TF R_3 = data[ nb_pixels() * 5 + o ];
                TF R_4 = dx; TF R_5 = pow(R_4,2); R_5 = R_0*R_5; R_4 = 0.5*R_4;
                TF R_6 = x; R_4 = R_6+R_4; R_0 = R_0*R_4; R_0 = R_1+R_0;
                R_0 = R_4*R_0; R_0 = R_2+R_0; R_2 = data[ nb_pixels() * 4 + o ]; R_4 = R_2*R_4;
                R_2 = data[ nb_pixels() * 2 + o ]; R_4 = R_2+R_4; R_2 = dy; R_1 = pow(R_2,2);
                R_1 = R_3*R_1; R_1 = R_5+R_1; R_1 = (1.0/12.0)*R_1; R_2 = 0.5*R_2;
                R_5 = y; R_2 = R_5+R_2; R_3 = R_3*R_2; R_3 = R_4+R_3;
                R_3 = R_2*R_3; R_0 = R_3+R_0; R_1 = R_0+R_1; res += R_1;
            }
        }
        res /= nb_pixels();
    } else {
        TODO;
    }

    for( std::size_t i = 0; i < dim; ++i )
        res *= max_pt[ i ] - min_pt[ i ];

    return res;
}

template<class Pc>
typename ScaledImage<Pc>::TF ScaledImage<Pc>::coeff_at( const Pt &pos ) const {
    TI index = 0, acc = 1;
    for( std::size_t d = 0; d < dim; ++d ) {
        TI p = std::max( pos[ d ] - min_pt[ d ], TF( 0 ) ) * sizes[ d ] / ( max_pt[ d ] - min_pt[ d ] );
        p = std::min( p, sizes[ d ] - 1 );
        index += acc * p;
        acc *= sizes[ d ];
    }
    // index += acc * num_coeff;

    if ( nb_coeffs == 1 )
        return data[ index ];

    if ( nb_coeffs == 6 )
        return data[ index + acc * 0 ] +
               data[ index + acc * 1 ] * pos[ 0 ] +
               data[ index + acc * 2 ] * pos[ 1 ] +
               data[ index + acc * 3 ] * pos[ 0 ] * pos[ 0 ] +
               data[ index + acc * 4 ] * pos[ 0 ] * pos[ 1 ] +
               data[ index + acc * 5 ] * pos[ 1 ] * pos[ 1 ] ;

    TODO;
    return 0;
}

template<class Pc> template<class F>
void ScaledImage<Pc>::for_each_intersection( CP &cp, const F &f ) const {
    using std::min;
    using std::max;

    if ( nb_pixels() == 1 ) {
        return SpaceFunctions::apply_poly( N<dim>(), data.data(), nb_pixels(), nb_coeffs, [&]( const auto &sf ) {
            return f( cp, sf );
        } );
    }

    // find min_y, max_y
    Pt ps;
    std::array<TI,dim> min_i;
    std::array<TI,dim> max_i;
    Pt min_pos = cp.min_position();
    Pt max_pos = cp.max_position();
    for( std::size_t d = 0; d < dim; ++d ) {
        min_i[ d ] = ( min_pos[ d ] - min_pt[ d ] ) * sizes[ d ] / ( max_pt[ d ] - min_pt[ d ] );
        max_i[ d ] = ( max_pos[ d ] - min_pt[ d ] ) * sizes[ d ] / ( max_pt[ d ] - min_pt[ d ] );
        min_i[ d ] = max( TI( 1 ), min_i[ d ] ) - 1;
        max_i[ d ] = min( sizes[ d ], max_i[ d ] + 2 );
        ps[ d ] = ( max_pt[ d ] - min_pt[ d ] ) / sizes[ d ];
    }

    //
    CP ccp;
    CrossProdOfRanges<TI,dim> cr( min_i, max_i );
    cr.for_each( [&]( auto p ) {
        Pt pf;
        TI num_pix = 0, acc = 1;
        for( std::size_t d = 0; d < dim; ++d ) {
            num_pix += acc * p[ d ];
            acc *= sizes[ d ];
            pf[ d ] = p[ d ];
        }

        ccp = { typename CP::Box{
            min_pt + ps * ( pf + TF( 0 ) ),
            min_pt + ps * ( pf + TF( 1 ) )
        }, typename Pc::CI( -1 ) };

        ccp.intersect_with( cp );

        SpaceFunctions::apply_poly( N<dim>(), data.data() + num_pix, acc, nb_coeffs, [&]( const auto &sf ) {
            f( ccp, sf );
        } );
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
