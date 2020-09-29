#include <eigen3/Eigen/Cholesky>
#include "../Support/Stream.h"
#include "../Support/Assert.h"
#include "Arf.h"

namespace sdot {
namespace FunctionEnum {

constexpr unsigned Arf::nb_coeffs;

const Arf::Approximation *Arf::approx_for( TF r ) const {
    for( const Arf::Approximation &ap : approximations )
        if ( ap.end >= r )
            return &ap;
    return nullptr;
}

void Arf::make_approximations_if_not_done() const {
    if ( approximations.size() )
        return;
    mutex.lock();
    if ( approximations.size() ) {
        mutex.unlock();
        return;
    }

    //
    if ( stops.empty() )
        TODO;

    // get approximation of ( xs, ys )
    for( std::size_t i = 0; i < stops.size(); ++i )
        _append_approx( i ? stops[ i - 1 ] : 0, stops[ i ] );

    //add an approx to +oo
    //    Approximation approx;
    //    approx.beg = approximations.back().end;
    //    approx.end = std::numeric_limits<TF>::max();
    //    approx.coeffs[ 0 ] = ;
    //    for( std::size_t i = 1; i < nb_coeffs; ++i )
    //        approx.coeffs[ i ] = 0;
    //    approximations.push_back( approx );

    mutex.unlock();
}

void Arf::_append_approx( TF beg, TF end, std::size_t nb_points ) const {
    using EM = Eigen::Matrix<TF,Eigen::Dynamic,Eigen::Dynamic>;
    using EV = Eigen::Matrix<TF,Eigen::Dynamic,1>;
    using std::max;

    // system to try to fit a polynomial
    EM M( nb_coeffs, nb_coeffs );
    EV V( nb_coeffs );
    for( unsigned i = 0; i < nb_coeffs; ++i )
        for( unsigned j = 0; j < nb_coeffs; ++j )
            M.coeffRef( i, j ) = 0;
    for( unsigned j = 0; j < nb_coeffs; ++j )
        V[ j ] = 0;
    for( std::size_t index = 0; index < nb_points; ++index ) {
        TF x = beg + ( end - beg ) * index / nb_points;
        for( unsigned i = 0; i < nb_coeffs; ++i ) {
            for( unsigned j = 0; j < nb_coeffs; ++j )
                M.coeffRef( i, j ) += pow( x, 2 * i ) * pow( x, 2 * j );
            V[ i ] += pow( x, 2 * i ) * values( x );
        }
    }

    // cholesky
    Eigen::LLT<EM> C;
    C.compute( M );

    // solve and update the weights
    EV D = C.solve( V );

    // save coeffs
    Approximation approx;
    approx.beg = beg;
    approx.end = end;
    for( std::size_t j = 0; j < nb_coeffs; ++j )
        approx.coeffs[ j ] = D[ j ];

    // compute error
    TF error = 0;
    for( std::size_t index = 0; index < nb_points; ++index ) {
        TF loc = 0;
        TF x = beg + ( end - beg ) * index / nb_points;
        for( std::size_t j = 0; j < nb_coeffs; ++j )
            loc += D[ j ] * pow( x, 2 * j );
        error = max( error, abs( loc - values( x ) ) );
    }

    if ( error > 1e-7 ) {
        P( error );
        std::size_t mid = beg + ( end - beg ) / 2;
        _append_approx( beg, mid, nb_points );
        _append_approx( mid, end, nb_points );
        return;
    }

    P( approx.beg, approx.end, approx.coeffs );
    approximations.push_back( approx );
}

}
}
