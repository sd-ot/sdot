#include <eigen3/Eigen/Cholesky>
#include "../Support/Stream.h"
#include "../Support/Assert.h"
#include "Arfd.h"

namespace sdot {
namespace FunctionEnum {

constexpr unsigned Arfd::nb_coeffs;

const Arfd::Approximation *Arfd::approx_for( TF r ) const {
    for( const Arfd::Approximation &ap : approximations )
        if ( ap.end >= r )
            return &ap;
    return nullptr;
}

void Arfd::make_approximations_if_not_done() const {
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

    // make points ( xs, ys ) of int r V( r )
    TF sum = 0;
    for( std::size_t i = 0; i < stops.size(); ++i )
        _append_approx( sum, i ? stops[ i - 1 ] : 0, stops[ i ] );

    //
    _append_approx( sum, stops.back(), 1e20 * stops.back() );
    approximations.back().end = std::numeric_limits<TF>::max();

    mutex.unlock();
}

void Arfd::_append_approx( TF &sum, TF beg, TF end, unsigned nb_points ) const {
    using EM = Eigen::Matrix<TF,Eigen::Dynamic,Eigen::Dynamic>;
    using EV = Eigen::Matrix<TF,Eigen::Dynamic,1>;
    using std::max;
    using std::pow;

    // shape functions
    auto sc = [&]( int i ) {
        return pow( end, 2 * i );
    };

    auto vx = [&]( TF x, int i ) {
        return pow( x, 2 * i ) / sc( i );
    };

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
                M.coeffRef( i, j ) += vx( x, i ) * vx( x, j );
            V[ i ] += vx( x, i ) * values( x );
        }
    }

    // cholesky
    Eigen::LLT<EM> C;
    C.compute( M );

    // solve and update the weights
    EV D = C.solve( V );

    // compute error
    TF error = 0;
    for( std::size_t index = 0; index < nb_points; ++index ) {
        TF x = beg + ( end - beg ) * index / nb_points;
        TF loc = 0;
        for( std::size_t i = 0; i < nb_coeffs; ++i )
            loc += D[ i ] * vx( x, i );
        error = max( error, abs( loc - values( x ) ) );
    }

    P( beg, end, error );
    if ( error > 1e-10 ) {
        std::size_t mid = beg + ( end - beg ) / 2;
        _append_approx( sum, beg, mid, nb_points );
        _append_approx( sum, mid, end, nb_points );
        return;
    }

    // save coeffs
    Approximation approx;
    approx.beg = beg;
    approx.end = end;
    approx.integration_coeffs[ 0 ] = sum;
    for( std::size_t i = 0; i < nb_coeffs; ++i ) {
        approx.integration_coeffs[ i + 1 ] = D[ i ] * sc( i ) / ( 2 * i + 2 );
        approx.value_coeffs[ i ] = D[ i ] * sc( i );
    }
    approximations.push_back( approx );

    // update sum
    for( std::size_t i = 0; i < nb_coeffs; ++i )
        sum += D[ i ] * sc( i ) / ( 2 * i + 2 ) * ( pow( end, 2 * i + 2 ) - pow( beg, 2 * i + 2 ) );
}

}
}
