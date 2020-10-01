#include <eigen3/Eigen/Cholesky>
#include "../Support/Stream.h"
#include "../Support/Assert.h"
#include "Arfd.h"

namespace sdot {
namespace FunctionEnum {

constexpr unsigned Arfd::nb_coeffs;

Arfd::Arfd( const std::function<TF( TF r )> &values, const std::function<TF( TF w )> &inp_scaling, const std::function<TF( TF w )> &out_scaling,
            const std::function<TF( TF r )> &der_values, const std::function<TF( TF w )> &der_inp_scaling, const std::function<TF( TF w )> &der_out_scaling,
            const std::vector<TF> &stops, TF prec ) :  inp_scaling( inp_scaling ), out_scaling( out_scaling ), values( values ), stops( stops ), prec( prec ) {
    make_approximations_if_not_done();

    if ( der_values ) {
        std::function<TF( TF r )> vf;
        der_w = std::make_unique<Arfd>( der_values, der_inp_scaling, der_out_scaling, vf, vf, vf, stops, prec );
    }
}

Arfd::Arfd() {
    prec = 1e-9;
}

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
    if ( stops.empty() ) {
        mutex.unlock();
        TODO;
    }

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
    using std::log;
    using std::abs;

    // shape functions
    auto sc = [&]( int i ) {
        TF b = beg ? abs( log( beg ) ) : 0;
        TF e = abs( log( end ) );
        return pow( b > e ? beg : end, 2 * i );
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
        TF x = beg + ( end - beg ) * ( index + 0.5 ) / nb_points;
        for( unsigned i = 0; i < nb_coeffs; ++i ) {
            for( unsigned j = 0; j < nb_coeffs; ++j )
                M.coeffRef( i, j ) += vx( x, i ) * vx( x, j );
            V[ i ] += vx( x, i ) * values( x );
        }
    }

    for( unsigned i = 0; i < nb_coeffs; ++i )
        M.coeffRef( i, i ) += 1e-9;

    // cholesky
    Eigen::LLT<EM> C;
    C.compute( M );

    // solve and update the weights
    EV D = C.solve( V );

    // compute error
    TF error = 0;
    for( std::size_t index = 0; index < nb_points; ++index ) {
        TF x = beg + ( end - beg ) * ( index + 0.5 ) / nb_points;
        TF loc = 0;
        for( std::size_t i = 0; i < nb_coeffs; ++i )
            loc += D[ i ] * vx( x, i );
        error = max( error, abs( loc - values( x ) ) );
    }

    if ( error > prec && end - beg > 1e-2 ) {
        TF mid = beg + ( end - beg ) / 2;
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
        approx.integration_coeffs[ i + 1 ] = D[ i ] / sc( i ) / ( 2 * i + 2 );
        approx.value_coeffs[ i ] = D[ i ] / sc( i );
    }
    approximations.push_back( approx );

    // update sum
    for( std::size_t i = 0; i < nb_coeffs; ++i )
        sum += D[ i ] / sc( i ) / ( 2 * i + 2 ) * ( pow( end, 2 * i + 2 ) - pow( beg, 2 * i + 2 ) );
}

Arfd::TF Arfd::approx_value( TF r ) const {
    TF res = 0;
    const Approximation *af = approx_for( r );
    for( std::size_t j = 0; j < nb_coeffs; ++j )
        res += af->value_coeffs[ j ] * pow( r, 2 * j );
    return res;
}

}
}
