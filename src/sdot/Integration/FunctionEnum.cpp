#include "../Support/Stream.h"
#include "../Support/Assert.h"
#include <eigen3/Eigen/Cholesky>
#include "FunctionEnum.h"

namespace sdot {
namespace FunctionEnum {

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

    for( std::size_t i = 0; i < stops.size(); ++i )
        _append_approx( i ? stops[ i - 1 ] : 0, stops[ i ] );

    // start_integration
    TF res = 0;
    for( std::size_t i = 0; i < approximations.size(); ++i ) {
        approximations[ i ].start_integration = res;
        for( unsigned p = 0; p <= poly_deg; ++p )
            res += approximations[ i ].coeffs[ p ] / ( 2 * p + 2 ) * ( 
                pow( approximations[ i ].end, 2 * p + 2 ) - 
                pow( approximations[ i ].beg, 2 * p + 2 )
            );
    }

    //add an approx to +oo
    Approximation approx;
    approx.start_integration = res;
    approx.beg = approximations.back().end;
    approx.end = std::numeric_limits<TF>::max();
    for( std::size_t i = 0; i <= poly_deg; ++i )
        approx.coeffs[ i ] = 0;
    approximations.push_back( approx );

    mutex.unlock();
}

const Arf::Approximation *Arf::approx_for( TF r ) const {
    for( const Arf::Approximation &ap : approximations )
        if ( ap.end >= r )
            return &ap;
    return nullptr;
}

void Arf::_append_approx( TF beg, TF end ) const {
    using EM = Eigen::Matrix<TF,Eigen::Dynamic,Eigen::Dynamic>;
    using EV = Eigen::Matrix<TF,Eigen::Dynamic,1>;
    using std::max;

    constexpr unsigned nb_indices = 1000;
    auto xs = [&]( unsigned ind ) {
        return beg + ( end - beg ) * ind / nb_indices;
    };

    auto ys = [&]( unsigned ind ) {
        return values( xs( ind ) );
    };

    // system to try to fit a polynomial
    EM M( poly_deg + 1, poly_deg + 1 );
    EV V( poly_deg + 1 );
    for( unsigned i = 0; i <= poly_deg; ++i )
        for( unsigned j = 0; j <= poly_deg; ++j )
            M.coeffRef( i, j ) = 0;
    for( unsigned j = 0; j <= poly_deg; ++j )
        V[ j ] = 0;
    for( std::size_t index = 0; index < nb_indices; ++index ) {
        for( unsigned i = 0; i <= poly_deg; ++i ) {
            for( unsigned j = 0; j <= poly_deg; ++j )
                M.coeffRef( i, j ) += pow( xs( index ), 2 * i ) * pow( xs( index ), 2 * j );
            V[ i ] += pow( xs( index ), 2 * i ) * ys( index );
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
    for( std::size_t j = 0; j <= poly_deg; ++j )
        approx.coeffs[ j ] = D[ j ];

    // compute error
    TF error = 0;
    for( std::size_t index = 0; index < nb_indices; ++index ) {
        TF loc = 0;
        for( std::size_t j = 0; j <= poly_deg; ++j )
            loc += D[ j ] * pow( xs( index ), 2 * j );
        error = max( error, abs( loc - ys( index ) ) );
    }

    approximations.push_back( approx );
}

}
}
