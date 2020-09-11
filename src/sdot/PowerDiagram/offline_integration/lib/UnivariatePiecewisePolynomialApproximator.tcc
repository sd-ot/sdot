#include "UnivariatePiecewisePolynomialApproximator.h"
#include <Eigen/Cholesky>

template<class TF>
typename UnivariatePiecewisePolynomialApproximator<TF>::Poly UnivariatePiecewisePolynomialApproximator<TF>::run( const std::vector<TF> &x, const std::vector<TF> &y, const std::vector<TF> &p ) {
    Poly poly;

    // find the cuts
    for( std::size_t prev_cut = 0; prev_cut < x.size(); ) {
        std::size_t beg_cut = prev_cut + 1;
        std::size_t end_cut = x.size();
        while ( beg_cut < end_cut ) {
            std::size_t mid_cut = beg_cut + ( end_cut + 1 - beg_cut ) / 2;
            if ( error_with( x, y, p, prev_cut, mid_cut ) <= precision )
                beg_cut = mid_cut;
            else
                end_cut = mid_cut - 1;
        }

        poly.cuts.push_back( {
             beg_cut < x.size() ? x[ beg_cut ] : 1e40,
             make_poly( x, y, p, prev_cut, end_cut ),
             beg_cut,
        } );

        prev_cut = beg_cut;
    }

    return poly;
}

template<class TF>
TF UnivariatePiecewisePolynomialApproximator<TF>::error_with( const std::vector<TF> &x, const std::vector<TF> &y, const std::vector<TF> &p, std::size_t beg_index, std::size_t end_index ) const {
    using std::max;

    // get coeffs (L2 norm)
    std::vector<TF> coeffs = make_poly( x, y, p, beg_index, end_index );

    // compute error
    TF error = 0;
    for( std::size_t index = beg_index; index < end_index; ++index ) {
        TF loc = 0;
        for( std::size_t j = 0; j < coeffs.size(); ++j )
            loc += coeffs[ j ] * pow( x[ index ], j );
        error = max( error, ( index < p.size() ? p[ index ] : TF( 1 ) ) * abs( loc - y[ index ] ) );
    }
    return error;
}

template<class TF>
std::vector<TF> UnivariatePiecewisePolynomialApproximator<TF>::make_poly( const std::vector<TF> &x, const std::vector<TF> &y, const std::vector<TF> &p, std::size_t beg_index, std::size_t end_index ) const {
    using EM = Eigen::Matrix<TF,Eigen::Dynamic,Eigen::Dynamic>;
    using EV = Eigen::Matrix<TF,Eigen::Dynamic,1>;

    // system to try to fit a polynomial
    EM M( degp + 1, degp + 1 );
    EV V( degp + 1 );
    for( unsigned i = 0; i <= degp; ++i )
        for( unsigned j = 0; j <= degp; ++j )
            M.coeffRef( i, j ) = 0;
    for( unsigned j = 0; j <= degp; ++j )
        V[ j ] = 0;
    for( std::size_t index = beg_index; index < end_index; ++index ) {
        TF pond = index < p.size() ? p[ index ] : TF( 1 );
        for( unsigned i = 0; i <= degp; ++i ) {
            for( unsigned j = 0; j <= degp; ++j )
                M.coeffRef( i, j ) += pond * pow( x[ index ], i ) * pow( x[ index ], j );
            V[ i ] += pond * pow( x[ index ], i ) * y[ index ];
        }
    }

    // cholesky
    Eigen::LLT<EM> C;
    C.compute( M );

    // solve and update the weights
    EV D = C.solve( V );

    // save coeffs
    std::vector<TF> res( degp + 1 );
    for( std::size_t j = 0; j < res.size(); ++j )
        res[ j ] = D[ j ];
    return res;
}
