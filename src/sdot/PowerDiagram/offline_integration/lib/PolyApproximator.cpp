#include "PolyApproximator.h"
#include <Eigen/Cholesky>
#include <iostream>
using namespace boost::multiprecision;

//// nsmake cpp_flag -g3
//// nsmake cpp_flag -O3
//// nsmake lib_name gmp
//// nsmake lib_name mpfr

// hum
std::size_t part_factorial( std::size_t n, unsigned nb_iter ) {
  return nb_iter ? n * part_factorial( n - 1, nb_iter - 1 ) : 1;
}

PolyApproximator::PolyApproximator( TF epsilon ) : epsilon( epsilon ) {
    continuity = 0;
    degp       = 5;
}

void PolyApproximator::run( const std::vector<TF> &xs, const std::vector<TF> &ys ) {
    // try intervals
    for( std::size_t prev_cut = 0; prev_cut < xs.size(); ) {
        std::size_t beg = prev_cut + 1;
        std::size_t end = xs.size();
        std::vector<TF> coeffs;
        while ( beg < end ) {
            std::size_t mid = beg + ( end - beg ) / 2;
            if ( error_with( xs, ys, prev_cut, mid, coeffs ) < epsilon )
                beg = mid + 1;
            else
                end = mid;
        }

        Cut cut;
        cut.index = beg;
        cut.coeffs = coeffs;
        cut.position = xs[ beg - 1 ];

        cuts.push_back( cut );

        prev_cut = beg;
    }

    // update coeffs to get continuity
    force_continuity( xs, ys );

    //
    cleanup();
}

void PolyApproximator::cleanup() {
    for( Cut &cut : cuts )
        for( TF &c : cut.coeffs )
            if ( abs( c ) < 1e-40 )
                c = 0;
}

PolyApproximator::TF PolyApproximator::error_with( const std::vector<TF> &xs, const std::vector<TF> &ys, std::size_t beg_index, std::size_t end_index, std::vector<TF> &coeffs ) const {
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
        for( unsigned i = 0; i <= degp; ++i ) {
            for( unsigned j = 0; j <= degp; ++j )
                M.coeffRef( i, j ) += pow( xs[ index ], i ) * pow( xs[ index ], j );
            V[ i ] += pow( xs[ index ], i ) * ys[ index ];
        }
    }

    // cholesky
    Eigen::LLT<EM> C;
    C.compute( M );

    // solve and update the weights
    EV D = C.solve( V );

    // save coeffs
    coeffs.resize( 0 );
    for( std::size_t j = 0; j <= degp; ++j )
        coeffs.push_back( D[ j ] );

    // compute error
    TF error = 0;
    for( std::size_t index = beg_index; index < end_index; ++index ) {
        TF loc = 0;
        for( std::size_t j = 0; j <= degp; ++j )
            loc += D[ j ] * pow( xs[ index ], j );
        error = max( error, abs( loc - ys[ index ] ) );
    }
    return error;
}

void PolyApproximator::force_continuity( const std::vector<TF> &xs, const std::vector<TF> &ys ) {
    using EM = Eigen::Matrix<TF,Eigen::Dynamic,Eigen::Dynamic>;
    using EV = Eigen::Matrix<TF,Eigen::Dynamic,1>;

    // fit all the polynomials separatly
    EM M( cuts.size() * ( degp + 1 ), cuts.size() * ( degp + 1 ) );
    EV V( cuts.size() * ( degp + 1 ) );
    for( unsigned i = 0; i < cuts.size() * ( degp + 1 ); ++i )
        for( unsigned j = 0; j < cuts.size() * ( degp + 1 ); ++j )
            M.coeffRef( i, j ) = i == j ? 1e-40 : 0;
    for( unsigned j = 0; j < cuts.size() * ( degp + 1 ); ++j )
        V[ j ] = 0;

    for( std::size_t num_cut = 0; num_cut < cuts.size(); ++num_cut ) {
        std::size_t beg_index = num_cut ? cuts[ num_cut - 1 ].index : 0;
        std::size_t end_index = cuts[ num_cut ].index;
        for( std::size_t index = beg_index; index < end_index; ++index ) {
            TF m = index == 0 || index == xs.size() - 2 ? 1e2 * xs.size() : 1;
            for( std::size_t i = 0; i <= degp; ++i ) {
                for( std::size_t j = 0; j <= degp; ++j )
                    M.coeffRef( num_cut * ( degp + 1 ) + i, num_cut * ( degp + 1 ) + j ) += m * pow( xs[ index ], i ) * pow( xs[ index ], j );
                V[ num_cut * ( degp + 1 ) + i ] += m * pow( xs[ index ], i ) * ys[ index ];
            }
        }
    }

    // + continuity
    for( unsigned c = 0; c <= continuity; ++c ) {
        std::vector<TF> vals;
        std::vector<std::size_t> indices;
        for( std::size_t num_cut = 1; num_cut < cuts.size(); ++num_cut ) {
            // coeff * ( p0 + p1 * x^i + ... - q0 + q1 * x^i - ... )²
            vals.resize( 0 );
            indices.resize( 0 );
            TF pos = cuts[ num_cut - 1 ].position;
            for( unsigned i = c; i <= degp; ++i ) {
                indices.push_back( ( num_cut - 1 ) * ( degp + 1 ) + i );
                vals.push_back( TF( +1 ) * part_factorial( i, c ) * pow( pos, i - c ) );
            }
            for( unsigned i = c; i <= degp; ++i ) {
                indices.push_back( ( num_cut - 0 ) * ( degp + 1 ) + i );
                vals.push_back( TF( -1 ) * part_factorial( i, c ) * pow( pos, i - c ) );
            }

            for( std::size_t i = 0; i < indices.size(); ++i )
                for( std::size_t j = 0; j < indices.size(); ++j )
                    M.coeffRef( indices[ i ], indices[ j ] ) += 1e2 * xs.size() * vals[ i ] * vals[ j ];
        }
    }



    // cholesky
    Eigen::LLT<EM> C;
    C.compute( M );
    P( C.rcond() );

    // solve and update the weights
    EV D = C.solve( V );

    for( std::size_t num_cut = 0, ind_D = 0; num_cut < cuts.size(); ++num_cut ) {
        Cut &c = cuts[ num_cut ];
        c.coeffs.clear();
        for( std::size_t j = 0; j <= degp; ++j, ++ind_D )
            c.coeffs.push_back( D[ ind_D ] );
    }
}

PolyApproximator::TF PolyApproximator::apply( TF pos ) const {
    for( std::size_t nc = 0; ; ++nc ) {
        if ( nc == cuts.size() - 1 || pos <= cuts[ nc ].position ) {
            TF res = 0;
            for( std::size_t i = 0; i < cuts[ nc ].coeffs.size(); ++i )
                res += cuts[ nc ].coeffs[ i ] * pow( pos, i );
            return res;
        }
    }
    return 0;
}

void PolyApproximator::Cut::write_to_stream( std::ostream &os ) const {
    os.precision( 16 );
    os << "Cut( " << (long double)position << ", [ ";
    for( std::size_t i = 0; i < coeffs.size(); ++i )
        os << ( i ? ", " : "" ) << (long double)coeffs[ i ];
    os << " ] )";
}
