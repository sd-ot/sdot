#include "Poly2dApproximator.h"
#include <Eigen/Cholesky>
#include <iostream>
using namespace boost::multiprecision;

//// nsmake cpp_flag -g3
//// nsmake cpp_flag -O3
//// nsmake lib_name gmp
//// nsmake lib_name mpfr

Poly2dApproximator::Poly2dApproximator( TF epsilon, int degp ) : epsilon( epsilon ), degp( degp ) {
    continuity = 0;
}

void Poly2dApproximator::run_with_values( const std::vector<TF> &radii, const std::vector<TF> &thetas, const std::vector<DF> &values ) {
    // try intervals
    for( std::size_t prev_radius_cut = 0; prev_radius_cut < radii.size(); ) {
        std::size_t beg_radius_cut = prev_radius_cut + 1;
        std::size_t end_radius_cut = radii.size();
        std::vector<TF> coeffs;
        while ( beg_radius_cut < end_radius_cut ) {
            std::size_t mid_radius_cut = beg_radius_cut + ( end_radius_cut + 1 - beg_radius_cut ) / 2;
            if ( error_with( radii, thetas, values, prev_radius_cut, mid_radius_cut, 0, degp, coeffs ) <= epsilon )
                beg_radius_cut = mid_radius_cut;
            else
                end_radius_cut = mid_radius_cut - 1;
        }

        RadiusCut radius_cut;
        radius_cut.index = beg_radius_cut;
        radius_cut.radius = beg_radius_cut < radii.size() ? radii[ beg_radius_cut ] : 1e40;

        // find the theta cuts
        for( std::size_t prev_theta_cut = 0; prev_theta_cut < thetas.size(); ) {
            std::size_t beg_theta_cut = prev_theta_cut + 1;
            std::size_t end_theta_cut = thetas.size();
            std::vector<TF> coeffs;
            while ( beg_theta_cut < end_theta_cut ) {
                std::size_t mid_theta_cut = beg_theta_cut + ( end_theta_cut + 1 - beg_theta_cut ) / 2;
                if ( error_with( radii, thetas, values, prev_radius_cut, beg_radius_cut, prev_theta_cut, mid_theta_cut, coeffs ) <= epsilon )
                    beg_theta_cut = mid_theta_cut;
                else
                    end_theta_cut = mid_theta_cut - 1;
            }
            error_with( radii, thetas, values, prev_radius_cut, beg_radius_cut, prev_theta_cut, beg_theta_cut, coeffs );

            ThetaCut theta_cut;
            theta_cut.coeffs = coeffs;
            theta_cut.index  = beg_theta_cut;
            theta_cut.theta  = beg_theta_cut < thetas.size() ? thetas[ beg_theta_cut ] : 1e40;

            radius_cut.theta_cuts.push_back( theta_cut );
            prev_theta_cut = beg_theta_cut;
        }

        radius_cuts.push_back( radius_cut );
        prev_radius_cut = beg_radius_cut;
    }

    //
    cleanup();
}

Poly2dApproximator::TF Poly2dApproximator::val_for_coef_num( std::size_t num_coeff, TF x, TF y ) const {
    using std::pow;
    using std::sqrt;
    double v = 0.5 * sqrt( 8.0 * num_coeff + 1.0 + 8.0 ) - 1.5;
    std::size_t s = ceil( v ), d = num_coeff - s * ( s + 1 ) / 2;
    return pow( x * x + y * y, s - d ) * pow( x * x - y * y, d );
}

std::size_t Poly2dApproximator::nb_coeffs() const {
    return ( degp + 1 ) * ( degp + 2 ) / 2;
}

void Poly2dApproximator::write_to_stream( std::ostream &os ) const {
    os << radius_cuts;
}

void Poly2dApproximator::cleanup() {
    for( RadiusCut &radius_cut : radius_cuts )
        for( ThetaCut &theta_cut : radius_cut.theta_cuts )
            for( TF &c : theta_cut.coeffs )
                if ( abs( c ) < 1e-40 )
                    c = 0;
}

Poly2dApproximator::TF Poly2dApproximator::error_with( const std::vector<TF> &radii, const std::vector<TF> &thetas, const std::vector<DF> &values, std::size_t beg_radius_index, std::size_t end_radius_index, std::size_t beg_theta_index, std::size_t end_theta_index, std::vector<TF> &coeffs ) const {
    using EM = Eigen::Matrix<TF,Eigen::Dynamic,Eigen::Dynamic>;
    using EV = Eigen::Matrix<TF,Eigen::Dynamic,1>;

    // declaration of the system
    EM M( nb_coeffs(), nb_coeffs() );
    EV V( nb_coeffs() );
    for( unsigned i = 0; i < nb_coeffs(); ++i )
        for( unsigned j = 0; j < nb_coeffs(); ++j )
            M.coeffRef( i, j ) = 0;
    for( unsigned j = 0; j < nb_coeffs(); ++j )
        V[ j ] = 0;

    // values
    for( std::size_t theta_index = beg_theta_index; theta_index < end_theta_index; ++theta_index ) {
        for( std::size_t radius_index = beg_radius_index; radius_index < end_radius_index; ++radius_index ) {
            TF x = radii[ radius_index ] * cos( thetas[ theta_index ] );
            TF y = radii[ radius_index ] * sin( thetas[ theta_index ] );
            for( unsigned i = 0; i < nb_coeffs(); ++i ) {
                for( unsigned j = 0; j < nb_coeffs(); ++j )
                    M.coeffRef( i, j ) += val_for_coef_num( i, x, y ) * val_for_coef_num( j, x, y );
                V[ i ] += val_for_coef_num( i, x, y ) * values[ radii.size() * theta_index + radius_index ][ 0 ];
            }
        }
    }

    // derivatives on the boundaries

    // solver
    Eigen::LDLT<EM> C;
    C.compute( M );
    EV D = C.solve( V );

    // save coeffs
    coeffs.resize( nb_coeffs() );
    for( std::size_t j = 0; j < nb_coeffs(); ++j )
        coeffs[ j ] = D[ j ];

    // compute error
    TF error = 0;
    for( std::size_t theta_index = beg_theta_index; theta_index < end_theta_index; ++theta_index ) {
        for( std::size_t radius_index = beg_radius_index; radius_index < end_radius_index; ++radius_index ) {
            TF x = radii[ radius_index ] * cos( thetas[ theta_index ] );
            TF y = radii[ radius_index ] * sin( thetas[ theta_index ] );
            TF loc = 0;
            for( std::size_t j = 0; j < nb_coeffs(); ++j )
                loc += D[ j ] * val_for_coef_num( j, x, y );
            error = max( error, abs( loc - values[ radii.size() * theta_index + radius_index ][ 0 ] ) );
        }
    }
    return error;
}

Poly2dApproximator::TF Poly2dApproximator::apply( TF x, TF y ) const {
    TF radius = sqrt( x * x + y * y ), theta = atan2( y, x );
    if ( theta < 0 )
        theta += 2 * M_PI;
    for( std::size_t nr = 0; ; ++nr ) {
        if ( nr == radius_cuts.size() - 1 || radius < radius_cuts[ nr ].radius ) {
            for( std::size_t nt = 0; ; ++nt ) {
                if ( nt == radius_cuts[ nr ].theta_cuts.size() - 1 || theta < radius_cuts[ nr ].theta_cuts[ nt ].theta - 1e-10 ) {
                    TF res = 0;
                    for( std::size_t i = 0; i < radius_cuts[ nr ].theta_cuts[ nt ].coeffs.size(); ++i )
                        res += radius_cuts[ nr ].theta_cuts[ nt ].coeffs[ i ] * val_for_coef_num( i, x, y );
                    return res;
                }
            }
        }
    }
    return 0;
}

void Poly2dApproximator::ThetaCut::write_to_stream( std::ostream &os ) const {
    os.precision( 16 );
    os << "\n  ThetaCut( " << (long double)theta << ", [ ";
    for( std::size_t i = 0; i < coeffs.size(); ++i )
        os << ( i ? ", " : "" ) << (long double)coeffs[ i ];
    os << " ] )";
}

void Poly2dApproximator::RadiusCut::write_to_stream( std::ostream &os ) const {
    os.precision( 16 );
    os << "RadiusCut( " << (long double)radius << ", [ " << theta_cuts << "\n] )\n";
}

