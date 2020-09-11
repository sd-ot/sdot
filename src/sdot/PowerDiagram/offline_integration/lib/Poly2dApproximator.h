#pragma once

#include <boost/multiprecision/mpfr.hpp>
using boost::multiprecision::sqrt;
#include <matplotlibcpp.h>

#include "../../system/Stream.h"
#include "FADBAD++/fadiff.h"
#include "FADBAD++/badiff.h"
#include <functional>
#include <iostream>
#include <vector>

/**
  Find approximation of a function given by [(x,y)] values using piecewise polynomials

*/
struct Poly2dApproximator {
    using                   TF                = boost::multiprecision::mpfr_float_100;
    using                   DF                = std::array<TF,1+2+3>; // coeffs for { 1, x, y, x*x, x*y, y*y }
    using                   FB                = fadbad::B<fadbad::F<TF,2>>;
    struct                  ThetaCut          { void write_to_stream( std::ostream &os ) const; TF theta ; std::size_t index; std::vector<TF> coeffs; };
    struct                  RadiusCut         { void write_to_stream( std::ostream &os ) const; TF radius; std::size_t index; std::vector<ThetaCut> theta_cuts; };

    /* */                   Poly2dApproximator( TF epsilon = 1e-6, int degp = 7 );

    // compute
    void                    run_with_values   ( const std::vector<TF> &radii, const std::vector<TF> &thetas, const std::vector<DF> &values ); ///< values.size() = radius.size() * theta.size() (values[ radius.size() * theta + radius ])
    template<class Fu> void run_with_func     ( const std::vector<TF> &radii, const std::vector<TF> &thetas, const Fu &func ); ///<

    TF                      val_for_coef_num  ( std::size_t num_coeff, TF x, TF y ) const;
    std::size_t             nb_coeffs         () const;

    // use
    void                    write_to_stream   ( std::ostream &os ) const;
    TF                      apply             ( TF x, TF y ) const;

    // input parameters
    unsigned                continuity;       ///< 0 -> C0, 1 -> C1 (continuity of the derivative ), ...
    TF                      epsilon;          ///< prescribed accuracy
    unsigned                degp;             ///< degree of polynoms

    // output
    std::vector<RadiusCut>  radius_cuts;

private:
    void                    cleanup           ();
    TF                      error_with        ( const std::vector<TF> &radii, const std::vector<TF> &thetas, const std::vector<DF> &values, std::size_t beg_radius_index, std::size_t end_radius_index, std::size_t beg_theta_index, std::size_t end_theta_index, std::vector<TF> &coeffs ) const;
    void                    force_continuity  ( const std::vector<TF> &xs, const std::vector<TF> &ys );
};

template<class Fu>
void Poly2dApproximator::run_with_func( const std::vector<TF> &radii, const std::vector<TF> &thetas, const Fu &func ) {
    std::vector<DF> values;
    values.reserve( thetas.size() * radii.size() );

    for( std::size_t it = 0; it < thetas.size(); ++it ) {
        for( std::size_t ir = 0; ir < radii.size(); ++ir ) {
            TF vx = radii[ ir ] * cos( thetas[ it ] );
            TF vy = radii[ ir ] * sin( thetas[ it ] );

            FB x = vx;
            FB y = vy;
            x.x().diff( 0 ); // Second order wrt. x
            y.x().diff( 1 ); // Second order wrt. y

            FB vf = func( x, y );
            vf.diff( 0, 1 );  // Differentiate f

            values.push_back( {
                vf.x().x(),  // 1
                x.d(0).x(),  // dx
                y.d(0).x(),  // dy
                x.d(0).d(0), // dxdx
                x.d(0).d(1), // dxdy
                y.d(0).d(1), // dydy
            } );
        }
    }

    run_with_values( radii, thetas, values );
}

