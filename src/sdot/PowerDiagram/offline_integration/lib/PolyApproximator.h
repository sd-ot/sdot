#pragma once

#include <boost/multiprecision/mpfr.hpp>
#include "../../system/Stream.h"
#include <functional>
#include <vector>

/**
  Find approximation of a function given by [(x,y)] values using piecewise polynomials

*/
struct PolyApproximator {
    using               TF              = boost::multiprecision::mpfr_float_100;
    struct              Cut             { void write_to_stream( std::ostream &os ) const; TF position; std::size_t index; std::vector<TF> coeffs; };

    /* */               PolyApproximator( TF epsilon = 1e-6 );

    // compute
    void                run             ( const std::vector<TF> &xs, const std::vector<TF> &ys );

    // use
    TF                  apply           ( TF pos ) const;

    // input parameters
    unsigned            continuity;     ///< 0 -> continuity, 1 -> continuity of the derivative, ...
    TF                  epsilon;        ///< prescribed accuracy
    unsigned            degp;           ///< degree of polynoms

    // output
    std::vector<Cut>    cuts;

private:
    void                cleanup         ();
    TF                  error_with      ( const std::vector<TF> &xs, const std::vector<TF> &ys, std::size_t beg, std::size_t end, std::vector<TF> &coeffs ) const;
    void                force_continuity( const std::vector<TF> &xs, const std::vector<TF> &ys );
};
