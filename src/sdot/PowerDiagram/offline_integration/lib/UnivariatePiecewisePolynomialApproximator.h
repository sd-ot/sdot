#pragma once

#include "UnivariatePiecewisePolynomial.h"

/**
*/
template<class TF>
class UnivariatePiecewisePolynomialApproximator {
public:
    using           Poly = UnivariatePiecewisePolynomial<TF>;

    //
    Poly            run       ( const std::vector<TF> &x, const std::vector<TF> &y, const std::vector<TF> &ponderation = {} );

    // input parameters
    TF              precision;
    int             degp;

private:
    TF              error_with( const std::vector<TF> &x, const std::vector<TF> &y, const std::vector<TF> &p, std::size_t beg_index, std::size_t end_index ) const;
    std::vector<TF> make_poly ( const std::vector<TF> &x, const std::vector<TF> &y, const std::vector<TF> &p, std::size_t beg_index, std::size_t end_index ) const;
};

#include "UnivariatePiecewisePolynomialApproximator.tcc"
