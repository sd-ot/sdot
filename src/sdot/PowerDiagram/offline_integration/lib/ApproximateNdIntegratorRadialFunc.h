#pragma once

#include "PolyApproximator.h"

/**
  Integration of a radial func over a surface in 2 dimensions or a volume in 3 dimensions.

*/
class ApproximateNdIntegratorRadialFunc {
public:
    using            TF                               = PolyApproximator::TF;

    /*  */           ApproximateNdIntegratorRadialFunc( TF epsilon = 1e-10, int degp = 7 );

    void             run                              ( const std::vector<TF> &x, const std::vector<TF> &y ); ///< compute the coefficients given values of the radial function
    void             run                              ( const std::function<TF(TF)> &func,
                                                        TF beg_of_log_scale,
                                                        std::size_t nb_values_before_log_scale,
                                                        TF end_of_log_scale = 1e4,
                                                        std::size_t nb_values_after_log_scale = 1e3 ); ///< compute the coefficients given values of the radial function

    void             write_to_stream                  ( std::ostream &os ) const;
    TF               integrate                        ( const std::vector<std::array<TF,2>> &points ) const;
    void             generate                         ( std::ostream &os ) const;


    // input parameters
    int              continuity; ///< continuity level
    TF               epsilon;    ///< precision
    int              degp;       ///< degree of the polynomials

    //
    PolyApproximator coeffs;
};

