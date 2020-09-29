#pragma once

#include "../Support/N.h"
#include <functional>
#include <vector>
#include <array>
#include <mutex>

/**
*/
namespace sdot {
namespace FunctionEnum {

/** "Any Radial Func" => radial func is given by its derivatives.
 *
 (2 c ArcTan[(b c - a d)/(a b + c d)] -
 2 c ArcTan[(b c - a d)/(a (a + b) + c (c + d))] +
 a (-Log[b^2 + d^2] + Log[(a + b)^2 + (c + d)^2]))/(2 (a^2 + c^2))
 */
struct Arfd {
    static constexpr unsigned nb_coeffs = 4;
    using TF = double;

    struct Approximation {
        std::array<TF,nb_coeffs+1> coeffs; ///< sum_i coeffs_i * r^( 2 * i - 2 ) is an approximation of value( r ) for r in [ beg, end ]
        TF beg; ///< beg radius
        TF end; ///< end radius
    };

    template<class PT,class TF>
    auto operator()( PT p, PT c, TF /*w*/ ) const {
        return values( norm_2( p - c ) );
    }

    const char *name() const {
        return "Arf";
    }

    const auto &func_for_final_cp_integration() const {
        return *this;
    }

    N<0> need_ball_cut() const {
        return {};
    }

    template<class TF,class TS>
    void span_for_viz( const TF&, TS ) const {}

    void make_approximations_if_not_done() const;
    const Approximation *approx_for( TF r ) const;

    //
    std::function<TF( TF w )> inp_scaling;
    std::function<TF( TF w )> out_scaling;
    std::function<TF( TF r )> values;
    std::vector<TF> stops;

private:
    void _append_approx( TF &sum, TF beg, TF end, unsigned nb_points = 100 ) const;

    mutable std::vector<Approximation> approximations;
    mutable std::mutex mutex;
};

}
}
