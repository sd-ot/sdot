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

/** "Any Radial Func". It makes a polynomial approximation of V( r ).
 *  Integration is then made on intersection of disks/balls with triangles/tetras made by boundaries and sphere_centers.
 *
 *  Arfd (which makes an approximation of f = "1/r^2 * int r V( r )") may be faster on cases where f can be approximated by polynomials with even (positive) degrees.
 */
struct Arf {
    static constexpr unsigned nb_coeffs = 4;
    using TF = double;

    struct Approximation {
        std::array<TF,nb_coeffs> coeffs; ///< sum_i coeffs_i * r^i is an approximation of value( r ) for r in [ beg, end ]
        bool last; ///< true if it is the last disc/sphere
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
    void _append_approx( TF beg, TF end, std::size_t nb_points = 100 ) const;

    mutable std::vector<Approximation> approximations;
    mutable std::mutex mutex;
};

}
}
