#pragma once

#include "../Support/N.h"
#include <functional>
#include <vector>
#include <memory>
#include <array>
#include <mutex>

/**
*/
namespace sdot {
namespace FunctionEnum {

/** "Any Radial Func" => radial func is given by function values.
 * A polynomial approximation
 */
struct Arfd {
    static constexpr unsigned nb_coeffs = 4;
    using TF = double;

    struct Approximation {
        std::array<TF,nb_coeffs> coeffs; ///< x^0, x^2, ...
        TF off; ///< contribution of the previous integration
        TF beg; ///< beg radius
        TF end; ///< end radius
    };

    Arfd( const std::function<TF( TF r )> &values, const std::function<TF( TF w )> &inp_scaling, const std::function<TF( TF w )> &out_scaling,
          const std::function<TF( TF r )> &der_values, const std::function<TF( TF w )> &der_inp_scaling, const std::function<TF( TF w )> &der_out_scaling, const std::vector<TF> &stops, TF prec = 1e-9 );
    Arfd();

    template<class PT,class TF>
    auto operator()( PT p, PT c, TF w ) const {
        TF i = inp_scaling ? inp_scaling( w ) : 1;
        TF o = out_scaling ? out_scaling( w ) : 1;
        return values( norm_2( p - c ) * i ) * o;
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
    TF approx_value( TF r ) const;

    std::size_t nb_polynomials() const { return approximations.size(); }

    //
    std::function<TF( TF w )> inp_scaling;
    std::function<TF( TF w )> out_scaling;
    std::function<TF( TF r )> values;
    std::vector<TF> stops;
    TF prec;

    std::unique_ptr<Arfd> der_w;

private:
    void _append_approx( TF &off, TF beg, TF end, unsigned nb_points = 100 ) const;

    mutable std::vector<Approximation> approximations;
    mutable std::mutex mutex;
};

}
}
