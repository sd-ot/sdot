#pragma once

#include "../Support/N.h"
#include <cmath>

namespace sdot {
namespace FunctionEnum {

/**
*/
template<class TS>
struct CompressibleFunc {
    template<class PT,class TF>
    TF operator()( PT p, PT c, TF w ) const {
        throw "not implemented";
        return 0;
    }

    const char *name() const {
        return "CompressibleFunc";
    }

    const CompressibleFunc &func_for_final_cp_integration() const {
        return *this;
    }

    N<0> need_ball_cut() const {
        return N<0>();
    }

    template<class TF>
    void span_for_viz( const TF &f, TS w ) const {
    }

    TS kappa, gamma, g, f_cor, pi_0, c_p;
};

}
}
