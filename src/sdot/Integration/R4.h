#pragma once

#include "../Support/N.h"
#include <cmath>

namespace sdot {
namespace FunctionEnum {

struct R4 {
    template<class PT,class TF>
    auto operator()( PT p, PT c, TF /*w*/ ) const {
        using std::pow;
        return pow( norm_2_p2( p - c ), 2 );
    }

    const char *name() const {
        return "R4";
    }

    auto func_for_final_cp_integration() const {
        return *this;
    }

    N<0> need_ball_cut() const {
        return {};
    }

    template<class TF,class TS>
    void span_for_viz( const TF&, TS ) const {}
};


}
}
