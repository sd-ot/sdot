#pragma once

#include "R2.h"

namespace sdot {
namespace FunctionEnum {

struct R2InBallW05 {
    template<class PT,class TF>
    auto operator()( PT p, PT c, TF w ) const {
        TF r2 = norm_2_p2( p - c );
        return ( r2 <= w ) * r2;
    }

    const char *name() const {
        return "R2InBallW05";
    }

    auto func_for_final_cp_integration() const {
        return R2{};
    }

    N<1> need_ball_cut() const {
        return {};
    }

    template<class TF,class TS>
    void span_for_viz( const TF&, TS ) const {}
};

}
}
