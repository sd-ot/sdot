#pragma once

#include "Unit.h"

namespace sdot {
namespace FunctionEnum {

struct InBallW05 {
    template<class PT,class TF>
    auto operator()( PT p, PT c, TF w ) const {
        return norm_2_p2( p - c ) <= w;
    }

    const char *name() const {
        return "InBallW05";
    }

    auto func_for_final_cp_integration() const {
        return Unit{};
    }

    N<1> need_ball_cut() const {
        return {};
    }

    template<class TF,class TS>
    void span_for_viz( const TF&, TS ) const {}
};

}
}
