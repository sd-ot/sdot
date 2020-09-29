#pragma once

#include "../Support/N.h"

namespace sdot {
namespace FunctionEnum {

/*
*/
struct Unit {
    template<class PT,class TF>
    auto operator()( PT /*p*/, PT /*c*/, TF /*w*/ ) const {
        return 1;
    }

    const char *name() const {
        return "1";
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
