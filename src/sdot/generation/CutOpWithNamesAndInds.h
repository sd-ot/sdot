#pragma once

#include "CutOp.h"

namespace sdot {

struct CutOpWithNamesAndInds {
    using            TI     = std::size_t;
    struct           Out    { std::string shape_name; std::vector<TI> inds; };

    std::vector<Out> outputs;
    std::vector<TI>  inputs;
    CutOp            cut_op;
};

}
