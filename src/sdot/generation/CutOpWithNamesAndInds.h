#pragma once

#include "CutOp.h"

namespace sdot {

struct CutOpWithNamesAndInds {
    using            TI     = std::size_t;
    struct           Out    { std::string shape_name; std::vector<TI> output_node_inds, output_face_inds; };

    std::vector<TI>  input_node_inds;
    std::vector<TI>  input_face_inds;
    std::vector<Out> outputs;
    CutOp            cut_op;
};

}
