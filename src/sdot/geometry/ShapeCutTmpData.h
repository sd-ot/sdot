#pragma once

#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>
#include <string>

namespace sdot {

/***/
template<class TF,class TI>
struct ShapeCutTmpData {
    static std::string         type_name         () { return "sdot::ShapeCutTmpData<" + parex::type_name<TF>() + "," + parex::type_name<TI>() + ">"; }

    parex::Vec<parex::Vec<TI>> cut_case_offsets; ///< for each case, a vector with offsets of each sub case
    parex::Tensor<TF>          scalar_products;  ///< all the scalar products for node 0, all the scalar products for node 1, ...
    parex::Vec<TI>             indices;          ///<
};

} // namespace sdot

