#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>
class ShapeType;

namespace parex {

/***/
template<class TF,class TI,int dim>
struct ShapeData {
    using            RNE                       = std::map<std::string,>;

    Tensor<TF>       coordinates;              ///< all the x for node 0, all the y for node 0, ... all the x for node 1, ...
    Tensor<TI>       face_ids;                 ///< all the ids for node 0, all the ids for node 1, ...
    Vec<TI>          ids;                      ///<

    mutable Value    reservation_new_elements; ///< map[ name elem ] => Vec[ nb element for each case ]
    mutable Value    cut_case_offsets;         ///< for each case, a vector with offsets of each sub case
    mutable Value    scalar_products;          ///< all the scalar products for node 0, all the scalar products for node 1, ...
    mutable Value    indices;                  ///<
};

} // namespace parex
