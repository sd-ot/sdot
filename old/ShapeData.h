#pragma once

#include <parex/Value.h>
#include "ShapeType.h"

namespace sdot {

/**
*/
template<class TF,class TI,int dim>
class ShapeData {
public:
    using            Value                     = parex::Value;
    using            TI                        = ShapeType::TI;

    /**/             ShapeData                 ( const ShapeType *shape_type, std::size_t dim, std::string scalar_type, std::string index_type );

    Value            coordinates;              ///< all the x for node 0, all the y for node 0, ... all the x for node 1, ...
    const ShapeType* shape_type;               ///< is it a quad, a pyramid, ... ?
    Value            face_ids;                 ///< all the ids for node 0, all the ids for node 1, ...
    Value            ids;                      ///<

    mutable Value    reservation_new_elements; ///< map[ name elem ] => Vec[ nb element for each case ]
    mutable Value    cut_case_offsets;         ///< for each case, a vector with offsets of each sub case
    mutable Value    scalar_products;          ///< all the scalar products for node 0, all the scalar products for node 1, ...
    mutable Value    indices;                  ///<
};

}
