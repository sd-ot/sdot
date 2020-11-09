#pragma once

#include <parex/Value.h>
#include "ShapeType.h"

namespace sdot {

/**
*/
class ShapeData {
public:
    using                       Value             = parex::Value;
    using                       TI                = ShapeType::TI;

    /**/                        ShapeData         ( const ShapeType *shape_type, std::size_t dim, std::string scalar_type, std::string index_type );

    Value                       coordinates;      ///< all the x for node 0, all the y for node 0, ... all the x for node 1, ...
    const ShapeType*            shape_type;       ///< is it a quad, a pyramid, ... ?
    Value                       face_ids;         ///< all the ids for node 0, all the ids for node 1, ...
    Value                       ids;              ///<
};

}
