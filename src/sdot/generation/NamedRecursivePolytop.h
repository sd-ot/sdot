#pragma once

#include "RecursivePolytop.h"
#include <string>

namespace sdot {

struct NamedRecursivePolytop {
    void             write_primitive_shape_incl( std::ostream &os ) const;
    void             write_primitive_shape_impl( std::ostream &os, const std::vector<NamedRecursivePolytop> &available_primitive_shapes ) const;

    RecursivePolytop polytop;
    std::string      name;
};

}
