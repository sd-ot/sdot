#pragma once

#include "RecursivePolytop.h"
#include <string>

namespace sdot {
class GlobGeneGeomData;
class CutCase;

struct NamedRecursivePolytop {
    void             write_primitive_shape_incl( std::ostream &os ) const;
    void             write_primitive_shape_impl( std::ostream &os, GlobGeneGeomData &gggd, const std::vector<NamedRecursivePolytop> &available_primitive_shapes ) const;

    void             write_cut_ops             ( std::ostream &os, GlobGeneGeomData &gggd, std::vector<CutCase> &cut_cases ) const;

    RecursivePolytop polytop;
    std::string      name;
};

}
