#pragma once

#include "NamedRecursivePolytop.h"
#include "CutOpWithNamesAndInds.h"

namespace sdot {

/**
*/
class CutCase {
public:
    using             TI            = std::size_t;
    struct            IndOut        { TI index; bool outside; };
    using             Upp           = std::unique_ptr<CutOpWithNamesAndInds>;

    void              init          ( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );

    void              _init_2D      ( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );
    void              _init_2D_rec  ( CutOpWithNamesAndInds &possibility, const std::vector<IndOut> &points, const std::vector<NamedRecursivePolytop> &primitive_shapes );
    bool              _has_2D_shape ( TI nb_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );

    std::vector<Upp>  possibilities;
    std::vector<bool> out_points;
};

}
