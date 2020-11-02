#pragma once

#include "NamedRecursivePolytop.h"
#include "CutOpWithNamesAndInds.h"

namespace sdot {

/**
*/
class CutCase {
public:
    using             TI            = std::size_t;
    struct            IndOut        { TI ind_0, ind_1, face_id; bool outside; void write_to_stream( std::ostream &os ) const { os << ind_0 << "_" << ind_1 << "_" << face_id << "_" << ( outside ? 'o' : 'i' ); } bool plain() const { return ind_0 == ind_1; } };
    using             Upp           = std::unique_ptr<CutOpWithNamesAndInds>;

    void              init          ( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );

    void              _init_2D      ( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );
    void              _init_2D_rec  ( CutOpWithNamesAndInds &possibility, const std::vector<IndOut> &points, const std::vector<NamedRecursivePolytop> &primitive_shapes );
    bool              _has_2D_shape ( TI nb_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );

    std::vector<Upp>  possibilities;
    std::vector<bool> out_points;
};

}
