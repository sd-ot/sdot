#pragma once

#include "NamedRecursivePolytop.h"
#include "CutOpWithNamesAndInds.h"

namespace sdot {

/**
*/
class CutCase {
public:
    using                 TI        = std::size_t;

    void                  init      ( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );

    TI                    nb_created( std::string name ) const;

    // input
    std::vector<bool>     out_points;

    // output
    TI                    nb_new_edges;
    CutOpWithNamesAndInds cownai;
};

}
