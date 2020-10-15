#pragma once

#include "NamedRecursivePolytop.h"
#include "CutOpWithNamesAndInds.h"

namespace sdot {

/**
*/
class CutCase {
public:
    using                TI        = std::size_t;
    using                CC        = CutOpWithNamesAndInds;

    void                 init      ( const RecursivePolytop &rp, const std::vector<bool> &out_points );

    TI                   nb_created( std::string name ) const;

    TI                   nb_new_edges;
    std::vector<bool>    out_points;
    CC                   cownai;
};

}
