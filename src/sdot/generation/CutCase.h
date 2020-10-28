#pragma once

#include "NamedRecursivePolytop.h"
#include "CutOpWithNamesAndInds.h"

namespace sdot {

/**
*/
class CutCase {
public:
    using             TI            = std::size_t;
    using             Possibilities = std::vector<CutOpWithNamesAndInds>;

    void              init          ( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );
    TI                nb_created    ( std::string name ) const;

    void              _init_2D      ( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );

    Possibilities     possibilities;
    std::vector<bool> out_points;
};

}
