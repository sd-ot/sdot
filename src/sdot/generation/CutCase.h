#pragma once

#include "NamedRecursivePolytop.h"

namespace sdot {

/**
*/
class CutCase {
public:
    using             TI  = std::size_t;

    void              init( const RecursivePolytop &rp, const std::vector<bool> &out_points );

    bool              all_inside() const;
    TI                nb_created( std::string name ) const;

    std::vector<bool> out_points;
    TI                nb_new_edges;
};

}
