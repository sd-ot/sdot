#pragma once

#include <vector>

namespace sdot {

/**
*/
struct CutItem {
    using           TI        = std::size_t;

    bool            operator< ( const CutItem &that ) const { return node_inds < that.node_inds; }

    std::vector<TI> node_inds;
};

}
