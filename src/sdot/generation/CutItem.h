#pragma once

#include <vector>
#include <array>

namespace sdot {

/**
*/
struct CutItem {
    using           TI        = std::size_t;
    using           NN        = std::array<TI,2>;

    bool            operator< ( const CutItem &that ) const { return nodes < that.nodes; }

    std::vector<NN> nodes;    ///<
};

}
