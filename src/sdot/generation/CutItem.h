#pragma once

#include <vector>
#include <tuple>
#include <array>

namespace sdot {

/**
*/
struct CutItem {
    using           TI        = std::size_t;
    using           NN        = std::array<TI,2>;

    bool            operator< ( const CutItem &that ) const { return std::tie( nodes, faces ) < std::tie( that.nodes, that.faces ); }

    std::vector<NN> nodes;    ///<
    std::vector<TI> faces;    ///< TI( -1 ) => take cut id
};

}
