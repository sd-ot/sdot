#pragma once

#include <string>
#include <vector>
#include <set>

namespace sdot {

/**
*/
class GlobGeneGeomData {
public:
    using                     TI          = std::size_t;

    std::string               mk_item_name( std::vector<TI> inds );

    std::set<std::vector<TI>> needed_cut_ops;
};

}
