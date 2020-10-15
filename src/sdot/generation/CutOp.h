#pragma once

#include "CutItem.h"
#include <string>

namespace sdot {

/**
*/
struct CutOp {
    std::string          mk_item_func_name() const;
    std::size_t          nb_output_shapes () const { return cut_items.size(); }
    bool                 operator<        ( const CutOp &that ) const;
    operator             bool             () const;

    std::vector<CutItem> cut_items;
};

} // namespace sdot
