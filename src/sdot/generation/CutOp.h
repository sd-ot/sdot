#pragma once

#include "CutItem.h"
#include <string>

namespace sdot {

/**
*/
struct CutOp {
    std::string          mk_item_func_name() const;
    std::size_t          nb_output_shapes () const { return cut_items.size(); }
    std::size_t          nb_input_nodes   () const;
    std::size_t          nb_input_faces   () const;
    bool                 operator<        ( const CutOp &that ) const;
    operator             bool             () const;

    std::vector<CutItem> cut_items;
    std::size_t          dim = 2;
};

} // namespace sdot
