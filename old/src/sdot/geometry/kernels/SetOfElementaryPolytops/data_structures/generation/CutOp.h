#ifndef SDOT_CUT_OP_H
#define SDOT_CUT_OP_H

#include "CutItem.h"
#include <string>

/**
*/
struct CutOp {
    using                TI               = CutItem::TI;

    std::string          mk_item_func_name() const;
    std::size_t          nb_output_shapes () const { return cut_items.size(); }
    std::size_t          nb_input_nodes   () const;
    std::size_t          nb_input_faces   () const;
    bool                 operator<        ( const CutOp &that ) const;
    operator             bool             () const;

    std::vector<CutItem> cut_items;
    std::size_t          dim;
};

#include "CutOp.tcc"

#endif // SDOT_CUT_OP_H
