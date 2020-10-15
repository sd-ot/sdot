#include "CutOp.h"
using TI = std::size_t;

namespace sdot {

std::string CutOp::mk_item_func_name() const {
    std::string res = "mk_items";
    for( TI n = 0; n < cut_items.size(); ++n ) {
        if ( n )
            res += "_";
        for( TI i : cut_items[ n ].node_inds )
            res += "_" + std::to_string( i );
    }
    return res;

}

bool CutOp::operator<( const CutOp &that ) const {
    return cut_items < that.cut_items;
}

CutOp::operator bool() const {
    return cut_items.size();
}

} // namespace sdot
