#include "CutOp.h"
#include <tuple>
using TI = std::size_t;
using std::max;

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
    return std::tie( cut_items, dim ) < std::tie( that.cut_items, that.dim );
}

CutOp::operator bool() const {
    return cut_items.size();
}

std::size_t sdot::CutOp::nb_input_nodes() const {
    TI res = 0;
    for( const CutItem &ci : cut_items )
        for( TI ind : ci.node_inds )
            res = max( res, ind + 1 );
    return res;
}

} // namespace sdot
