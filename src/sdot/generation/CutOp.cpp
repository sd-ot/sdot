#include "CutOp.h"
#include <tuple>
using TI = std::size_t;
using std::max;

namespace sdot {

std::string CutOp::mk_item_func_name() const {
    std::string res = "mk_items";
    for( TI n = 0; n < cut_items.size(); ++n ) {
        // nodes
        res += "_n" + std::to_string( cut_items[ n ].nodes.size() );
        for( auto inds : cut_items[ n ].nodes )
            res += "_" + std::to_string( inds[ 0 ] ) + "_" + std::to_string( inds[ 1 ] );
        // faces
        res += "_f" + std::to_string( cut_items[ n ].faces.size() );
        for( TI face : cut_items[ n ].faces ) {
            if ( face != TI( -1 ) )
                res += "_" + std::to_string( face );
            else
                res += "_c";
        }
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
        for( auto inds : ci.nodes )
            for( auto ind : inds )
                res = max( res, ind + 1 );
    return res;
}

std::size_t CutOp::nb_input_faces() const {
    TI res = 0;
    for( const CutItem &ci : cut_items )
        for( auto ind : ci.faces )
            res = max( res, ind + 1 );
    return res;
}

} // namespace sdot
