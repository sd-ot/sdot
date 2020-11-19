#include "CutOp.h"

std::string CutOp::mk_item_func_name() const {
    std::string res;
    for( TI n = 0; n < cut_items.size(); ++n ) {
        // nodes
        res += std::to_string( cut_items[ n ].nodes.size() );
        for( auto inds : cut_items[ n ].nodes )
            res += " " + std::to_string( inds[ 0 ] ) + " " + std::to_string( inds[ 1 ] );
        // faces
        res += " " + std::to_string( cut_items[ n ].faces.size() );
        for( TI face : cut_items[ n ].faces ) {
            if ( face == TI( CutItem::cut_face ) )
                res += " -1";
            else if ( face == TI( CutItem::internal_face ) )
                res += " -2";
            else
                res += " " + std::to_string( face );
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

std::size_t CutOp::nb_input_nodes() const {
    using std::max;
    TI res = 0;
    for( const CutItem &ci : cut_items )
        for( auto inds : ci.nodes )
            for( auto ind : inds )
                res = max( res, ind + 1 );
    return res;
}

std::size_t CutOp::nb_input_faces() const {
    using std::max;
    TI res = 0;
    for( const CutItem &ci : cut_items )
        for( auto ind : ci.faces )
            if ( ind != TI( CutItem::internal_face ) && ind != TI( CutItem::cut_face ) )
                res = max( res, ind + 1 );
    return res;
}
